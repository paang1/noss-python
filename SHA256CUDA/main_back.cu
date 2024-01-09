#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <chrono>
#include <cmath>
#include <thread>
#include <iomanip>
#include <string>
#include <cassert>
#include <cstring>

#include "main.h"
#include "sha256.cuh"

#define SHOW_INTERVAL_MS 2000
#define BLOCK_SIZE 256
#define SHA_PER_ITERATIONS 8'388'608
#define NUMBLOCKS (SHA_PER_ITERATIONS + BLOCK_SIZE - 1) / BLOCK_SIZE

static size_t difficulty = 16;

// Output string by the device read by host
char *g_out = nullptr;
unsigned char *g_hash_out = nullptr;
char *d_in = nullptr;
int *g_found = nullptr;
size_t input_size;
static uint64_t nonce = 0;
static uint64_t user_nonce = 0;
static uint64_t last_nonce_since_update = 0;

// Last timestamp we printed debug infos
static std::chrono::high_resolution_clock::time_point t_last_updated;

__device__ size_t my_strlen(const char *str) {
    size_t len = 0;
    while (str[len] != '\0') {
        len++;
    }
    return len;
}

__device__ int my_strtol(const char *str, char **endptr, int base) {
    int num = 0;
    int sign = 1;
    int i = 0;

    // 处理空白字符
    while (str[i] == ' ' || str[i] == '\t') {
        i++;
    }

    // 处理符号
    if (str[i] == '-' || str[i] == '+') {
        sign = (str[i] == '-') ? -1 : 1;
        i++;
    }

    // 转换数字
    while (str[i] >= '0' && str[i] <= '9') {
        num = num * base + (str[i] - '0');
        i++;
    }

    if (endptr) {
        *endptr = (char *)&str[i];
    }

    return num * sign;
}

__device__ int hexCharToInt(char ch) {
    if (ch >= '0' && ch <= '9') {
        return ch - '0';
    } else if (ch >= 'a' && ch <= 'f') {
        return 10 + ch - 'a';
    } else if (ch >= 'A' && ch <= 'F') {
        return 10 + ch - 'A';
    } else {
        return 0;
    }
}

__device__ int countLeadingZeroes(const char *hex) {
    int count = 0;

    for (int i = 0; i < my_strlen(hex); i++) {
        char tempStr[3] = {hex[i], '\0', '\0'};
        int nibble = hexCharToInt(tempStr[0]);
        if (nibble == 0) {
            count += 4;
        } else {
            // 这里需要一个适合CUDA的__builtin_clz替代
            // CUDA不支持标准__builtin_clz，这里提供一个简单的替代实现
            int leadingZeros = 0;
            while (nibble > 0) {
                nibble >>= 1;
                leadingZeros++;
            }
            count += 4 - leadingZeros;
            break;
        }
    }

    return count;
}

__device__ void bytesToHexStr(const unsigned char *bytes, size_t bytesLen, char *hexStr) {
    const char hexChars[] = "0123456789abcdef";
    for (size_t i = 0; i < bytesLen; i++) {
        hexStr[i * 2] = hexChars[(bytes[i] >> 4) & 0x0F];
        hexStr[i * 2 + 1] = hexChars[bytes[i] & 0x0F];
    }
    hexStr[bytesLen * 2] = '\0';
}

__device__ bool checkZeroPadding(unsigned char* sha, size_t shaLen) {
    char hexStr[256];
    bytesToHexStr(sha, shaLen, hexStr);

    int leadingZeroes = countLeadingZeroes(hexStr);
    return leadingZeroes == 21;
}

// 检查前导0
//__device__ bool checkZeroPadding(unsigned char* sha, uint8_t difficulty) {
//
//	bool isOdd = difficulty % 2 != 0;
//	uint8_t max = (difficulty / 2) + 1;
//
//	/*
//		Odd : 00 00 01 need to check 0 -> 2
//		Even : 00 00 00 1 need to check 0 -> 3
//		odd : 5 / 2 = 2 => 2 + 1 = 3
//		even : 6 / 2 = 3 => 3 + 1 = 4
//	*/
//	for (uint8_t cur_byte = 0; cur_byte < max; ++cur_byte) {
//		uint8_t b = sha[cur_byte];
//		if (cur_byte < max - 1) { // Before the last byte should be all zero
//			if(b != 0) return false;
//		}else if (isOdd) {
//			if (b > 0x0F || b == 0) return false;
//		}
//		else if (b <= 0x0f) return false;
//
//	}
//
//	return true;
//
//}

// nonce转str
__device__ uint8_t nonce_to_str(uint64_t nonce, unsigned char* out) {
	uint64_t result = nonce;
	uint8_t remainder;
	uint8_t nonce_size = nonce == 0 ? 1 : floor(log10((double)nonce)) + 1;
	uint8_t i = nonce_size;
	while (result >= 10) {
		remainder = result % 10;
		result /= 10;
		out[--i] = remainder + '0';
	}

	out[0] = result + '0';
	i = nonce_size;
	out[i] = 0;
	return i;
}


extern __shared__ char array[];
__global__ void sha256_kernel(char* out_input_string_nonce, unsigned char* out_found_hash, int *out_found, const char* in_input_string, size_t in_input_string_size, uint8_t difficulty, uint64_t nonce_offset) {

	// If this is the first thread of the block, init the input string in shared memory
	char* in = (char*) &array[0];
	if (threadIdx.x == 0) {
		memcpy(in, in_input_string, in_input_string_size + 1);
	}

	__syncthreads(); // Ensure the input string has been written in SMEM

	uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	uint64_t nonce = idx + nonce_offset;

	// The first byte we can write because there is the input string at the begining
	// Respects the memory padding of 8 bit (char).
	size_t const minArray = static_cast<size_t>(ceil((in_input_string_size + 1) / 8.f) * 8);

	uintptr_t sha_addr = threadIdx.x * (64) + minArray;
	uintptr_t nonce_addr = sha_addr + 32;

	unsigned char* sha = (unsigned char*)&array[sha_addr];
	unsigned char* out = (unsigned char*)&array[nonce_addr];
	memset(out, 0, 32);

	uint8_t size = nonce_to_str(nonce, out);

	assert(size <= 32);

	{
		unsigned char tmp[32];

		SHA256_CTX ctx;
		sha256_init(&ctx);
		sha256_update(&ctx, out, size);
		sha256_update(&ctx, (unsigned char *)in, in_input_string_size);
		sha256_final(&ctx, tmp);

		// Second round of SHA256
		sha256_init(&ctx);
		sha256_update(&ctx, tmp, 32);
		sha256_final(&ctx, sha);
	}

	if (checkZeroPadding(sha, difficulty) && atomicExch(out_found, 1) == 0) {
		memcpy(out_found_hash, sha, 32);
		memcpy(out_input_string_nonce, out, size);
		memcpy(out_input_string_nonce + size, in, in_input_string_size + 1);
	}
}

// 计算前的准备工作
void pre_sha256() {
	checkCudaErrors(cudaMemcpyToSymbol(dev_k, host_k, sizeof(host_k), 0, cudaMemcpyHostToDevice));
}

// 打印16进制hash值
void print_hash(const unsigned char* sha256) {
	for (uint8_t i = 0; i < 32; ++i) {
		std::cout << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(sha256[i]);
	}
	std::cout << std::dec << std::endl;
}

// 打印当前状态
void print_state() {
	std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double, std::milli> last_show_interval = t2 - t_last_updated;

	if (last_show_interval.count() > SHOW_INTERVAL_MS) {
		std::chrono::duration<double, std::milli> span = t2 - t_last_updated;
		float ratio = span.count() / 1000;
		std::cout << span.count() << " " << nonce - last_nonce_since_update << std::endl;

		std::cout << std::fixed << static_cast<uint64_t>((nonce - last_nonce_since_update) / ratio) << " hash(es)/s" << std::endl;


		std::cout << std::fixed << "Nonce : " << nonce << std::endl;

		t_last_updated = std::chrono::high_resolution_clock::now();
		last_nonce_since_update = nonce;
	}

	if (*g_found) {
		std::cout << g_out << std::endl;
		print_hash(g_hash_out);
	}
}


void init(const std::string& in) {
    cudaSetDevice(0);
    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

    t_last_updated = std::chrono::high_resolution_clock::now();

    input_size = in.size();  // 输入字符串的长度

    // 创建设备的输入字符串
    cudaMalloc(&d_in, input_size + 1);
    cudaMemcpy(d_in, in.c_str(), input_size + 1, cudaMemcpyHostToDevice);

    cudaMallocManaged(&g_out, input_size + 32 + 1);
    cudaMallocManaged(&g_hash_out, 32);
    cudaMallocManaged(&g_found, sizeof(int));
    *g_found = 0;  // 初始化结束标志

    pre_sha256();
}

// SHA-256处理函数
void process_sha256(const std::string& in, uint64_t user_nonce) {
    nonce += user_nonce;
    last_nonce_since_update += user_nonce;

    size_t dynamic_shared_size = (ceil((input_size + 1) / 8.f) * 8) + (64 * BLOCK_SIZE);
//	std::cout << "share memory: " << dynamic_shared_size / 1024 << "KB" << std::endl;

    while (!*g_found) {
        sha256_kernel<<<NUMBLOCKS, BLOCK_SIZE, dynamic_shared_size>>>(g_out, g_hash_out, g_found, d_in, input_size, difficulty, nonce);

        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            throw std::runtime_error("设备异常");
        }

        nonce += NUMBLOCKS * BLOCK_SIZE;

        print_state();
    }
}

// 关闭函数
void close() {
    cudaFree(g_out);
    cudaFree(g_hash_out);
    cudaFree(g_found);
    cudaFree(d_in);
    cudaDeviceReset();
}

int main() {
    uint64_t user_nonce = 12756;
    std::string in = "["
     "  0,"
     "  \"0b79a9aca45603f873948c6682ec6373dd6997e69854c0a09df948a1b454bfcb\","
     "  1704758771,"
     "  1,"
     "  '[[\"p\",\"9be107b0d7218c67b4954ee3e6bd9e4dba06ef937a93f684e42f730a0c3d053c\"],[\"e\",\"51ed7939a984edee863bfbb2e66fdc80436b000a8ddca442d83e6a2bf1636a95\",\"wss://relay.noscription.org/\",\"root\"],[\"e\",\"00000647c189220f934fd2b72d345d3d109a75654f0562b77a426fb206150ce1\",\"wss://relay.noscription.org/\",\"reply\"],[\"seq_witness\",\"168501742\",\"0x0adfc5ea0b8041d528d47cbb15a85cb959d3f6dbb7b06847277a91a0eac0ff82\"],[\"nonce\",\"12756\",\"21\"]]',"
     "  \"{\\\"p\\\":\\\"nrc-20\\\",\\\"op\\\":\\\"mint\\\",\\\"tick\\\":\\\"noss\\\",\\\"amt\\\":\\\"10\\\"}\""
     "]";
    init(in);
    process_sha256(in, user_nonce);
    close();

    return 0;
}

//int main() {
//
//	cudaSetDevice(0);
//	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
//
//	t_last_updated = std::chrono::high_resolution_clock::now();
//
//	std::string in;
//
//	std::cout << "Enter a message : ";
//	getline(std::cin, in);
//    in = "["
//     "  0,"
//     "  \"0b79a9aca45603f873948c6682ec6373dd6997e69854c0a09df948a1b454bfcb\","
//     "  1704758771,"
//     "  1,"
//     "  '[[\"p\",\"9be107b0d7218c67b4954ee3e6bd9e4dba06ef937a93f684e42f730a0c3d053c\"],[\"e\",\"51ed7939a984edee863bfbb2e66fdc80436b000a8ddca442d83e6a2bf1636a95\",\"wss://relay.noscription.org/\",\"root\"],[\"e\",\"00000647c189220f934fd2b72d345d3d109a75654f0562b77a426fb206150ce1\",\"wss://relay.noscription.org/\",\"reply\"],[\"seq_witness\",\"168501742\",\"0x0adfc5ea0b8041d528d47cbb15a85cb959d3f6dbb7b06847277a91a0eac0ff82\"],[\"nonce\",\"12756\",\"21\"]]',"
//     "  \"{\\\"p\\\":\\\"nrc-20\\\",\\\"op\\\":\\\"mint\\\",\\\"tick\\\":\\\"noss\\\",\\\"amt\\\":\\\"10\\\"}\""
//     "]";
//;
//    user_nonce = 12756;
//	std::cout << "Nonce : ";
//	std::cin >> user_nonce;
//
//写死为21
//	std::cout << "Difficulty : ";
//	std::cin >> difficulty;
//	std::cout << std::endl;
//
//
//	const size_t input_size = in.size();  // 输入字符串的长度
//
//	// 设备的输入字符串
//	char *d_in = nullptr;
//
//	// 创建设备的输入字符串
//	cudaMalloc(&d_in, input_size + 1);
//	cudaMemcpy(d_in, in.c_str(), input_size + 1, cudaMemcpyHostToDevice);
//
//	cudaMallocManaged(&g_out, input_size + 32 + 1);
//	cudaMallocManaged(&g_hash_out, 32);
//	cudaMallocManaged(&g_found, sizeof(int));
//	*g_found = 0;  // 初始化结束标志
//
//	nonce += user_nonce;
//	last_nonce_since_update += user_nonce;
//
//	pre_sha256();
// 共享内存
//	size_t dynamic_shared_size = (ceil((input_size + 1) / 8.f) * 8) + (64 * BLOCK_SIZE);
//	std::cout << "share memory: " << dynamic_shared_size / 1024 << "KB" << std::endl;
//
//	while (!*g_found) {
//		sha256_kernel << < NUMBLOCKS, BLOCK_SIZE, dynamic_shared_size >> > (g_out, g_hash_out, g_found, d_in, input_size, difficulty, nonce);
//
//		cudaError_t err = cudaDeviceSynchronize();
//		if (err != cudaSuccess) {
//			throw std::runtime_error("设备异常");
//		}
//
//		nonce += NUMBLOCKS * BLOCK_SIZE;
//
//		print_state();
//	}
//
//
//	cudaFree(g_out);
//	cudaFree(g_hash_out);
//	cudaFree(g_found);
//
//	cudaFree(d_in);
//
//	cudaDeviceReset();
//
//	return 0;
//}