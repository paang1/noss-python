
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

static size_t difficulty = 21;  // 写死21

// Output string by the device read by host
char *g_out = nullptr;
unsigned char *g_hash_out = nullptr;
int *g_found = nullptr;

static uint64_t nonce = 0;
static uint64_t user_nonce = 0;
static uint64_t last_nonce_since_update = 0;

// Last timestamp we printed debug infos
static std::chrono::high_resolution_clock::time_point t_last_updated;

__device__ bool checkZeroPadding(unsigned char* sha, uint8_t difficulty) {

	bool isOdd = difficulty % 2 != 0;
	uint8_t max = (difficulty / 2) + 1;

	/*
		Odd : 00 00 01 need to check 0 -> 2
		Even : 00 00 00 1 need to check 0 -> 3
		odd : 5 / 2 = 2 => 2 + 1 = 3
		even : 6 / 2 = 3 => 3 + 1 = 4
	*/
	for (uint8_t cur_byte = 0; cur_byte < max; ++cur_byte) {
		uint8_t b = sha[cur_byte];
		if (cur_byte < max - 1) { // Before the last byte should be all zero
			if(b != 0) return false;
		}else if (isOdd) {
			if (b > 0x0F || b == 0) return false;
		}
		else if (b <= 0x0f) return false;
		
	}

	return true;

}

// Does the same as sprintf(char*, "%d%s", int, const char*) but a bit faster
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

void pre_sha256() {
	checkCudaErrors(cudaMemcpyToSymbol(dev_k, host_k, sizeof(host_k), 0, cudaMemcpyHostToDevice));
}

// Prints a 32 bytes sha256 to the hexadecimal form filled with zeroes
void print_hash(const unsigned char* sha256) {
	for (uint8_t i = 0; i < 32; ++i) {
		std::cout << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(sha256[i]);
	}
	std::cout << std::dec << std::endl;
}

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

int main() {

    cudaSetDevice(0);  // 默认CUDA

    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);  // CUDA缓存

    t_last_updated = std::chrono::high_resolution_clock::now();  // 当前时间

    std::string in;

    std::cout << "Enter a message : ";
    getline(std::cin, in);

    std::cout << "Nonce : ";
    std::cin >> user_nonce;

    // 固定21
    // std::cout << "Difficulty : ";
    // std::cin >> difficulty;
    // std::cout << std::endl;

    const size_t input_size = in.size();  // 获取输入消息的长度

    // Input string for the device
    char *d_in = nullptr;  // 定义一个指向设备内存的指针

    // Create the input string for the device
    cudaMalloc(&d_in, input_size + 1);  // 在设备上分配内存用于存储输入字符串
    cudaMemcpy(d_in, in.c_str(), input_size + 1, cudaMemcpyHostToDevice);  // 将输入字符串复制到设备内存中

    cudaMallocManaged(&g_out, input_size + 32 + 1);  // 分配托管内存用于存储输出字符串
    cudaMallocManaged(&g_hash_out, 32);  // 分配托管内存用于存储输出的哈希值
    cudaMallocManaged(&g_found, sizeof(int));  // 分配托管内存用于存储找到标志
    *g_found = 0;  // 初始化找到标志为0

    nonce += user_nonce;  // 将用户输入的nonce值加到全局nonce上
    last_nonce_since_update += user_nonce;  // 更新上次更新以来的nonce值

    pre_sha256();  // 调用预处理函数，初始化SHA256相关的数据

    size_t dynamic_shared_size = (ceil((input_size + 1) / 8.f) * 8) + (64 * BLOCK_SIZE);  // 计算动态共享内存的大小

    std::cout << "Shared memory is " << dynamic_shared_size / 1024 << "KB" << std::endl;  // 输出共享内存的大小

    while (!*g_found) {  // 当还没有找到符合条件的哈希值时循环
        sha256_kernel << < NUMBLOCKS, BLOCK_SIZE, dynamic_shared_size >> > (g_out, g_hash_out, g_found, d_in, input_size, difficulty, nonce);  // 调用核函数

        cudaError_t err = cudaDeviceSynchronize();  // 同步CUDA设备，等待核函数执行完成
        if (err != cudaSuccess) {  // 如果同步时发生错误
            throw std::runtime_error("Device error");  // 抛出异常
        }

        nonce += NUMBLOCKS * BLOCK_SIZE;  // 更新nonce值

        print_state();  // 打印当前状态
    }

    cudaFree(g_out);  // 释放输出字符串的内存
    cudaFree(g_hash_out);  // 释放哈希值的内存
    cudaFree(g_found);  // 释放找到标志的内存

    cudaFree(d_in);  // 释放设备上的输入字符串内存

    cudaDeviceReset();  // 重置CUDA设备

    return 0;  // 程序结束
}
