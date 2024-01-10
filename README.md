## 使用GPU加速计算nip-13算法

本文档旨在指导用户如何使用GPU加速计算nip-13算法。以下是详细的安装和使用指南。

### 基础项目
本项目基于以下项目开发：
- [noss](https://github.com/maxiaoxiong/noss) - 作者：[maxiaoxiong](https://github.com/maxiaoxiong/)
- [SHA256CUDA](https://github.com/moffa13/SHA256CUDA) - 作者：[moffa13](https://github.com/moffa13/)

### 安装步骤
#### 对于Windows用户：
1. 安装 Python (版本3.11或更高): [下载Python](https://www.python.org/ftp/python/3.12.0/python-3.12.0-amd64.exe)
2. 安装 Node.js: [下载Node](https://nodejs.org/dist/v20.10.0/node-v20.10.0-x64.msi)
3. 安装 CUDA Toolkit: [下载CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

### 安装完成后的测试
完成安装后，请在项目根目录下进行以下测试：

运行命令：
```bash
.\hash.exe 任意字符串 任意数字
```
例如：
```bash
.\hash.exe rhger 6565
```
如果输出的字符串以 `00000` 开头，表示程序运行正常。
![示例输出.png](https://github.com/Cloxl/noss-python/blob/main/docs/example.png)  
接下来，打开 `key.txt` 文件，输入您的私钥输入下列命令安装环境并且运行程序  
```bash
pip install -r requirements.txt
python main.py
```
### 如果无法使用
如果上述测试未通过，请按以下步骤操作：

1. 确保已安装 C++ Build Tools。
2. 进入 SHA256CUDA 文件夹
3. 运行以下命令以编译CUDA程序，用于计算SHA256：
   ```bash
   nvcc -O2 -o hash main.cu
   ```
完成后，您应该能够正常运行程序。

### 常见问题
**Q: 为什么不将cu文件编译为Python扩展？**

**A:** 由于时间限制，目前没有将cu文件编译为Python扩展。不过，即使使用subprocess调用，对于noss项目来说也已经足够。

如果有任何疑问 请联系我: cloxl@cloxl.com