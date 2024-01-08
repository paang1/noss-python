@echo off
echo 正在安装依赖...
pip install -r requirements.txt

echo 检查key.txt文件...
if not exist key.txt (
    echo key.txt文件不存在，正在创建文件...
    type nul > key.txt
    echo key.txt文件已创建，请写入私钥。
    goto end
)

for %%I in (key.txt) do set size=%%~zI
if %size%==0 (
    echo key.txt文件为空，请写入私钥。
) else (
    echo key.txt文件检查完毕。
)

:end
pause
