@echo off
echo ���ڰ�װ����...
pip install -r requirements.txt

echo ���key.txt�ļ�...
if not exist key.txt (
    echo key.txt�ļ������ڣ����ڴ����ļ�...
    type nul > key.txt
    echo key.txt�ļ��Ѵ�������д��˽Կ��
    goto end
)

for %%I in (key.txt) do set size=%%~zI
if %size%==0 (
    echo key.txt�ļ�Ϊ�գ���д��˽Կ��
) else (
    echo key.txt�ļ������ϡ�
)

:end
pause
