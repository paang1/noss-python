import re
import requests
import subprocess
from loguru import logger


def get_init_hex() -> str:
    init_pattern = r'init\(\"([a-z0-9]+)\"\)'
    res = requests.get("https://noscription.org/_next/static/chunks/pages/index-2b8a738e2190676c.js")
    match = re.search(init_pattern, res.text)
    if match:
        logger.success("成功找到init hex")
        return match.group(1)
    else:
        logger.warning("未找到init hex, 可能是url地址被更改")


def get_x_gorgon(init_hex: str, event_id: str) -> str:
    command = f"node js/a.js {init_hex} {event_id}"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # 获取命令的输出和错误信息
    output, error = process.communicate()

    if error:
        error_message = error.decode('utf-8', 'replace')  # 使用 'replace' 选项
        logger.error(f"获取X-Gorgon失败\n\t状态码:{process.returncode}\n\t错误信息:{error_message}")

    return output.decode('utf-8', 'ignore').strip()
