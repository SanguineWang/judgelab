from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path


class JudgeLabLauncher:
    def __init__(self, app_path: Path, address: str = "127.0.0.1", port: int = 8501, log_level: str = "debug") -> None:
        self.app_path = app_path
        self.address = address
        self.port = port
        self.log_level = log_level.lower()
        self.logger = logging.getLogger("judgelab.launcher")

    def run(self) -> int:
        if not self.app_path.exists():
            raise FileNotFoundError(f"找不到应用入口文件: {self.app_path}")

        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")

        command = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(self.app_path),
            f"--server.address={self.address}",
            f"--server.port={self.port}",
            f"--logger.level={self.log_level}",
        ]

        self.logger.info("启动 JudgeLab")
        self.logger.info("应用入口: %s", self.app_path)
        self.logger.info("监听地址: http://%s:%s", self.address, self.port)
        self.logger.info("日志级别: %s", self.log_level.upper())
        self.logger.debug("执行命令: %s", " ".join(command))

        process = subprocess.run(command, cwd=str(self.app_path.parent), env=env, check=False)
        self.logger.info("JudgeLab 已退出，返回码: %s", process.returncode)
        return int(process.returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyCharm 友好的 JudgeLab 启动入口")
    parser.add_argument("--address", default="127.0.0.1", help="Streamlit 监听地址，默认 127.0.0.1")
    parser.add_argument("--port", type=int, default=8501, help="Streamlit 监听端口，默认 8501")
    parser.add_argument(
        "--log-level",
        default="debug",
        choices=["error", "warning", "info", "debug"],
        help="Streamlit 日志级别，默认 debug",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main() -> int:
    configure_logging()
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    launcher = JudgeLabLauncher(
        app_path=project_root / "app.py",
        address=str(args.address),
        port=int(args.port),
        log_level=str(args.log_level),
    )
    try:
        return launcher.run()
    except KeyboardInterrupt:
        logging.getLogger("judgelab.launcher").warning("收到中断信号，停止 JudgeLab。")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
