import logging
import os
import sys
from pathlib import Path


def setup_logger(name: str = "drone_ppo", run_name:str = "test") -> logging.Logger:
    logger = logging.getLogger(name)

    #TODO path fix
    log_file = f"C:/Prog/drone_drl/drone_learning/train/tb_logs/{run_name}/logs.log"

    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    file_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_formatter = logging.Formatter(
        fmt="%(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger