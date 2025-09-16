import logging
import sys
import os
from logging.handlers import RotatingFileHandler


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        os.makedirs("logs", exist_ok=True)
        console_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)


        file_handler = RotatingFileHandler("logs/app.log",
                                           maxBytes=5_000_000,
                                           backupCount=5)
        
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)


    return logger