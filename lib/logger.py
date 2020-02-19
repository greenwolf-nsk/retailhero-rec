import logging
import os

from colorama import Fore


def configure_logger(logger_name: str, logger_level=logging.INFO, log_dir: str = '/tmp'):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logger_level)
    formatter = logging.Formatter(f'{Fore.LIGHTBLUE_EX}%(asctime)s{Fore.GREEN}  %(message)s{Fore.RESET}')
    file_formatter = logging.Formatter(f'%(asctime)s %(message)s')

    log_file = os.path.join(log_dir, f'{logger_name}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logger_level)
    file_handler.setFormatter(file_formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logger_level)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.info(f'Logging to {log_file}')

    return logger
