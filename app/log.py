import logging
from config import CONFIG

def setup_custom_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.setLevel(CONFIG['general']['log_level'].upper())
        logger.addHandler(handler)
    return logger
