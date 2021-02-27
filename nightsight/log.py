import logging


# Setup logger
logger = logging.getLogger('nightsight')
logger.setLevel(logging.DEBUG)

# Create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# Create formatter
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')

# Add formatter to ch
ch.setFormatter(formatter)

# Add ch to logger
logger.addHandler(ch)


if __name__ == "__main__":
    # 'Test' logger
    logger.debug('debug message')
    logger.info('info message')
    logger.warning('warn message')
    logger.error('error message')
    logger.critical('critical message')

    # Check versions
    logger.info(f"Torch: {torch.__version__}, "
                f"Torchvision: {tv.__version__}, "
                f"Pytorch Lightning: {pl.__version__}, "
                f"albumentations: {A.__version__}")

