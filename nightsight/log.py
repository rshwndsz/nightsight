import logging
from logging.config import dictConfig

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': True,
    'formatters': {
        'standard': {
            'format': '%(asctime)s %(filename)9s: %(levelname)8s %(message)s'
        },
    },
    'handlers': {
        'stdout': {
            'level': 'DEBUG',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout',  # Default is stderr
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'DEBUG',
            'formatter': 'standard',
            'filename': '.logs/LOG.log',
            'mode': 'a',
            'maxBytes': 10485760,
            'backupCount': 5,
        }
    },
    'loggers': {
        'nightsight': {
            'handlers': ['stdout', 'file'],
            'level': 'DEBUG',
            'propagate': True
        },
    }
}


class Logger:
    def __init__(self):
        dictConfig(LOGGING_CONFIG)
        self._logger = logging.getLogger('nightsight')

    def setLevel(self, level):
        self._logger.setLevel(level)

    def info(self, message, *args, **kwargs):
        self._logger.info(message, *args, **kwargs)

    def debug(self, message, *args, **kwargs):
        self._logger.debug(message, *args, **kwargs)

    def warning(self, message, *args, **kwargs):
        self._logger.warning(message, *args, **kwargs)

    def error(self, message, *args, **kwargs):
        self._logger.error(message, *args, **kwargs)

    def critical(self, message, *args, **kwargs):
        self._logger.critical(message, *args, **kwargs)
