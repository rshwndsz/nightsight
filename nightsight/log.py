import logging
from logging.config import dictConfig

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s %(filename)9s::%(funcName)12s %(levelname)8s %(message)s'
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
       '': {
            'handlers': ['stdout', 'file'],
            'level': 'DEBUG',
            'propagate': False 
        },
    }
}


def initLogger(lconfig=LOGGING_CONFIG):
    dictConfig(lconfig)


if __name__ == "__main__":
    initLogger()
