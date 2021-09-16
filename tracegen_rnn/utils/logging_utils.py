"""Help set up a console logger.

License:
    MIT License

    Copyright (c) 2021 HUAWEI CLOUD

"""
import logging
LOG_FORMAT = "%(asctime)s [%(name)-10.35s] [%(levelname)-5.5s]  %(message)s"


def init_console_logger(logger_levels):
    """Configure the loggers to write to STDOUT.

    """
    log_formatter = logging.Formatter(LOG_FORMAT)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    for loggername, loglevel in logger_levels:
        mylogger = logging.getLogger(loggername)
        mylogger.setLevel(loglevel)
        mylogger.addHandler(console_handler)
        mylogger.propagate = False
