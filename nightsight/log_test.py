import os
import logging
import pytest
from nightsight.log import Logger


def test_logger(caplog):
    logger = Logger('nightsight')
    logger.debug("Testing logger with DEBUG")
    assert "Testing logger with DEBUG" in open('./LOG.log').read()
