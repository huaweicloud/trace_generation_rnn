"""Shared constants.

License:
    MIT License

    Copyright (c) 2021 HUAWEI CLOUD

"""
from enum import Enum

# A special boundary marker for either BOS or EOS of a SUBSCRIPTION.
BOUND = "|"


class ExampleKeys(Enum):
    """Keys we can use to extract values from an example."""
    INPUT = "input"
    TARGET = "target"
    OUT_MASK = "mask"


class CensorChar(Enum):
    """Characters we use to encode censoring situation."""
    CENSORED = "C"
    UNCENSORED = "U"
