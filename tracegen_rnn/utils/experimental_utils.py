"""Utilities related to experiments.

License:
    MIT License

    Copyright (c) 2021 HUAWEI CLOUD

"""
import os
import random
import numpy as np
import torch


def set_all_seeds(seed_value):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
