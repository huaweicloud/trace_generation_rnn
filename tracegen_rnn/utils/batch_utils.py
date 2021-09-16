"""Utilities for working with batches.

License:
    MIT License

    Copyright (c) 2021 HUAWEI CLOUD

"""
from itertools import groupby
from tracegen_rnn.constants import BOUND


def flavs_to_batchsize(flav_lst, batch_size_map, other_code):
    """Return flavors, but with every flavor replaced by batch size."""
    # Get all sizes:
    sizes = []
    for isbound, group in groupby(flav_lst, lambda x: x == BOUND):
        if isbound:
            continue
        sizes.append(len(list(group)))
    # Now, assign each flav a batch size and increment after each bound:
    size_idx = 0
    batches = []
    prev_flav = None
    for flav in flav_lst:
        if flav == BOUND and prev_flav is not None and prev_flav != BOUND:
            batches.append(BOUND)
            size_idx += 1
        else:
            out_code = batch_size_map.get(sizes[size_idx], other_code)
            batches.append(out_code)
        prev_flav = flav
    return batches
