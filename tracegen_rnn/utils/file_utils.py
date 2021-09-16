"""Utilities for reading helper files.

License:
    MIT License

    Copyright (c) 2021 HUAWEI CLOUD

"""
ITEM_DATA_SEP = ","
TRACE_DATA_SEP = " "
MAP_SEP = " "


def get_flav_map(flav_map_fn):
    """Create a mapping from flavors to letters using the given map."""
    flav_map = {}
    with open(flav_map_fn) as flav_map_file:
        for line in flav_map_file:
            line = line.rstrip("\n")
            flav, code = line.split(MAP_SEP)
            flav_map[flav] = code
    return flav_map


def read_interval_map(map_fn):
    """Read and returnf interval map and number of intervals (including 1
    for the one extra (that goes to infinity) that's not in the list).

    """
    interval_map = {}
    with open(map_fn) as map_file:
        for line in map_file:
            line = line.rstrip()
            dur, interval = line.split(MAP_SEP)
            interval_map[int(dur)] = int(interval)
    nintervals = len(set(interval_map.values())) + 1
    return interval_map, nintervals


def get_batch_size_map(batch_size_map_fn):
    """Create a mapping from batch sizes to codes using the given map."""
    batch_size_map = {}
    other_code = None
    with open(batch_size_map_fn) as batch_size_map_file:
        for line in batch_size_map_file:
            line = line.rstrip("\n")
            batch, code = line.split(MAP_SEP)
            try:
                batch_size_map[int(batch)] = code
            except ValueError:
                # It's the 'other' category:
                other_code = code
    return batch_size_map, other_code


def yield_trace_lines(trace_fn):
    """Read and yield data from the trace line-by-line: for either
    flavors, or durations.

    """
    with open(trace_fn) as trace_file:
        for line in trace_file:
            line = line.rstrip('\n')
            timestamp, itemstr = line.split(TRACE_DATA_SEP)
            items = itemstr.split(ITEM_DATA_SEP)
            yield timestamp, items


def encode_trace_line(timestamp, item_lst):
    itemstr = ITEM_DATA_SEP.join(item_lst)
    out_str = "{}{}{}".format(timestamp, TRACE_DATA_SEP, itemstr)
    return out_str
