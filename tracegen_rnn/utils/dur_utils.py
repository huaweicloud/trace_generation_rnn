"""Utilities for working with durations.

License:
    MIT License

    Copyright (c) 2021 HUAWEI CLOUD

"""
import torch
from tracegen_rnn.constants import CensorChar, BOUND


def encode_dur_str(interval, censored):
    """Create the duration output symbol."""
    censor_char = CensorChar.CENSORED.value \
        if censored else CensorChar.UNCENSORED.value
    out_str = "{}{}".format(censor_char, interval)
    return out_str


def decode_dur_str(dur_str):
    """Parse a dur_str and return censor_char and interval (as integer).

    """
    if dur_str == BOUND:
        return None, None
    censor_char = CensorChar(dur_str[0])
    interval = int(dur_str[1:])
    return censor_char, interval


def encode_dur_slice(dur_tensor, nintervals, idx, cchar, itvl):
    """Encode one slice of the dur tensor, i.e., at one offset in the
    sequence list (idx).

    """
    # Identify boundaries at feature 0:
    if itvl is None:
        dur_tensor[idx][0] = 1
    else:
        # Mark where we know you survived:
        offset = 1  # skipping first feature BOS flag:
        curr = dur_tensor[idx][offset:]
        curr[:itvl + 1] = 1
        # Now mark where we know you died, if we know:
        if cchar == CensorChar.UNCENSORED:
            offset = 1 + nintervals  # skipping the surv feats too
            curr = dur_tensor[idx][offset:]
            curr[itvl:] = 1


def precreate_dur_embedding(nintervals):
    """Precreate 1 + 2*nintervals different encodings of durations
    (depending on whether item is a boundary, or whether each interval
    is censored or not).

    """
    nencodes = 1 + nintervals * 2
    # Each possiblity also corresponds to number of features:
    ndur_feats = nencodes
    dur_tensor = torch.zeros(nencodes, ndur_feats)
    # Do bound:
    encode_dur_slice(dur_tensor, nintervals, 0, None, None)
    # Next, uncensored, then censored ones:
    offset = 1
    for idx in range(nintervals):
        encode_dur_slice(dur_tensor, nintervals, idx + offset,
                         CensorChar.UNCENSORED, idx)
    offset = 1 + nintervals
    for idx in range(nintervals):
        encode_dur_slice(dur_tensor, nintervals, idx + offset,
                         CensorChar.CENSORED, idx)
    return dur_tensor


def precreate_targ_mask_encoding(nintervals):
    """Precreate 1 + 2*nintervals different target/mask encodings
    (depending whether item is a boundary, or whether each interval is
    censored or not).

    """
    nencodes = 1 + nintervals * 2
    targ_tensor = torch.zeros(nencodes, nintervals)
    mask_tensor = torch.zeros(nencodes, nintervals)
    # For bounds, leave both all-zeros
    # Next uncensored then censored ones:
    offset = 1
    for idx in range(nintervals):
        # Mark point where item suffered hazard:
        targ_tensor[idx + offset][idx] = 1
        # Include up to and incl. this one in the loss:
        mask_tensor[idx + offset][:idx+1] = 1
    offset = 1 + nintervals
    for idx in range(nintervals):
        # No hazard seen, but include survival points (leave target
        # values as zero) via the mask:
        mask_tensor[idx + offset][:idx] = 1
    return targ_tensor, mask_tensor


def durstr_to_embed_idx(dur_str, nintervals):
    """Convert duration string to index into precreated embeddings
    (whether features or target/mask encodings).

    """
    cchar, itvl = decode_dur_str(dur_str)
    if itvl is None:
        return 0
    offset = 1
    if cchar == CensorChar.CENSORED:
        offset += nintervals
    return offset + itvl
