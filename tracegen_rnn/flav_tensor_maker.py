"""Make tensors for input and output of flav LSTM.

License:
    MIT License

    Copyright (c) 2021 HUAWEI CLOUD

"""

from datetime import datetime
import torch
from tracegen_rnn.constants import BOUND
from tracegen_rnn.utils.file_utils import get_flav_map

# How far to step for the range features, in seconds
RANGE_FEAT_STEP = 86400  # a day


class FlavTensorMaker():
    """Make tensors for inputs and outputs, as requested, given flavor
    map, which is used only to get list of flavors, which we turn into
    a mapping from flavors to indexes in the features tensor.

    """

    def __init__(self, flav_map_fn, range_start, range_stop,
                 range_idx=None):
        """The tensors depend on how many codes in the flav_map.

        Args:

        flav_map_fn: String, filename with map from flavors to their codes.

        range_start/stop: Int: timestamps for start/end of training data.

        range_idx: Int: If given, tensor maker will use this index in
        the range features (modulo the number of range features).

        """
        flavstr_map = get_flav_map(flav_map_fn)
        self.flav_idxs, self.idx_flavs = self.get_flav_idxs(flavstr_map)
        self.ninput = len(self.flav_idxs)
        # For timestamp feats:
        self.ninput += 24 + 7
        self.noutput = len(self.flav_idxs)
        self.range_start = range_start
        self.range_stop = range_stop
        self.range_idx = range_idx
        self.ninput += self.get_nrange_feats(
            self.range_start, self.range_stop)

    def get_ninput(self):
        """Getter for the ninput

        """
        return self.ninput

    def get_noutput(self):
        """Getter for the noutput

        """
        return self.noutput

    @staticmethod
    def get_flav_idxs(flavstr_map):
        """Return map from flavor to index, and one with reverse mapping,
        using values in flavstr_map.

        """
        flav_idxs = {}
        idx_flavs = {}
        flavs = list(flavstr_map.values())
        flavs.append(BOUND)
        for idx, flav in enumerate(flavs):
            flav_idxs[flav] = idx
            idx_flavs[idx] = flav
        return flav_idxs, idx_flavs

    def __one_hot_flav_line(self, line):
        """One-hot-encode line of flavs as a tensor of LEN(LINE) x 1 x NDIMS.

        """
        ndims = len(self.flav_idxs)
        fvals = torch.tensor([self.flav_idxs[f] for f in line])
        tensor = torch.nn.functional.one_hot(fvals,
                                             num_classes=ndims) \
                                    .unsqueeze(dim=1)
        return tensor

    @staticmethod
    def one_hots_timestamp(timestamp):
        """Encode both hour-of-day (from 1 to 24) and day-of-week (from 1 to
        7) using 1-hot ecoding and return 31-dimensional tensor.

        This is public so we can re-use it in other classes
        (e.g. features for Poisson Regression in narrivals).

        """
        # We don't know what REAL day it was, but even if back at
        # start of Linux, it's fine for finding patterns:
        ts_date = datetime.utcfromtimestamp(timestamp)
        weekday = ts_date.weekday()
        hour = ts_date.hour
        tensor = torch.zeros(1, 31)
        tensor[0][hour] = 1.0
        tensor[0][24 + weekday] = 1.0
        return tensor

    @staticmethod
    def get_nrange_feats(range_start, range_stop):
        """Return total number of range features.  Public so that clients that
        sample number of range features can determine range to sample from.

        """
        # There will be ones at zero AND at the quotient here:
        nfeats = 1 + (range_stop - range_start) // RANGE_FEAT_STEP
        return nfeats

    @classmethod
    def range_features(cls, range_start, range_stop, timestamp,
                       range_idx=None):
        """Encode each day of the range using features.

        range_idx: Int: If given, use this index in range features
        (modulo the number of range features).

        Note: public so we can re-use method in other classes
        (e.g. features for Poisson Regression in narrivals).

        """
        nfeats = cls.get_nrange_feats(range_start, range_stop)
        tensor = torch.zeros(1, nfeats)
        # Determine where we are in this range:
        if range_idx is not None:
            feat_num = range_idx % nfeats
        elif timestamp < range_start:
            feat_num = 0
        elif timestamp > range_stop:
            feat_num = nfeats - 1
        else:
            elapsed = timestamp - range_start
            feat_num = elapsed // RANGE_FEAT_STEP
        # Encode UP TO feat_num (survival-style):
        tensor[0][0:feat_num+1] = 1.0
        return tensor

    def encode_input(self, timestamp, line):
        """Given line of NFLAVS, output should be (NFLAVS-1) x 1 x NFEATURES
        [since last flav is not part of INPUT].

        """
        input_line = line[:-1]
        oneh_flavs = self.__one_hot_flav_line(input_line)
        # timestamp/range encoded once, then tiled:
        oneh_ts = self.one_hots_timestamp(timestamp)
        oneh_ts_line = oneh_ts.repeat(len(input_line), 1, 1)
        range_feats = self.range_features(
            self.range_start, self.range_stop, timestamp, self.range_idx)
        range_line = range_feats.repeat(len(input_line), 1, 1)
        return torch.cat([oneh_flavs, oneh_ts_line, range_line], dim=2)

    def encode_target(self, line):
        """For line of NFLAVS, return NFLAVS-1 targets giving indexes of the
        true flavs.

        """
        flav_idxs = [self.flav_idxs[flav] for flav in line[1:]]
        return torch.LongTensor(flav_idxs)

    def replace_flav_input(self, my_input, new_flav):
        """Replace existing encoding of flavor in given input, in place, with
        encoding of 'new_flav' instead.

        """
        nhot_flav_feats = len(self.flav_idxs)
        fval = torch.tensor([self.flav_idxs[new_flav]])
        new_flav_tensor = torch.nn.functional.one_hot(
            fval, num_classes=nhot_flav_feats)[0]
        my_input[0, 0, :nhot_flav_feats] = new_flav_tensor
