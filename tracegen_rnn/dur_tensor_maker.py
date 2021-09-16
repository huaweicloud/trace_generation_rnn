"""Make tensors for the input and output for the duration sequence.
Input encodes current flavor and previous duration.  Output encodes a
vector of hazard outputs, and another vector that's a mask over
targets telling which hazards to care about.

License:
    MIT License

    Copyright (c) 2021 HUAWEI CLOUD

"""
from itertools import groupby
import torch
from tracegen_rnn.constants import BOUND
from tracegen_rnn.flav_tensor_maker import FlavTensorMaker
from tracegen_rnn.utils.batch_utils import flavs_to_batchsize
from tracegen_rnn.utils.file_utils import \
    read_interval_map, get_batch_size_map
from tracegen_rnn.utils.dur_utils import \
    durstr_to_embed_idx, precreate_dur_embedding, precreate_targ_mask_encoding


class DurTensorMaker():
    """Make tensors for duration inputs and outputs.

    """

    def __init__(self, flav_map_fn, interval_map_fn, bsize_map_fn,
                 range_start, range_stop, range_idx=None):
        """flav_map_fn: String, filename with map from flavors to their codes.

        interval_map_fn: String, filename where mapping from durations
        to intervals stored. Used here to get list of intervals.

        bsize_map_fn: String, mapping from integers to bsize codes
        (e.g. 11-15, or 26-50).  This is batches of flavors, not to be
        confused with "batches" of examples for ML.

        range_start/stop: Int: timestamps for start/end of training data.

        range_idx: Int: If given, tensor maker will use this index in
        the range features (modulo the number of range features).

        """
        # Features for flavs, hazard/mask (2*), and batch sizes:
        self.flav_tmaker = FlavTensorMaker(flav_map_fn, range_start,
                                           range_stop, range_idx)
        _, self.nintervals = read_interval_map(interval_map_fn)
        self.ninput = self.flav_tmaker.get_ninput()
        self.ninput += 1 + self.nintervals * 2
        self.bsize_map, self.bsize_other = get_batch_size_map(
            bsize_map_fn)
        self.bsize_idxs = self.get_bsize_idxs(self.bsize_map,
                                              self.bsize_other)
        self.ninput += len(self.bsize_idxs)
        self.noutput = self.nintervals
        # This allows faster lookup of our encodings:
        self.dur_embedding = precreate_dur_embedding(self.nintervals)
        self.targ_encoding, self.mask_encoding = precreate_targ_mask_encoding(
            self.nintervals)

    @staticmethod
    def get_bsize_idxs(bsize_map, bsize_other):
        """Return a map from bsize code to index.  We include bsize_other
        because it's not actually in the map (we handle separately).

        """
        bsize_idxs = {}
        unique_bsizes = [x[0] for x in groupby(bsize_map.values())]
        bsizes = unique_bsizes + [bsize_other]
        for idx, bsize in enumerate(bsizes):
            bsize_idxs[bsize] = idx
        return bsize_idxs

    def get_ninput(self):
        """Getter for the ninput

        """
        return self.ninput

    def get_noutput(self):
        """Getter for the noutput

        """
        return self.noutput

    def __encode_durs(self, dur_list):
        """Encode the durations as a tensor of dimensionality NFLAVS x 1 x
        NDUR_FEATS.

        """
        # exclude last duration as target:
        embed_idxs = torch.tensor([durstr_to_embed_idx(d, self.nintervals)
                                   for d in dur_list[:-1]])
        dur_tensor = torch.nn.functional.embedding(
            embed_idxs, self.dur_embedding).unsqueeze(dim=1)
        return dur_tensor

    def __encode_bsize(self, flavs):
        """Encode the bsizes as a tensor of dimensionality NFLAVS x 1 x
        NBSIZE_FEATS.

        """
        flavs = flavs[:-1]
        bsizes = flavs_to_batchsize(flavs, self.bsize_map,
                                    self.bsize_other)
        nbsize_feats = len(self.bsize_idxs)
        bsize_tensor = torch.zeros(len(flavs), 1, nbsize_feats)
        for idx, bsize in enumerate(bsizes):
            # Boundary flavors are already encoded in the flavor
            # features, so just leave all-zero here:
            if bsize != BOUND:
                bsize_tensor[idx][0][self.bsize_idxs[bsize]] = 1
        return bsize_tensor

    def encode_input(self, timestamp, flavs, dur_list):
        """Given a timestamp, flavor sequence (as a string) and a dur list, we
        encode it.  For NFLAVS inputs, output should be NFLAVS x 1 x
        NFEATURES.  The last flav *is* included as a feature for durs,
        and all the durs are features, plus a new BOUND first one, and
        excluding the last duration.

        """
        # Durs are encoded with previous dur, and current flav. Since
        # flav_tensor_maker.encode_input strips off last flav, we can
        # line it up with dur input by having an extra BOUND.
        flavs = flavs + [BOUND]
        dur_list = [BOUND] + dur_list
        # Flavs will be (NFLAVS-1 x 1 x NFLAV_FEATS)
        flav_input = self.flav_tmaker.encode_input(timestamp, flavs)
        # Durs will be: (NFLAVS-1 x 1 x NDUR_FEATS)
        dur_input = self.__encode_durs(dur_list)
        bsize_input = self.__encode_bsize(flavs)
        return torch.cat([flav_input, dur_input, bsize_input], dim=2)

    def encode_target(self, dur_list):
        """Given a whole input line, we encode the targets.

        """
        embed_idxs = torch.tensor([durstr_to_embed_idx(d, self.nintervals)
                                   for d in dur_list])
        targ_tensor = torch.nn.functional.embedding(
            embed_idxs, self.targ_encoding)
        mask_tensor = torch.nn.functional.embedding(
            embed_idxs, self.mask_encoding)
        return targ_tensor, mask_tensor

    def replace_dur_input(self, my_input, idx, new_dur):
        """Replace existing encoding of dur at idx offset in input, in place,
        with encoding of 'new_dur' instead.

        """
        nflav_feats = self.flav_tmaker.get_ninput()
        ndur_feats = 1 + self.nintervals * 2
        ndur_ends = ndur_feats + nflav_feats
        embed_idx = torch.tensor([durstr_to_embed_idx(new_dur, self.nintervals)])
        new_dur_tensor = torch.nn.functional.embedding(
            embed_idx, self.dur_embedding)[0]
        my_input[idx, :, nflav_feats:ndur_ends] = new_dur_tensor
