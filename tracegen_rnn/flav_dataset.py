"""Dataset for flavors.  We pre-create all the tensors at the
beginning, so getting examples online is trivial.

License:
    MIT License

    Copyright (c) 2021 HUAWEI CLOUD

"""

import logging
from math import ceil
import torch
from torch.utils.data import Dataset
from tracegen_rnn.constants import BOUND, ExampleKeys
from tracegen_rnn.flav_tensor_maker import FlavTensorMaker
from tracegen_rnn.utils.file_utils import yield_trace_lines

logger = logging.getLogger("tracegen_rnn.flav_dataset")
REPORT = 1000

# This is a known quantity for PyTorch loss functions:
IGNORE_INDEX = -100


class FlavDataset(Dataset):
    """A dataset that can be used for flavor sequence modeling.

    """

    def __init__(self, flav_map_fn, seq_len, dataset_fn, range_start,
                 range_stop):
        """Initialize the dataset class.

        Args:

        flav_map_fn: String, filename with map from flavors to their codes.

        seq_len: Int, how long to make the sequences for each example.

        dataset_fn: String, filename where input dataset lies.

        range_start/stop: Int: timestamps for start/end of training data.

        """
        self.seq_len = seq_len
        self.tmaker = FlavTensorMaker(flav_map_fn,
                                      range_start=range_start,
                                      range_stop=range_stop)
        trace_data = yield_trace_lines(dataset_fn)
        # make one giant example, getitem() & len() will take pieces:
        self.all_inputs, self.all_targets = self.__make_example_tensor(
            trace_data)
        logger.info("Creating input tensor: %s", self.all_inputs.shape)
        logger.info("Creating target tensor: %s", self.all_targets.shape)
        assert len(self.all_inputs) == len(self.all_targets)

    def __make_example_tensor(self, trace_data):
        """Go through lines in trace and create one big example tensor, where
        the first dimension is example number.

        """
        all_inputs, all_targets = [], []
        for idx, line in enumerate(trace_data):
            my_input, my_target = self.__make_example_from_line(line)
            all_inputs.append(my_input)
            all_targets.append(my_target)
            if idx > 1 and idx % REPORT == 0:
                logger.info("Read %s dataset lines", idx)
        logger.info("Read %s dataset lines", idx)
        # Create a single vector for each of these by reshaping:
        all_inputs = torch.cat(all_inputs)
        all_targets = torch.cat(all_targets)
        all_inputs, all_targets = self.__reshape_data(
            all_inputs, all_targets)
        return all_inputs, all_targets

    def __reshape_data(self, all_inputs, all_targets):
        """Depending on the sequence length, reshape accordingly.  Also, pad
         with targets with IGNORE_INDEX so that we divide evenly.

        """
        nflavs = len(all_inputs)
        nseqs = ceil(1.0 * nflavs / self.seq_len)
        padding_needed = nseqs * self.seq_len - nflavs
        fake_targets = (torch.ones(padding_needed, dtype=torch.long) *
                        IGNORE_INDEX)
        all_targets = torch.cat([all_targets, fake_targets])
        fake_input_shape = list(all_inputs.shape)
        fake_input_shape[0] = padding_needed
        fake_input = torch.zeros(fake_input_shape)
        all_inputs = torch.cat([all_inputs, fake_input])
        # After padding, reshape into sequences of seq_len:
        reshaped_inputs = all_inputs.reshape(-1, self.seq_len, 1,
                                             fake_input_shape[-1])
        reshaped_targets = all_targets.reshape(-1, self.seq_len)
        return reshaped_inputs, reshaped_targets

    def __make_example_from_line(self, line):
        """Unpack the line and make the example from it.

        """
        timestamp, flavs = line
        timestamp = int(timestamp)
        # Lines don't BEGIN with BOUND, so put it on:
        flavs = [BOUND] + flavs
        ex_input = self.tmaker.encode_input(timestamp, flavs)
        ex_target = self.tmaker.encode_target(flavs)
        return ex_input, ex_target

    def __len__(self):
        """Return number of sequences of length seq_len:

        """
        return len(self.all_targets)

    def __getitem__(self, idx):
        """Return an example from all our pre-made tensors."""
        ex_input = self.all_inputs[idx]
        ex_target = self.all_targets[idx]
        sample = {ExampleKeys.INPUT: ex_input,
                  ExampleKeys.TARGET: ex_target}
        return sample
