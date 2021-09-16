"""Dataset for durations.  All examples pre-created at once.

License:
    MIT License

    Copyright (c) 2021 HUAWEI CLOUD

"""
import logging
from math import ceil
import torch
from torch.utils.data import Dataset
from tracegen_rnn.constants import ExampleKeys
from tracegen_rnn.dur_tensor_maker import DurTensorMaker
from tracegen_rnn.utils.file_utils import yield_trace_lines
logger = logging.getLogger("tracegen_rnn.dur_dataset")
REPORT = 1000


class DurDataset(Dataset):
    """A dataset that can be used for training and testing trace sequence
    generation for duration data.

    """

    def __init__(self, flav_map_fn, interval_map_fn, seq_len,
                 flavs_dataset_fn, durs_dataset_fn, bsize_map_fn,
                 range_start=None, range_stop=None):
        """flav_map_fn: String, filename with map from flavors to their codes.

        interval_map_fn: String, filename where mapping from durations
        to intervals stored. Used here to get list of intervals.

        seq_len: Int, how long to make the sequences for each example.

        flavs_dataset_fn: String, the filename where our flavs dataset is.

        durs_dataset_fn: String, the filename where our durs dataset is.

        bsize_map_fn: String, mapping from integers to bsize codes
        (e.g. 11-15, or 26-50).  This is batches of flavors, not to be
        confused with "batches" of examples for ML.

        range_start/stop: Int: timestamps for start/end of training data.
        flav_map_fn: String, the filename where the map is from
        flavors to their letters.  We just use this for to define our
        1-hot-encoding and whatnot.

        """
        self.seq_len = seq_len
        self.tmaker = DurTensorMaker(flav_map_fn, interval_map_fn,
                                     bsize_map_fn, range_start,
                                     range_stop)
        flavs_data = yield_trace_lines(flavs_dataset_fn)
        durs_data = yield_trace_lines(durs_dataset_fn)
        # Last step, using the above member variables:
        self.all_inputs, self.all_targets, self.all_masks = (
            self.__make_example_tensor(flavs_data, durs_data))
        logger.info("Creating input tensor: %s", self.all_inputs.shape)
        logger.info("Creating target tensor: %s", self.all_targets.shape)
        logger.info("Creating mask tensor: %s", self.all_masks.shape)

    def get_ninput(self):
        """In this dataset, how many dimensions are in the input?

        """
        return self.tmaker.get_ninput()

    def get_noutput(self):
        """In this dataset, how many dimensions are in the output?

        """
        return self.tmaker.get_noutput()

    def __make_example_tensor(self, flavs_data, durs_data):
        """Go through all the lines in the traces and create one big example
        tensor, where the first dimension is the example number.

        """
        all_inputs, all_targets, all_masks = [], [], []
        cnt = 0
        total_nflavs = 0
        for flav_line, dur_line in zip(flavs_data, durs_data):
            my_input, my_target, my_mask = self.__make_example_from_line(
                flav_line, dur_line)
            total_nflavs += len(my_target)
            all_inputs.append(my_input)
            all_targets.append(my_target)
            all_masks.append(my_mask)
            if cnt > 1 and cnt % REPORT == 0:
                logger.info("Read %s flav/dur lines", cnt)
            cnt += 1
        logger.info("Read %s flav/dur lines", cnt)
        # Now create a single vector for each of these by reshaping
        # (do them one-at-a-time to lower memory footprint):
        all_inputs_tensor = self.__reshape_tensor_list(
            all_inputs, total_nflavs, is_inputs=True)
        all_targets_tensor = self.__reshape_tensor_list(
            all_targets, total_nflavs)
        # Since the padding value is also zero, it automatically masks
        # out targets we shouldn't evaluate on:
        all_masks_tensor = self.__reshape_tensor_list(
            all_masks, total_nflavs)
        return all_inputs_tensor, all_targets_tensor, all_masks_tensor

    def __get_padding_needed(self, nflavs):
        """Based on the length of the list of tensors (could be inputs,
        targets, or masks), figure out how much padding we need.

        """
        nseqs = ceil(1.0 * nflavs / self.seq_len)
        padding_needed = nseqs * self.seq_len - nflavs
        return padding_needed

    def __reshape_tensor_list(self, all_tensors_lst, total_nflavs,
                              is_inputs=False):
        """For input, targets, or masks: Depending on sequence length, reshape
        accordingly. Also, pad so that we divide evenly. Overall
        purpose is to let us pick an example via reshaped_inputs[idx],
        or reshaped_targets[idx], etc. when we want to draw examples.

        Arguments:

        lists of tensors (can be inputs, targets, or masks)

        total_nflavs: Int

        is_inputs: Boolean, we reshape a little differently for input.

        """
        padding_needed = self.__get_padding_needed(total_nflavs)
        padding_tensor_shape = list(all_tensors_lst[0].shape)
        padding_tensor_shape[0] = padding_needed
        padding_tensor = torch.zeros(padding_tensor_shape)
        all_tensors_lst.append(padding_tensor)
        all_tensors = torch.cat(all_tensors_lst)
        if is_inputs:
            reshaped_tensor = all_tensors.reshape(
                -1, self.seq_len, 1, padding_tensor_shape[-1])
        else:
            reshaped_tensor = all_tensors.reshape(
                -1, self.seq_len, padding_tensor_shape[-1])
        return reshaped_tensor

    def __make_example_from_line(self, flav_line, dur_line):
        """Unpack the line and make the example from it.

        """
        timestamp, flavs = flav_line
        timestamp2, dur_list = dur_line
        assert timestamp == timestamp2, \
            "Flavs and durs not aligned: {} vs. {}". \
            format(timestamp, timestamp2)
        timestamp = int(timestamp)
        ex_input = self.tmaker.encode_input(
            timestamp, flavs, dur_list)
        ex_output, ex_mask = self.tmaker.encode_target(dur_list)
        return ex_input, ex_output, ex_mask

    def __len__(self):
        """Read how many sequences we have of length seq_len:

        """
        return len(self.all_targets)

    def __getitem__(self, idx):
        ex_input = self.all_inputs[idx]
        ex_target = self.all_targets[idx]
        ex_masks = self.all_masks[idx]
        new_example = {
            ExampleKeys.INPUT: ex_input,
            ExampleKeys.TARGET: ex_target,
            ExampleKeys.OUT_MASK: ex_masks
        }
        return new_example
