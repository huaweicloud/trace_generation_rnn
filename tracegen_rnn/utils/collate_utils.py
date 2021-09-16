"""Utility collate functions for the DataLoaders

License:
    MIT License

    Copyright (c) 2021 HUAWEI CLOUD

"""
import torch
from tracegen_rnn.constants import ExampleKeys


class CollateUtils():
    """Utility collate functions for the DataLoaders."""

    @staticmethod
    def batching_collator(batch):
        """Return an example (dict) where the values are now minibatches.

        Arguments: batch: an iterable over example (dicts) in a Dataset

        Returns: collated, a single example with minibatch payload

        """
        all_inputs = []
        all_targets = []
        all_masks = []
        do_masks = batch[0].get(ExampleKeys.OUT_MASK) is not None
        for example in batch:
            all_inputs.append(example[ExampleKeys.INPUT])
            # targets are either a flat SEQ_LEN vector of targets (in
            # flavors) or 47 (in durs), so reshape accordingly:
            targets = example[ExampleKeys.TARGET]
            if len(targets.shape) == 1:
                all_targets.append(targets.reshape(-1, 1, 1))
            else:
                all_targets.append(targets.reshape(-1, 1, targets.shape[-1]))
            if do_masks:
                masks = example[ExampleKeys.OUT_MASK]
                all_masks.append(masks.reshape(-1, 1, targets.shape[-1]))
        # Join them together along the batch dimension:
        new_inputs = torch.cat(all_inputs, dim=1)
        new_targets = torch.cat(all_targets, dim=1)
        collated = {ExampleKeys.INPUT: new_inputs,
                    ExampleKeys.TARGET: new_targets}
        if do_masks:
            new_masks = torch.cat(all_masks, dim=1)
            collated[ExampleKeys.OUT_MASK] = new_masks
        return collated
