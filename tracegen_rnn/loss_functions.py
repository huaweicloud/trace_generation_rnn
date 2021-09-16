"""Custom loss functions, used either with flavors or durations.

License:
    MIT License

    Copyright (c) 2021 HUAWEI CLOUD

"""

import torch
from torch import nn
from tracegen_rnn.flav_dataset import IGNORE_INDEX
BIG = 1e5


class FlavLossFunctions():
    """Utility, parameter-free loss functions for training/evaluation of
    flavor predictors.

    """
    @staticmethod
    def next_step_err(outputs, targets):
        """How often guess of next flavor is incorrect.

        """
        good_targets = targets != IGNORE_INDEX
        targets = targets[good_targets]
        outputs = outputs[good_targets]
        # Inject a tiny amount of randomness (otherwise some baselines
        # may guess same class every time):
        out_rand = torch.rand(outputs.shape) / BIG
        out_rand = out_rand.to(outputs.device)
        outputs += out_rand
        guesses = outputs.argmax(dim=1)
        nwrong = torch.sum(guesses != targets).double()
        return nwrong/len(targets)


class DurLossFunctions():
    @staticmethod
    def masked_bce_logits_loss(outputs, targets, masks):
        """Compute the BCEWithLogitsLoss, but only on the points in the mask.
        Also return nincluded, which may be used in loss aggregation.

        Inputs:

        outputs: Tensor[Float]: nexamples x nintervals

        targets: Tensor[Float]: nexamples x nintervals

        masks: Tensor[Float]: nexamples x nintervals

        """
        # Reshape as if everything was batch of 1:
        outputs = outputs.reshape(-1, 1)
        targets = targets.reshape(-1, 1)
        masks = masks.reshape(-1, 1)
        bceloss = nn.BCEWithLogitsLoss(weight=masks, reduction="sum")
        tot_loss = bceloss(outputs, targets)
        nincluded = torch.sum(masks).item()
        avg_loss = tot_loss / nincluded
        return avg_loss, nincluded

    @staticmethod
    def max_likelihood_err(outputs, targets, masks):
        """Guess most likely termination interval according to PMF, compute
        percentage incorrect. PMF(i) = S(i-1)H(i)

        Inputs:

        outputs: Tensor[Float]: nexamples x nintervals

        targets: Tensor[Float]: nexamples x nintervals

        masks: Tensor[Float]: nexamples x nintervals

        Note: if target is masked out (whether just target point, or
        entire target vector), don't count in error or total.

        Returns:

        (err, tot): the error, and the number of instances included in
        the error.

        """
        sigmoid = nn.Sigmoid()
        out_probs = sigmoid(outputs)
        out_probs *= masks
        targets *= masks
        # Cumulatively multiply hazards to get survival, then pmf:
        surv = torch.cumprod((1.0 - out_probs), dim=1)
        pmf = out_probs
        pmf[:, 1:] = out_probs[:, 1:] * surv[:, :-1]
        max_pmf_pts = torch.argmax(pmf, dim=1)
        max_pmf = torch.zeros_like(pmf, device=pmf.device)
        max_pmf[torch.arange(len(max_pmf)), max_pmf_pts] = 1
        # Ones we got right are where max_pmf matches a target:
        nmatches = (max_pmf * targets).sum()
        tot = targets.sum()
        err = (tot - nmatches) / tot
        return err, tot
