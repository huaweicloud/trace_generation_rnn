"""Abstract base class for modules that run forward passes and get
scores on test data.

License:
    MIT License

    Copyright (c) 2021 HUAWEI CLOUD

"""
from abc import ABC, abstractmethod
import torch
import logging
from tracegen_rnn.utils.loss_stats import LossStats
logger = logging.getLogger("tracegen_rnn.evaluator")


class Evaluator(ABC):
    """Class to run the forward pass and compute test scores."""
    def __init__(self, net, device_str, testloader):
        self.net = net
        self.device = torch.device(device_str)
        self.testloader = testloader
        if self.device.type == "cuda":
            if self.device.index == 0:
                self.net = self.net.cuda(0)
            else:
                self.net = self.net.cuda(1)

    @abstractmethod
    def batch_forward(self, batch, criterion):
        """Override to pick the outputs for the batch, compute the loss."""

    def get_test_score(self, epoch, criterion):
        """Get the score of current net on test set."""
        loss_stats = LossStats()
        with torch.no_grad():
            self.net.eval()
            for iter_num, batch in enumerate(self.testloader, 1):
                num, loss = self.batch_forward(batch, criterion)
                loss_stats.update(num, loss)
        overall_loss = loss_stats.overall_loss()
        if epoch is not None:
            logger.info('Test loss, epoch [%d]: %.7f', epoch, overall_loss)
        else:
            logger.info('Test loss: %.7f', overall_loss)
        return overall_loss
