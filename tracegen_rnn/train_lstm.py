"""Train a generic LSTM, for flavors or durations (depending on the
passed LSTM evaluator).

License:
    MIT License

    Copyright (c) 2021 HUAWEI CLOUD

"""
import logging
import torch
from typing import NamedTuple
from tracegen_rnn.trace_lstm import TraceLSTM
from tracegen_rnn.utils.loss_stats import LossStats
logger = logging.getLogger("tracegen_rnn.train_lstm")


class TrainArgs(NamedTuple):
    """Arguments to be used in training."""
    learn_rate: float
    weight_decay: float
    max_iters: int


class TrainLSTM():
    """Class to handle flavor-LSTM training."""
    def __init__(self, eval_lstm, net, train_args, trainloader):
        self.eval_lstm = eval_lstm
        self.net = net
        self.train_args = train_args
        self.trainloader = trainloader

    def run_train_iteration(self, data, optimizer, criterion):
        """Run a single training step and return the number of inputs
        processed and the loss.

        """
        optimizer.zero_grad()
        num, loss = self.eval_lstm.batch_forward(data, criterion)
        loss.backward()
        optimizer.step()
        return num, loss

    def iterate_models(self, optimizer, criterion):
        """Run a single training iteration and yield the loss"""
        for epoch in range(self.train_args.max_iters):
            self.net.train()
            loss_stats = LossStats()
            for iter_num, batch in enumerate(self.trainloader, 1):
                num, loss = self.run_train_iteration(batch, optimizer,
                                                     criterion)
                loss_stats.update(num, loss)
            overall_loss = loss_stats.overall_loss()
            tot_examples = loss_stats.get_tot_examples()
            logger.info('Train loss, epoch [%d, %7d]: %.7f',
                        epoch, tot_examples, overall_loss)
            yield overall_loss

    def run(self, criterion):
        """Run training on the given neural network.

        """
        optimizer = torch.optim.Adam(self.net.parameters(),
                                     lr=self.train_args.learn_rate,
                                     weight_decay=self.train_args.weight_decay)
        logger.info("Optimizer: %s", optimizer)
        logger.info("Starting training")
        for iter_num, train_loss in enumerate(self.iterate_models(
                optimizer, criterion), 1):
            self.eval_lstm.get_test_score(iter_num, criterion)
        logger.info("Finished training")
        return self.net


def get_init_model(args, tmaker):
    """Return an initial LSTM model for training, given the tensor
    maker for this LSTM.

    """
    # Get ndims from the tensor_maker for this flav_map:
    ninput = tmaker.get_ninput()
    noutput = tmaker.get_noutput()
    model = TraceLSTM(ninput, args.nhidden, noutput, args.nlayers)
    return model
