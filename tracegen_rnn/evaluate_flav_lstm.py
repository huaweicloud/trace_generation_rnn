"""Run the forward pass to evaluate the flavor LSTM.

License:
    MIT License

    Copyright (c) 2021 HUAWEI CLOUD

"""
import argparse
import logging
import torch
from torch.utils.data import DataLoader
from tracegen_rnn.constants import ExampleKeys
from tracegen_rnn.evaluator import Evaluator
from tracegen_rnn.flav_dataset import FlavDataset, IGNORE_INDEX
from tracegen_rnn.loss_functions import FlavLossFunctions
from tracegen_rnn.trace_lstm import TraceLSTM
from tracegen_rnn.utils.collate_utils import CollateUtils
from tracegen_rnn.utils.common_args import add_common_args
from tracegen_rnn.utils.logging_utils import init_console_logger
logger = logging.getLogger("tracegen_rnn.evaluate_flav_lstm")


def make_flav_dataloaders(args):
    try:
        trainset = FlavDataset(args.flav_map_fn, args.seq_len,
                               args.train_flavs, args.range_start,
                               args.range_stop)
        trainloader = DataLoader(trainset, batch_size=args.batch_size,
                                 collate_fn=CollateUtils.batching_collator,
                                 shuffle=True)
    except AttributeError:
        # No train_flavs provided:
        trainloader = None
    testset = FlavDataset(args.flav_map_fn, args.seq_len,
                          args.test_flavs, args.range_start,
                          args.range_stop)
    testloader = DataLoader(testset, batch_size=args.batch_size,
                            collate_fn=CollateUtils.batching_collator,
                            shuffle=False)
    return trainloader, testloader


class EvaluateFlavLSTM(Evaluator):
    """Class to help with testing of a flavor LSTM."""
    def batch_forward(self, batch, criterion):
        """Run the forward pass and get the number of examples and the loss.

        """
        inputs = batch[ExampleKeys.INPUT]
        targets = batch[ExampleKeys.TARGET]
        num = targets[targets != IGNORE_INDEX].numel()
        inputs, targets = (inputs.to(self.device),
                           targets.to(self.device))
        batch_size = inputs.shape[1]
        self.net.hidden = self.net.init_hidden(self.device, batch_size)
        outputs = self.net(inputs)
        outputs = outputs.reshape(-1, outputs.shape[-1])
        targets = targets.reshape(-1)
        loss = criterion(outputs, targets)
        return num, loss


def main(args):
    """Run the evaluation of the saved model.

    """
    # Init a logger that writes to console:
    logger_levels = [("tracegen_rnn", logging.DEBUG)]
    init_console_logger(logger_levels)
    _, testloader = make_flav_dataloaders(args)
    net = TraceLSTM.create_from_path(args.lstm_model, args.device)
    eval_lstm = EvaluateFlavLSTM(net, args.device, testloader)
    criterions = [torch.nn.CrossEntropyLoss(), FlavLossFunctions.next_step_err]
    labels = ["NLL", "Err%"]
    for criterion, label in zip(criterions, labels):
        logger.info(label)
        eval_lstm.get_test_score(None, criterion)


def parse_arguments():
    """Helper function to parse the command-line arguments, return as an
    'args' object.

    """
    parser = argparse.ArgumentParser(
        description="Eval of flavor LSTM.")
    add_common_args(parser)
    parser.add_argument(
        '--lstm_model', type=str, required=False,
        help="The trained model for the LSTM.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    MY_ARGS = parse_arguments()
    main(MY_ARGS)
