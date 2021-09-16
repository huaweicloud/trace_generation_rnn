"""Run the forward pass to evaluate the duration LSTM.

License:
    MIT License

    Copyright (c) 2021 HUAWEI CLOUD

"""
import argparse
import logging
from torch.utils.data import DataLoader
from tracegen_rnn.constants import ExampleKeys
from tracegen_rnn.dur_dataset import DurDataset
from tracegen_rnn.evaluator import Evaluator
from tracegen_rnn.loss_functions import DurLossFunctions
from tracegen_rnn.trace_lstm import TraceLSTM
from tracegen_rnn.utils.collate_utils import CollateUtils
from tracegen_rnn.utils.common_args import add_duration_args
from tracegen_rnn.utils.logging_utils import init_console_logger
logger = logging.getLogger("tracegen_rnn.evaluate_dur_lstm")


def make_dur_dataloaders(args):
    try:
        trainset = DurDataset(args.flav_map_fn, args.interval_map_fn,
                              args.seq_len, args.train_flavs,
                              args.train_durs, args.bsize_map_fn,
                              args.range_start, args.range_stop)
        trainloader = DataLoader(trainset, batch_size=args.batch_size,
                                 collate_fn=CollateUtils.batching_collator,
                                 shuffle=True)
    except AttributeError:
        # No train_flavs provided
        trainloader = None
    testset = DurDataset(args.flav_map_fn, args.interval_map_fn,
                         args.seq_len, args.test_flavs,
                         args.test_durs, args.bsize_map_fn,
                         args.range_start, args.range_stop)
    testloader = DataLoader(testset, batch_size=args.batch_size,
                            collate_fn=CollateUtils.batching_collator,
                            shuffle=False)
    return trainloader, testloader


class EvaluateDurLSTM(Evaluator):
    """Class to help with testing of a duration LSTM."""
    def batch_forward(self, batch, criterion):
        inputs = batch[ExampleKeys.INPUT]
        targets = batch[ExampleKeys.TARGET]
        masks = batch[ExampleKeys.OUT_MASK]
        inputs, targets, masks = (inputs.to(self.device),
                                  targets.to(self.device),
                                  masks.to(self.device))
        batch_size = inputs.shape[1]
        self.net.hidden = self.net.init_hidden(self.device, batch_size)
        outputs = self.net(inputs)
        outputs = outputs.reshape(-1, outputs.shape[-1])
        targets = targets.reshape(outputs.shape[0], -1)
        masks = masks.reshape(outputs.shape[0], -1)
        loss, num = criterion(outputs, targets, masks)
        return num, loss


def main(args):
    """Run the evaluation of the saved model.

    """
    # Init a logger that writes to console:
    logger_levels = [("tracegen_rnn", logging.DEBUG)]
    init_console_logger(logger_levels)
    _, testloader = make_dur_dataloaders(args)
    net = TraceLSTM.create_from_path(args.lstm_model, args.device)
    eval_lstm = EvaluateDurLSTM(net, args.device, testloader)
    criterions = [DurLossFunctions.masked_bce_logits_loss,
                  DurLossFunctions.max_likelihood_err]
    labels = ["BCE", "Err%"]
    for criterion, label in zip(criterions, labels):
        logger.info(label)
        eval_lstm.get_test_score(None, criterion)


def parse_arguments():
    """Helper function to parse the command-line arguments, return as an
    'args' object.

    """
    parser = argparse.ArgumentParser(
        description="Eval of dur LSTM.")
    add_duration_args(parser)
    parser.add_argument(
        '--lstm_model', type=str, required=False,
        help="The trained model for the LSTM.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    MY_ARGS = parse_arguments()
    main(MY_ARGS)
