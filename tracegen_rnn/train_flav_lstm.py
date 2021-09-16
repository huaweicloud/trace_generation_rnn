"""Read in training and testing flavor traces, turn the input data
into Datasets, run training and evaluate as you go.

License:
    MIT License

    Copyright (c) 2021 HUAWEI CLOUD

"""
import argparse
import logging
import torch
from tracegen_rnn.evaluate_flav_lstm import EvaluateFlavLSTM, make_flav_dataloaders
from tracegen_rnn.flav_tensor_maker import FlavTensorMaker
from tracegen_rnn.train_lstm import TrainArgs, TrainLSTM, get_init_model
from tracegen_rnn.utils.common_args import add_common_args, add_train_args
from tracegen_rnn.utils.logging_utils import init_console_logger
logger = logging.getLogger("tracegen_rnn.train_flav_lstm")
REPORT = 200


def main(args):
    logger_levels = [("tracegen_rnn", logging.DEBUG)]
    init_console_logger(logger_levels)
    tmaker = FlavTensorMaker(args.flav_map_fn, args.range_start,
                             args.range_stop)
    net = get_init_model(args, tmaker)
    logger.info("Initial net: %s", str(net))
    trainloader, testloader = make_flav_dataloaders(args)
    train_args = TrainArgs(args.lr, args.weight_decay, args.max_iters)
    eval_lstm = EvaluateFlavLSTM(net, args.device, testloader)
    train_run = TrainLSTM(eval_lstm, net, train_args, trainloader)
    criterion = torch.nn.CrossEntropyLoss()
    trained_net = train_run.run(criterion)
    if args.model_save_fn is not None:
        trained_net.save(args.model_save_fn)


def parse_arguments():
    """Helper function to parse the command-line arguments, return as an
    'args' object.

    """
    parser = argparse.ArgumentParser(
        description="Training the PyTorch-Based Predictor of flavor sequences")
    add_common_args(parser)
    add_train_args(parser)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    MY_ARGS = parse_arguments()
    main(MY_ARGS)
