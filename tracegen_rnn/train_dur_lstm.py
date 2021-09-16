"""Read in training and testing duration traces, turn the input data
into dataloaders, run training and evaluate as you go.

License:
    MIT License

    Copyright (c) 2021 HUAWEI CLOUD

"""

import argparse
import logging
from tracegen_rnn.dur_tensor_maker import DurTensorMaker
from tracegen_rnn.evaluate_dur_lstm import EvaluateDurLSTM, make_dur_dataloaders
from tracegen_rnn.loss_functions import DurLossFunctions
from tracegen_rnn.train_lstm import TrainArgs, TrainLSTM, get_init_model
from tracegen_rnn.utils.common_args import add_train_args, add_duration_args
from tracegen_rnn.utils.logging_utils import init_console_logger
logger = logging.getLogger("tracegen_rnn.train_dur_lstm")


def main(args):
    logger_levels = [("tracegen_rnn", logging.DEBUG)]
    init_console_logger(logger_levels)
    tmaker = DurTensorMaker(args.flav_map_fn, args.interval_map_fn,
                            args.bsize_map_fn, args.range_start,
                            args.range_stop)
    net = get_init_model(args, tmaker)
    logger.info("Initial net: %s", str(net))
    trainloader, testloader = make_dur_dataloaders(args)
    train_args = TrainArgs(args.lr, args.weight_decay, args.max_iters)
    eval_lstm = EvaluateDurLSTM(net, args.device, testloader)
    train_run = TrainLSTM(eval_lstm, net, train_args, trainloader)
    criterion = DurLossFunctions.masked_bce_logits_loss
    trained_net = train_run.run(criterion)
    if args.model_save_fn is not None:
        trained_net.save(args.model_save_fn)


def parse_arguments():
    """Helper function to parse the command-line arguments, return as an
    'args' object.

    """
    parser = argparse.ArgumentParser(
        description="Training the LSTM for duration sequences")
    add_duration_args(parser)
    add_train_args(parser)
    parser.add_argument(
        '--train_durs', type=str, required=True,
        help="Dur data to use for training.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    MY_ARGS = parse_arguments()
    main(MY_ARGS)
