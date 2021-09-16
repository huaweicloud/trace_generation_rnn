"""Common arguments for use in training and testing.

License:
    MIT License

    Copyright (c) 2021 HUAWEI CLOUD

"""


def add_common_args(parser):
    """Helper to add args that provide common options used in training and
    evaluation.

    """
    parser.add_argument(
        '--test_flavs', type=str, required=True,
        help="Flavs data to use for test evaluation.")
    parser.add_argument(
        '--range_start', type=int, required=True,
        help="For time range features, when the training time range starts.")
    parser.add_argument(
        '--range_stop', type=int, required=True,
        help="For time range features, when the training time range stops.")
    parser.add_argument(
        '--flav_map_fn', type=str, required=True,
        help="File that maps flavors to letters for Azure.")
    parser.add_argument(
        '--seq_len', type=int, required=True,
        help="How long to take for sequences in the batches.")
    parser.add_argument(
        '--batch_size', type=int, required=True,
        help="How big to make the batches after collation.")
    parser.add_argument(
        '--device', type=str, required=True,
        help="Run optimization on GPU (\"cuda:0\") or on CPU (\"cpu\").")


def add_duration_args(parser):
    """Args that provide common options used in duration work."""
    # Use all the common args, plus more:
    add_common_args(parser)
    parser.add_argument(
        '--interval_map_fn', type=str, required=True,
        help="File that maps durations to intervals.")
    parser.add_argument(
        '--bsize_map_fn', type=str, required=True,
        help="File that maps batch sizes to codes (include if using bsize feats).")
    parser.add_argument(
        '--test_durs', type=str, required=True,
        help="Durs data to use for test evaluation.")


def add_train_args(parser):
    """Arguments for training (flavors or durations)."""
    parser.add_argument(
        '--nlayers', type=int, required=True,
        help="Number of layers in LSTM.")
    parser.add_argument(
        '--nhidden', type=int, required=True,
        help="Number of hidden units in each hidden layer.")
    parser.add_argument(
        '--lr', type=float, required=True,
        help="The learning rate of the optimizer.")
    parser.add_argument(
        '--max_iters', type=int, required=True,
        help="When to stop training.")
    parser.add_argument(
        '--weight_decay', type=float, required=True,
        help="Weight decay in the Adam optimizer (L2 penalty).")
    parser.add_argument(
        '--train_flavs', type=str, required=True,
        help="Flav data to use for training.")
    parser.add_argument(
        '--model_save_fn', type=str, required=False,
        help="Location for saving the model.")
