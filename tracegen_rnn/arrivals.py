"""Train and test the batch Poisson regression arrival model.

License:
    MIT License

    Copyright (c) 2021 HUAWEI CLOUD

"""
import argparse
import logging
import numpy as np
import statsmodels.api as sm
import torch
from pathlib import Path
from tracegen_rnn.constants import BOUND
from tracegen_rnn.flav_tensor_maker import FlavTensorMaker
from tracegen_rnn.utils.experimental_utils import set_all_seeds
from tracegen_rnn.utils.file_utils import yield_trace_lines
from tracegen_rnn.utils.logging_utils import init_console_logger

# Logging in this file:
logger = logging.getLogger("tracegen_rnn.arrivals")
REPORT = 50
SAMPLE_PERIOD_S = 300
# For the prediction intervals:
DEF_NRANGE_SAMPS = 500
DEF_NPOISSON = 500

# On average, sample one week back for geo features:
GEO_PROB = 1.0/7.0


def make_arrival_vector(timestamp, range_start, range_stop, range_idx):
    """Create a feature vector for this example's timestamp.

    Arguments:

    timestamp: Int: the timestamp for which we would like to generate
    the feature vector.

    range_start/stop: Int: timestamps for start/end of training data.

    range_idx: Int: If given, tensor maker will use this index in the
    range features (modulo the number of range features).

    """
    one_hots = FlavTensorMaker.one_hots_timestamp(timestamp)
    range_feats = FlavTensorMaker.range_features(
        range_start, range_stop, timestamp, range_idx)
    feats = torch.cat([one_hots, range_feats], dim=1)
    # Remove batch dimension:
    feats = feats.reshape(-1).tolist()
    return feats


def add_count_and_vector(x_vals, y_vals, nbatches, ts, range_start,
                         range_stop, range_idx):
    """Create a vector and target and add them to the growing lists."""
    x_vec = make_arrival_vector(ts, range_start, range_stop,
                                range_idx)
    x_vals.append(x_vec)
    y_vals.append(float(nbatches))


def make_dataset(filename, range_start, range_stop, range_idx=None,
                 sample_period_s=SAMPLE_PERIOD_S):
    """Make and return the dataset for the given filename.

    Arguments:

    filename: String, trace file with the flavors on each line

    range_start/stop: Int: timestamps for start/end of training data.

    range_idx: Int: If given, tensor maker will use this index in the
    range features (modulo the number of range features).

    sample_period_s: Int, the sampling period (default 5 minutes)

    Returns:

    np.array, np.array

    """
    y_vals = []
    x_vals = []
    prev_timestamp = None
    for timestamp, flav_lst in yield_trace_lines(filename):
        timestamp = int(timestamp)
        # Output 0s for any missing timestamps:
        while (prev_timestamp is not None and prev_timestamp <
               timestamp - sample_period_s):
            prev_timestamp += sample_period_s
            add_count_and_vector(x_vals, y_vals, 0, prev_timestamp,
                                 range_start, range_stop, range_idx)
        nbatches = sum(1 for f in flav_lst if f == BOUND)
        add_count_and_vector(x_vals, y_vals, nbatches, timestamp,
                             range_start, range_stop, range_idx)
        prev_timestamp = timestamp
    return np.array(y_vals), np.array(x_vals)


def get_quantiles(predict_arr, args):
    """Get prediction quantiles."""
    all_means = np.repeat(predict_arr, args.npoisson_samps,
                          axis=0).astype(np.float64)
    psamps = np.random.poisson(all_means)
    p95 = np.percentile(psamps, 95, axis=0)
    p50 = np.percentile(psamps, 50, axis=0)
    p05 = np.percentile(psamps, 5, axis=0)
    return p05, p50, p95


def get_intervals_on_test(poisson_fit, args):
    """Get the quantiles by sampling from the Poisson distribution on the
    test set.

    """
    nrange_feats = FlavTensorMaker.get_nrange_feats(
        args.range_start, args.range_stop)
    all_predict_cnts = []
    for idx in range(args.nrange_samps):
        if idx > 0 and idx % REPORT == 0:
            logger.info("Sampling test dataset number %d", idx)
        steps_back = np.random.geometric(GEO_PROB) - 1
        # Wrap around back to end at day zero:
        steps_back = steps_back % nrange_feats
        ridx = nrange_feats - steps_back
        y_test, x_test = make_dataset(args.test, args.range_start,
                                      args.range_stop, range_idx=ridx)
        preds = poisson_fit.predict(x_test)
        # Now, let's get the predicted counts (poisson means) for this
        # version of the test set:
        predicted_cnts = preds  # Now the preds are the predictions
        predict_cnts_arr = np.array(predicted_cnts)
        all_predict_cnts.append(predict_cnts_arr)
    logger.info("Sampled %d test datasets", args.nrange_samps)
    logger.info("Sampling Poisson values for %s", args.test)
    predict_arr = np.array(all_predict_cnts)
    p05, p50, p95 = get_quantiles(predict_arr, args)
    return p05, p50, p95, y_test


def get_coverage(p05, p95, truth):
    """How often is the truth between p05 and p95."""
    ncovered = 0
    ntot = 0
    for yval, lower, upper in zip(truth, p05, p95):
        ntot += 1
        if lower <= yval <= upper:
            ncovered += 1
    return ncovered/ntot


def main(args):
    set_all_seeds(42)
    logger_levels = [("tracegen_rnn", logging.DEBUG)]
    init_console_logger(logger_levels)
    logger.info("Making dataset for %s", args.train)
    y_train, x_train = make_dataset(args.train, args.range_start,
                                    args.range_stop)
    logger.info("Fitting model on %s", args.train)
    poisson_fit = sm.GLM(y_train, x_train,
                         family=sm.families.Poisson()) \
                    .fit_regularized(alpha=args.regularization)
    if args.out_model_pickle is not None:
        poisson_fit.save(args.out_model_pickle)
    logger.info("Getting intervals on %s", args.test)
    p05, p50, p95, y_test = get_intervals_on_test(poisson_fit, args)
    coverage = get_coverage(p05, p95, y_test)
    logger.info("p05-p95 coverage for %s = %f", args.test, coverage)


def validate_args(args):
    model_dir = Path(args.out_model_pickle).parent
    if not model_dir.exists():
        raise FileNotFoundError(f"Parent directory {model_dir} of given argument 'out_model_pickle' does not exist")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Training arrival model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--train', type=str, required=True,
        help="The filename of the training data.")
    parser.add_argument(
        '--test', type=str, required=True,
        help="The filename of the testing data.")
    parser.add_argument(
        '--regularization', type=float, required=True,
        help="For the Poisson model, how much to regularize.")
    parser.add_argument(
        '--nrange_samps', type=int, required=False, default=DEF_NRANGE_SAMPS,
        help="How many samples to use when getting arrival intervals.")
    parser.add_argument(
        '--npoisson_samps', type=int, required=False, default=DEF_NPOISSON,
        help="How many samples to use when getting arrival intervals.")
    parser.add_argument(
        '--range_start', type=int, required=False, default=None,
        help="For time range features, when the training time range starts.")
    parser.add_argument(
        '--range_stop', type=int, required=False, default=None,
        help="For time range features, when the training time range stops.")
    parser.add_argument(
        '--out_model_pickle', type=str, required=False,
        help="The filename of where to put the fit model, in pickle format."
    )
    args = parser.parse_args()
    validate_args(args)
    return args


if __name__ == "__main__":
    MY_ARGS = parse_arguments()
    main(MY_ARGS)
