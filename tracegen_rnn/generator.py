"""Generate a trace (flavs+durs) according to our lstm models:

1. Sample number of batch arrivals
2. Auto-regressively sample LSTM flav model to generate that many batches
3. Auto-regressively sample LSTM dur model to assign duration to each flav

License:
    MIT License

    Copyright (c) 2021 HUAWEI CLOUD

"""
import argparse
import logging
import numpy as np
import statsmodels.api as sm
import torch
from tracegen_rnn.arrivals import GEO_PROB, make_arrival_vector
from tracegen_rnn.constants import BOUND
from tracegen_rnn.dur_tensor_maker import DurTensorMaker
from tracegen_rnn.flav_tensor_maker import FlavTensorMaker
from tracegen_rnn.trace_lstm import TraceLSTM
from tracegen_rnn.utils.dur_utils import encode_dur_str
from tracegen_rnn.utils.file_utils import get_flav_map, encode_trace_line
from tracegen_rnn.utils.logging_utils import init_console_logger

# Logging in this file:
logger = logging.getLogger("tracegen_rnn.generator")
REPORT = 10

# How much the trace steps each time, by default:
STEP_S = 300


class GenLSTM():
    """An evaluator that only does forward pass (no loss calculation)."""
    def __init__(self, net, device):
        self.net = net
        self.device = device

    def init_hidden(self):
        self.net.hidden = self.net.init_hidden(self.device)

    def forward(self, my_input):
        outputs = self.net(my_input)
        return outputs


def get_random_range_idx(args):
    nrange_feats = FlavTensorMaker.get_nrange_feats(
        args.range_start, args.range_stop)
    # Subtract 1 to start from zero:
    steps_back = np.random.geometric(GEO_PROB) - 1
    steps_back = steps_back % nrange_feats
    range_idx = nrange_feats - steps_back
    return range_idx


class Generator():
    """Creates an object that generates a trace (flavs and durs) according
    to our batching baseline trace generator process.

    """
    def __init__(self, device, arrival_mdl, flav_lstm, dur_lstm,
                 flav_map_fn, interval_map_fn, bsize_map_fn,
                 range_start, range_stop, range_idx):
        """arrival_mdl: the Poisson GLM mdl from statsmodels

        flav_lstm/dur_lstm: LSTM evaluators to run forward passes

        flav_map_fn: String, filename with map from flavors to their codes.

        interval_map_fn: String, filename where mapping from durations
        to intervals stored. Used here to get list of intervals.

        bsize_map_fn: String, mapping from integers to bsize codes
        (e.g. 11-15, or 26-50).  This is batches of flavors, not to be
        confused with "batches" of examples for ML.

        range_start/stop: Int: timestamps for start/end of training data.

        range_idx: Int: If given, tensor maker will use this index in
        the range features (modulo the number of range features).

        """
        self.device = device
        self.arrival_mdl = arrival_mdl
        flav_map = get_flav_map(flav_map_fn)
        self.flav_to_idxs, self.idx_to_flavs = FlavTensorMaker.get_flav_idxs(
            flav_map)
        self.range_start = range_start
        self.range_stop = range_stop
        self.range_idx = range_idx
        self.flav_tmaker = FlavTensorMaker(flav_map_fn, range_start,
                                           range_stop, range_idx)
        self.dur_tmaker = DurTensorMaker(flav_map_fn, interval_map_fn, bsize_map_fn,
                                         range_start, range_stop, range_idx)
        self.flav_lstm = flav_lstm
        self.dur_lstm = dur_lstm

    def __get_narrivals(self, timestamp):
        arrival_vec = make_arrival_vector(
            timestamp, self.range_start, self.range_stop, self.range_idx)
        pred_mean = self.arrival_mdl.predict([arrival_vec])[0]
        narrivals = np.random.poisson(pred_mean)
        return narrivals

    def __init_flav_input(self, timestamp):
        """Initialize input tensor to a BOUND at given timestamp."""
        flav_lst = [BOUND, BOUND]  # second BOUND ignored by tmaker
        my_input = self.flav_tmaker.encode_input(timestamp, flav_lst)
        my_input = my_input.to(self.device)
        return my_input

    def __adjust_flav_input(self, my_input, prev_flav):
        """Replace the flavor part of the input only."""
        self.flav_tmaker.replace_flav_input(my_input, prev_flav)
        return my_input

    def __sample_flav(self, output):
        """Sample flavor from output of flavor LSTM."""
        probs = torch.softmax(output.reshape(-1), dim=0)
        flav_idx = torch.multinomial(probs, 1).item()
        # Also get flavor string itself:
        flav_flav = self.idx_to_flavs[flav_idx]
        return flav_flav, flav_idx

    def __generate_flavs(self, timestamp, target_nbatches):
        """Auto-regressively generate target_nbatches batches of flavors, at
        given timestamp.

        """
        my_input = self.__init_flav_input(timestamp)
        flav_flavs = []
        flav_idxs = []
        nseen_batches = 0
        prev_flav = BOUND
        while True:
            output = self.flav_lstm.forward(my_input)
            next_flav, next_idx = self.__sample_flav(output)
            # Shouldn't happen, but skip if it does:
            if next_flav == BOUND and prev_flav == BOUND:
                continue
            flav_flavs.append(next_flav)
            flav_idxs.append(next_idx)
            # Increment number batches seen on each bound:
            if next_flav == BOUND:
                nseen_batches += 1
                if nseen_batches == target_nbatches:
                    break
            # Otherwise, get next input and continue:
            my_input = self.__adjust_flav_input(my_input, next_flav)
            prev_flav = next_flav
        return flav_flavs, flav_idxs

    def __sample_from_hazard_logits(self, outputs):
        """Sample duration from output of duration LSTM."""
        hazard_logits = outputs.reshape(-1)
        sigmoid = torch.nn.Sigmoid()
        hazard_probs_tensor = sigmoid(hazard_logits)
        hazard_probs_tensor[-1] = 1.0  # kills-all interval
        rands = torch.rand(len(hazard_probs_tensor), device=self.device)
        killed_in_itvl = rands < hazard_probs_tensor
        out_itvl = torch.nonzero(killed_in_itvl)[0].item()
        out_dur = encode_dur_str(out_itvl, censored=False)
        return out_dur, out_itvl

    def __generate_durs(self, timestamp, flav_flavs):
        """Auto-regressively generate a dur for each input flavor."""
        nseq = len(flav_flavs)
        if nseq == 0:
            return []
        # For efficiency, encode timestamp/flav features with fake
        # durs, then replace durs as we go:
        dur_list = ["|"] * nseq  # fake durs
        my_input = self.dur_tmaker.encode_input(
            timestamp, flav_flavs, dur_list)
        my_input = my_input.to(self.device)
        dur_idxs = []
        for idx in range(nseq):
            next_input = my_input[idx].reshape(1, 1, -1)
            outputs = self.dur_lstm.forward(next_input)
            next_dur, next_dur_idx = self.__sample_from_hazard_logits(outputs)
            dur_idxs.append(next_dur_idx)
            if idx + 1 == nseq:
                break
            if flav_flavs[idx] == BOUND:
                next_dur = BOUND
            # Replace the next input tensor with encoding of this dur:
            self.dur_tmaker.replace_dur_input(my_input, idx + 1, next_dur)
        return dur_idxs

    def __generate_batches(self, timestamp, nbatches):
        """Use flav/dur LSTMs to generate trace for a single row/timestamp."""
        flav_flavs, flav_idxs = self.__generate_flavs(timestamp, nbatches)
        dur_idxs = self.__generate_durs(timestamp, flav_flavs)
        return flav_idxs, dur_idxs

    def __output_flavs(self, timestamp, flav_idxs, out_flavs_file):
        """Make and output the flavor line."""
        flav_lst = [self.idx_to_flavs[i] for i in flav_idxs]
        flav_out = encode_trace_line(timestamp, flav_lst)
        print(flav_out, file=out_flavs_file)

    def __output_durs(self, timestamp, flav_idxs, dur_idxs,
                      out_durs_file):
        """Make and output the duration line."""
        # Check for boundaries in *flav*_idxs:
        bound_idx = self.flav_to_idxs[BOUND]
        dur_strs = []
        for dur_idx, flav_idx in zip(dur_idxs, flav_idxs):
            if flav_idx == bound_idx:
                dur_str = BOUND
            else:
                dur_str = encode_dur_str(dur_idx, censored=False)
            dur_strs.append(dur_str)
        dur_out = encode_trace_line(timestamp, dur_strs)
        print(dur_out, file=out_durs_file)

    def __call__(self, start_s, stop_s, step_s, out_flavs_file,
                 out_durs_file):
        """Generate trace for timestamps from start_s to stop_s inclusive."""
        self.flav_lstm.init_hidden()
        self.dur_lstm.init_hidden()
        for ntimestamps, timestamp in enumerate(
                range(start_s, stop_s + 1, step_s)):
            nbatches = self.__get_narrivals(timestamp)
            if nbatches == 0:
                self.__output_zero_arrivals(timestamp, out_flavs_file,
                                            out_durs_file)
                continue
            flav_idxs, dur_idxs = self.__generate_batches(
                timestamp, nbatches)
            self.__output_flavs(timestamp, flav_idxs, out_flavs_file)
            self.__output_durs(timestamp, flav_idxs, dur_idxs, out_durs_file)
            if ntimestamps > 0 and ntimestamps % REPORT == 0:
                logger.info("Generated %d output lines (now on %d)",
                            ntimestamps, timestamp)


def get_lstm_eval(device, model_fn):
    """Initialize the generation LSTM evaluator."""
    net = TraceLSTM.create_from_path(model_fn, device)
    return GenLSTM(net, device)


def main(args):
    logger_levels = [("tracegen_rnn", logging.DEBUG)]
    init_console_logger(logger_levels)
    logger.info("Reading models")
    arrival_mdl = sm.load(args.arrival_model_pkl)
    flav_gen = get_lstm_eval(args.device, args.flav_model)
    dur_gen = get_lstm_eval(args.device, args.dur_model)
    range_idx = get_random_range_idx(args)
    lstm_generator = Generator(
        args.device, arrival_mdl, flav_gen, dur_gen, args.flav_map_fn,
        args.interval_map_fn, args.bsize_map_fn, args.range_start,
        args.range_stop, range_idx)
    with open(args.out_flavs_fn, "w") as flavs_file:
        with open(args.out_durs_fn, "w") as durs_file:
            logger.info("Running generation")
            lstm_generator(args.start_timestamp_s, args.stop_timestamp_s,
                           args.step_s, flavs_file, durs_file)


def parse_arguments():
    """Helper function to parse the command-line arguments, return as an
    'args' object.

    Args:

    None - when you call parser.parse_args(), it determines the
        cmd-line args from sys.argv.

    Returns:

        Args, an object with an attribute for everything you added
            with 'add_argument'.

    """
    # Parse the command line arguements
    parser = argparse.ArgumentParser(
        description="LSTM trace generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--arrival_model_pkl', type=str, required=True,
        help="Arrival poisson model.")
    parser.add_argument(
        '--flav_map_fn', type=str, required=True,
        help="File that maps flavors to letters for Azure.")
    parser.add_argument(
        '--interval_map_fn', type=str, required=True,
        help="File that maps durations to intervals.")
    parser.add_argument(
        '--bsize_map_fn', type=str, required=True,
        help="File that maps batch sizes to codes (include if using bsize feats).")
    parser.add_argument(
        '--start_timestamp_s', type=int, required=True,
        help="When to start the trace (inclusive)")
    parser.add_argument(
        '--stop_timestamp_s', type=int, required=True,
        help="When to stop the trace (inclusive)")
    parser.add_argument(
        '--step_s', type=int, required=False, default=STEP_S,
        help="How much to stop each time in the trace.")
    parser.add_argument(
        '--flav_model', type=str, required=True,
        help="Trained model for flavors.")
    parser.add_argument(
        '--dur_model', type=str, required=True,
        help="Trained model for durations.")
    parser.add_argument(
        '--device', type=str, required=True,
        help="Run optimization on GPU (\"cuda:0\") or on CPU (\"cpu\").")
    parser.add_argument(
        '--range_start', type=int, required=True,
        help="For time range features, when the training time range starts.")
    parser.add_argument(
        '--range_stop', type=int, required=True,
        help="For time range features, when the training time range stops.")
    parser.add_argument(
        '--out_flavs_fn', type=str, required=True,
        help="Where to store the flavor part of the generated trace")
    parser.add_argument(
        '--out_durs_fn', type=str, required=True,
        help="Where to store the duration part of the generated trace")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    MY_ARGS = parse_arguments()
    main(MY_ARGS)
