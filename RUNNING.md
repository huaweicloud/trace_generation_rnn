# How to run

## Training/Testing

Our trace generator consists of the three following stages, each of which is trained separately. More details on usage
of each script can be obtained by its help function (for example, `python -m tragegen_rnn.arrivals --help`).

The input data is in the format described in [INPUT.md](INPUT.md).

### Training/Testing: Poisson Arrival Model

```
python -m tracegen_rnn.arrivals --train resources/traces/v1.train.txt --test resources/traces/v1.test.txt --regularization 1e-2 --range_start 0 --range_stop 1800000 --out_model_pickle resources/models/test_arrival_model.pkl --nrange_samps 50 --npoisson_samps 50
```

Should get around 82.7%, up to 83.0% if using 500/500 samples

### Training/Testing: Flavor LSTM

```
python -m tracegen_rnn.train_flav_lstm --flav_map_fn resources/maps/flav_map.txt --train_flavs resources/traces/v1.train.txt --test_flavs resources/traces/v1.dev.txt --device cuda:0 --seq_len 500 --batch_size 100 --range_start 0 --range_stop 1800000 --max_iters 10 --lr 5e-3 --weight_decay 1e-5 --nlayers 2 --nhidden 200 --model_save_fn resources/models/test_flav_model.pt

python -m tracegen_rnn.evaluate_flav_lstm --flav_map_fn resources/maps/flav_map.txt --test_flavs resources/traces/v1.test.txt --device cuda:0 --seq_len 500 --batch_size 100 --range_start 0 --range_stop 1800000 --lstm_model resources/models/test_flav_model.pt
```

Should get a NLL of around 0.67, with a 1-Best-Err rate of 26%.

### Training/Testing: Duration LSTM

```
python -m tracegen_rnn.train_dur_lstm --flav_map_fn resources/maps/flav_map.txt --interval_map_fn resources/maps/v1.interval_map.txt --bsize_map_fn resources/maps/batch_size_map.txt --train_flavs resources/traces/v1.train.txt --train_durs resources/traces/v1.train.c1800000.txt --test_flavs resources/traces/v1.dev.txt --test_durs resources/traces/v1.dev.c2100000.txt --device cuda:0 --seq_len 500 --batch_size 100 --range_start 0 --range_stop 1800000 --max_iters 60 --lr 5e-3 --weight_decay 5e-6 --nlayers 2 --nhidden 256 --model_save_fn resources/models/test_dur_model.pt

python -m tracegen_rnn.evaluate_dur_lstm --flav_map_fn resources/maps/flav_map.txt --interval_map_fn resources/maps/v1.interval_map.txt --bsize_map_fn resources/maps/batch_size_map.txt --test_flavs resources/traces/v1.test.txt --test_durs resources/traces/v1.test.c2591400.txt --device cuda:0 --seq_len 500 --batch_size 100 --range_start 0 --range_stop 1800000 --lstm_model resources/models/test_dur_model.pt 
```

Should get a BCE of around 0.129 and a 1-Best-Err rate of 29%.

## Generation

```
python -m tracegen_rnn.generator --arrival_model_pkl resources/models/test_arrival_model.pkl --flav_map_fn resources/maps/flav_map.txt --interval_map_fn resources/maps/v1.interval_map.txt --bsize_map_fn resources/maps/batch_size_map.txt --device cpu --start_timestamp_s 2100000 --stop_timestamp_s 2130000 --flav_model resources/models/test_flav_model.pt --dur_model resources/models/test_dur_model.pt --range_start 0 --range_stop 1800000 --out_flavs_fn tmp.flavs --out_durs_fn tmp.durs
```

The output of `tracegen_rnn.generator` is given in two files, one describing the sequence of generated flavors (i.e. `tmp.flavs` above) and another describing the sequence of generated durations for those flavors (i.e. `tmp.durs` above) in the same format as the input traces for flavors and durations as described in [INPUT.md](INPUT.md).
