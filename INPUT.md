# Input trace data

The input traces provided in this directory are processed versions of the open source traces found at https://github.com/Azure/AzurePublicDataset/blob/master/AzurePublicDatasetV1.md (specifically, the file `vmtable.csv.gz`) and described in

> Cortez, E., Bonde, A., Muzio, A., Russinovich, M., Fontoura, M.,
and Bianchini, R. Resource central: Understanding and predicting
workloads for improved resource management in large cloud platforms.
In Proceedings of the 26th Symposium on Operating Systems Principles
(2017), pp. 153–167.

The original data was shared under the following Creative Commons Attribution 4.0 International Public License: https://github.com/Azure/AzurePublicDataset/blob/master/LICENSE

Flavor traces in the `resources/traces` folder are named as `v1.train.txt`, `v1.dev.txt`, `v1.test.txt` which correspond to the train-dev-test splits described in our paper.  An example snippet of the trace looks like:

```
300 g,|,g,|,b,m,|,b,|,g,g,|,b,|,b,|,g,|,b,|,b,b,b,b,b,b,b,b,|,m,|
600 g,|,a,|,g,g,g,k,k,k,k,g,g,g,g,g,g,g,|
900 g,g,g,g,|,b,|,b,|,b,|,g,g,|,b,|,a,|,k,|,k,|
```

where the first column of each line is a relative timestamp in seconds with respect to the start of the trace, and the remaining columns represent the sequence of VM flavors that have arrived since the timestamp on the previous line. Each flavor type is represented by an alphabetic character whose actual <vCPU core, Memory GB> tuple is given in `resources/maps/flav_map.txt`. VM arrivals in each time-bin (i.e. on each line) are comma-separated and batches of VM arrivals are separated by the `|` character, where each batch is a sequence of VMs initiated by the same tenant in the current time-bin.

Duration traces in the `resources/traces` folder are named as `v1.train.c1800000.txt`, `v1.dev.c2100000.txt`, and `v1.test.c2591400.txt` (where, for example, the `c1800000` in `v1.train.c1800000.txt` corresponds to the relative timestamp at which the durations are right-censored in that trace). An example snippet of the duration trace looks like:

```
300 U15,|,U14,|,U1,U0,|,C46,|,U1,U7,|,U2,|,U12,|,U45,|,U1,|,U4,U11,U7,U12,U9,U4,U7,U6,|,C46,|
600 U16,|,U7,|,U3,U1,U0,U0,U3,U3,U4,U1,U0,U4,U1,U3,U4,U1,|
900 U9,U9,U4,U4,|,C46,|,U36,|,C46,|,U4,U1,|,U3,|,U1,|,U6,|,U25,|
```

where the timestamp and sequence of durations directly correspond to the flavor sequences in the partner flavor trace. Each duration is composed of a character (`U` or `C`) followed by a duration-bin index (rather than the real-valued duration). Take, for example, `U15`: the `U` indicates that the duration of that flavor is uncensored, meaning the VM started and finished before the right boundary of the trace (e.g. `1800000` for `v1.train.c1800000.txt`); and the `15` means the real-valued duration falls in the 15th duration-bin, whose boundaries are given in `resources/maps/v1.interval_map.txt`. In cases where the bin index is prefixed by a `C`, the `C` indicates that the duration of that flavor is censored, meaning the VM started before the right boundary of the trace, but was still running at the right boundary of the trace (so its true finish time is unknown). The duration-bin index for censored durations conveys the amount of time the VM has been running at the time boundary of the trace.
