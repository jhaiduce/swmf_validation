# swmf_validation

swmf_validation provides functions that are useful for validating SWMF output with observational data

## Getting started

### Prerequisites

- python 2.7
- scipy
- spacepy
- cdaweb (https://github.com/jhaiduce/cdaweb)

### Installing

```shell
python setup.py install
```

## Modules

### build_imfinput

Script that downloads solar wind observations from the OMNI datset (https://omniweb.gsfc.nasa.gov/) for a specified time period and compiles them into an IMF input file for SWMF.

Example:

```shell
python -m swmf_validation.build_imfinput 2018-01-01 2018-01-02 sw_input_20180101.dat
```

Will create the file sw_input_20180101.dat, which contains OMNI solar wind data for January 1st, 2018.

### metrics

Provides the function interp_timeseries, which interpolates a time-series datset to a new set of times, as well as a number of functions for computing various forecast metrics.
