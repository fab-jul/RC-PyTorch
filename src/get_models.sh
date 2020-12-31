#!/bin/bash

set -e

MODELS_DIR=$1

if [[ -z $MODELS_DIR ]]; then
  echo "USAGE: $0 MODELS_DIR"
  exit 1
fi

mkdir -p $MODELS_DIR
pushd $MODELS_DIR

NAME_MAIN_MODEL="gdn_wide_deep3 new_oi_q12_14_128 unet_skip"
NAME_CLF_MODEL="clf@model1715 exp_min=6.25e-06 lr.initial=0.0001 lr.schedule=exp_0.25_i50000 n_resblock=4"
NAME_QHIST="q_histories"

if [[ ! -d "$NAME_MAIN_MODEL" ]]; then
  wget http://data.vision.ee.ethz.ch/mentzerf/rc_models/clf_1115_1729.tar.gz
  tar xvf clf_1115_1729.tar.gz
  rm clf_1115_1729.tar.gz
fi

if [[ ! -d "$NAME_CLF_MODEL" ]]; then
  wget http://data.vision.ee.ethz.ch/mentzerf/rc_models/1109_1715.tar.gz
  tar xvf 1109_1715.tar.gz
  rm 1109_1715.tar.gz
fi

if [[ ! -d "$NAME_QHIST" ]]
  wget http://data.vision.ee.ethz.ch/mentzerf/rc_models/q_histories.tar.gz
  tar xvf q_histories.tar.gz
rm q_histories.tar.gz
fi
