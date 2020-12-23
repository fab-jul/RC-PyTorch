#!/bin/bash

set -e

MODELS_DIR=$1

if [[ -z $MODELS_DIR ]]; then
  echo "USAGE: $0 MODELS_DIR"
  exit 1
fi

mkdir -p $MODELS_DIR
pushd $MODELS_DIR

wget http://data.vision.ee.ethz.ch/mentzerf/rc_models/clf_1115_1729.tar.gz
tar xvf clf_1115_1729.tar.gz
rm clf_1115_1729.tar.gz

wget http://data.vision.ee.ethz.ch/mentzerf/rc_models/1109_1715.tar.gz
tar xvf 1109_1715.tar.gz
rm 1109_1715.tar.gz

