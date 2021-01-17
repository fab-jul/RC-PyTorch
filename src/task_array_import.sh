#!/bin/bash

set -e

source ethrc
export PYTHONPATH=$(pwd)

python -u import_train_images.py "$@"