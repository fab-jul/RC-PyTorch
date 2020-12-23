#!/bin/bash

set -e

source ethrc
export PYTHONPATH=$(pwd)

if [[ -z $QSUB ]]; then
    python -u dataloaders/cached_listdir_imgs.py "$@"
else
    python -u dataloaders/cached_listdir_imgs.py "$@" --distributed_create
fi

