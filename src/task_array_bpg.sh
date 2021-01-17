#!/bin/bash

echo "$@"

Q=$1
shift
INP=$1
shift
OTP_BASE=$1
shift

echo $Q $INP $OTP_BASE

if [[ -z $Q || -z $INP || -z $OTP_BASE ]]; then
    echo "USAGE: $0 Q INP OTP_BASE"
    exit 1
fi

if [[ ! -f compressor.py ]]; then
    echo "ERROR: Run this script from src/ directory!"
    exit 1
fi

bpgenc -h >/dev/null
if [[ $? != 1 ]]; then
    echo 'ERROR: bgpenc seems to not be in $PATH!'
    exit 1
fi

set -e

source ethrc
export PYTHONPATH=$(pwd)

OTP_NAME=$(basename ${INP%/})
OTP="${OTP_BASE%/}/${OTP_NAME}"

python -u compressor.py bpg $INP $OTP --params q=$Q "$@"

echo "SUCCESS / OTP=$OTP"
