#!/bin/bash

set -e

Q=$1
DS_DIR=${2%/}  # remove trailing slash!

if [[ -z $Q || -z $DS_DIR ]]; then
  echo "USAGE: $0 Q DS_DIR [WAIT]"
  exit 1
fi

WAIT=$3
if [ -z $WAIT ]; then
  WAIT=1
fi

function task_array() {
    NUM=$1; shift
    NAME=$1; shift
    if [[ -z $QSUB ]]; then
      bash "$@"
    else
      LOG_DIR=~/net_scratch/task_array_logs/
      qsub -t 1-$NUM -cwd -j y -sync y -N $NAME -o $LOG_DIR -hold_jid $WAIT "$@"
    fi
}

DS_NAME=$(basename $DS_DIR)

if [ ! -f $DS_DIR/cached_glob.pkl ]; then
  echo "$DS_NAME: Caching $DS_DIR..."
  task_array 16 cache_in  task_array_cached.sh $DS_DIR --create_without_shitty
fi

echo "$DS_NAME: BPG Q=$Q"
task_array 50 bpg_q$Q task_array_bpg.sh $Q $DS_DIR $(dirname $DS_DIR) --discard_shitty=0

if [[ $Q == "A"* ]]; then
  R="${Q/A/}"
  R=$(echo $R| sed 's/_/ /')
  SEQ=$(seq $R)
else
  SEQ=$Q
fi
for Q in $SEQ; do
  BPG_DIR=${DS_DIR}_bpg_q$Q
  if [[ ! -d $BPG_DIR ]]; then
    echo "ERROR, expected $BPG_DIR"
    continue
  fi

  echo "$DS_NAME: Caching $BPG_DIR..."
  task_array 16 cache_out task_array_cached.sh $BPG_DIR --create_without_shitty

  BPG_DIR_TMP=${BPG_DIR}_bpg_out_tmp
  rm -rfv $BPG_DIR_TMP
done

