#!/bin/bash

set -e

if [[ -z $1 ]]; then
    echo "USAGE: $0 DATA_DIR"
    exit 1
fi

DATA_DIR=$(realpath $1)
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

progress () {
    COUNTER=0
    while read LINE; do
        COUNTER=$((COUNTER+1))
        if [[ $((COUNTER % 10)) == 0 ]]; then
            echo -ne "\rExtracting $LINE; Unpacked $COUNTER files."
        fi
    done
    echo ""
}

echo "DATA_DIR=$DATA_DIR; SCRIPT_DIR=$SCRIPT_DIR"

mkdir -pv $DATA_DIR

TRAIN_0=train_0
TRAIN_1=train_1
TRAIN_2=train_2
VAL=validation
TEST=test

# Download ----------
DOWNLOAD_DIR=$DATA_DIR/download
mkdir -p $DOWNLOAD_DIR
pushd $DOWNLOAD_DIR
for DIR in $TRAIN_0 $TRAIN_1 $TRAIN_2 $VAL $TEST; do
    TAR=${DIR}.tar.gz
    if [ ! -f "$TAR" ]; then
        echo "Downloading $TAR..."
        aws s3 --no-sign-request cp s3://open-images-dataset/tar/$TAR $TAR
    else
        echo "Found $TAR..."
    fi
done

# for DIR in $TRAIN_0 $TRAIN_1 $TRAIN_2 $VAL $TEST; do
#     TAR=${DIR}.tar.gz
#     if [ -d $DIR ]; then
#         echo "Found $DIR, not unpacking $TAR..."
#         continue
#     fi
#     if [ ! -f $TAR ]; then
#         echo "ERROR: Expected $TAR in $DOWNLOAD_DIR"
#         exit 1
#     fi
#     echo "Unpacking $TAR..."
#     ( tar xvf $TAR | progress ) &
# done
# wait
echo "Unpacked all!"
popd

# Convert ----------
FINAL_TRAIN_DIR=$DATA_DIR/train_oi
FINAL_VAL_DIR=$DATA_DIR/val_oi
FINAL_TEST_DIR=$DATA_DIR/test_oi

function task_array() {
    NUM=$1; shift
    echo qsub -t 1-$NUM -cwd -j y -o /scratch_net/fryr/mentzerf/task_array_logs "$@"
}


OUT_DIR=$DATA_DIR/imported
pushd $SCRIPT_DIR

echo "Resizing train..."
task_array 100 task_array_import.sh $DOWNLOAD_DIR $TRAIN_0 $TRAIN_1 $TRAIN_2 \
        --out_dir_clean=$FINAL_TRAIN_DIR \
        --out_dir_discard=$OUT_DIR/discard_train

echo "Resizing val..."
task_array 50 task_array_import.sh $DOWNLOAD_DIR $VAL \
        --out_dir_clean=$FINAL_VAL_DIR \
        --out_dir_discard=$OUT_DIR/discard_val

echo "Resizing test..."
task_array 50 task_array_import.sh $DOWNLOAD_DIR $TEST \
        --out_dir_clean=$FINAL_TEST_DIR \
        --out_dir_discard=$OUT_DIR/discard_test

# # Update Cache ----------
# CACHE_P=$DATA_DIR/cache.pkl
# export PYTHONPATH=$(pwd)
#
# echo "Updating cache $CACHE_P..."
# python dataloaders/images_loader.py update $FINAL_TRAIN_DIR "$CACHE_P" --min_size 128
# python dataloaders/images_loader.py update $FINAL_VAL_DIR "$CACHE_P" --min_size 128

echo "----------------------------------------"
echo "Done"
# echo "To train, you MUST UPDATE configs/dl/oi.cf:"
# echo ""
# echo "  image_cache_pkl = '$1/cache.pkl'"
# echo "  train_imgs_glob = '$(realpath $1/train_oi)'"
# echo "  val_glob = '$(realpath $1/val_oi)'"
# echo ""
echo "----------------------------------------"
