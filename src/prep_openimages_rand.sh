#!/bin/bash

set -e

if [[ -z $1 ]]; then
    echo "USAGE: $0 DATA_DIR [OUT_DIR]"
    exit 1
fi

Q=R12_14
DATA_DIR=$(realpath $1)
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

if [[ -n $2 ]]; then
  OUT_DIR=$2
else
  OUT_DIR=$DATA_DIR
fi

echo "Saving to $OUT_DIR..."

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

for DIR in $TRAIN_0 $TRAIN_1 $TRAIN_2 $VAL $TEST; do
   TAR=${DIR}.tar.gz
   if [ -d $DIR ]; then
       echo "Found $DIR, not unpacking $TAR..."
       continue
   fi
   if [ ! -f $TAR ]; then
       echo "ERROR: Expected $TAR in $DOWNLOAD_DIR"
       exit 1
   fi
   echo "Unpacking $TAR..."
   ( tar xvf $TAR | progress ) &
done
wait
echo "Unpacked all!"
popd

# Convert ----------
FINAL_TRAIN_DIR=$OUT_DIR/train_oi_r
FINAL_VAL_DIR=$OUT_DIR/val_oi_500_r
FINAL_TEST_DIR=$OUT_DIR/test_oi_500_r

function task_array() {
    NUM=$1; shift
    NAME=$1; shift
    qsub -t 1-$NUM -cwd -j y -sync y -N $NAME -o ~/net_scratch/task_array_logs/ -hold_jid 4687252 "$@"
}

DISCARD=$OUT_DIR/discard
pushd $SCRIPT_DIR

#echo "Resizing train..."
#task_array 100 resize task_array_import.sh $DOWNLOAD_DIR $TRAIN_0 $TRAIN_1 $TRAIN_2 \
#       --out_dir_clean=$FINAL_TRAIN_DIR \
#       --out_dir_discard=$DISCARD \
#       --random_scale=512 \
#       --filter_with_dir=/scratch_net/fryr/mentzerf/datasets/openimages_new/train_oi

#bash prep_bpg_ds.sh $Q $FINAL_TRAIN_DIR

echo "Resizing test_2k..."
TMP_TEST_DIR=$OUT_DIR/test_2k_resized
task_array 50 resize_test task_array_import.sh $DOWNLOAD_DIR test_2k  \
  --out_dir_clean=$TMP_TEST_DIR \
  --out_dir_discard=$DISCARD \
  --downsampling=lanczos

echo "Got $(ls $TMP_TEST_DIR | wc -l) images!"

python make_validation_set.py $TMP_TEST_DIR $FINAL_TEST_DIR

#echo "Resizing validation_500..."
#task_array 50 resize_val task_array_import.sh $DOWNLOAD_DIR validation_500 \
#  --out_dir_clean=$FINAL_VAL_DIR \
#  --out_dir_discard=$DISCARD \
#  --downsampling=lanczos
#
#NUM_IMGS=$(ls $FINAL_VAL_DIR/*.png | wc -l)
#if [[ $NUM_IMGS != 500 ]]; then
#  echo $NUM_IMGS
#  exit 1
#fi
#
#bash prep_bpg_ds.sh $Q $FINAL_VAL_DIR

### echo "Resizing val..."
### task_array 50 task_array_import.sh $DOWNLOAD_DIR $VAL \
###         --out_dir_clean=$FINAL_VAL_DIR \
###         --out_dir_discard=$OUT_DIR/discard_val
###
### echo "Resizing test..."
### task_array 50 task_array_import.sh $DOWNLOAD_DIR $TEST \
###         --out_dir_clean=$FINAL_TEST_DIR \
###         --out_dir_discard=$OUT_DIR/discard_test
##
### # Update Cache ----------
### CACHE_P=$DATA_DIR/cache.pkl
### export PYTHONPATH=$(pwd)
###
### echo "Updating cache $CACHE_P..."
### python dataloaders/images_loader.py update $FINAL_TRAIN_DIR "$CACHE_P" --min_size 128
### python dataloaders/images_loader.py update $FINAL_VAL_DIR "$CACHE_P" --min_size 128

echo "----------------------------------------"
echo "Done"
echo "$FINAL_TRAIN_DIR"
echo "$FINAL_VAL_DIR"
echo "$FINAL_TEST_DIR"
# echo "To train, you MUST UPDATE configs/dl/oi.cf:"
# echo ""
# echo "  image_cache_pkl = '$1/cache.pkl'"
# echo "  train_imgs_glob = '$(realpath $1/train_oi)'"
# echo "  val_glob = '$(realpath $1/val_oi)'"
# echo ""
echo "----------------------------------------"
