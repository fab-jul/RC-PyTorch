# RC-PyTorch

<div align="center">
  <img src='figs/overview.png' width="80%"/>
</div>

## Learning Better Lossless Compression Using Lossy Compression

## [[Paper]](https://arxiv.org/abs/2003.10184) [[Slides]](https://data.vision.ee.ethz.ch/mentzerf/rc/2008-slides.pdf)

## Abstract

We leverage the powerful lossy image compression algorithm BPG to build a
lossless image compression system. Specifically, the original image is first
decomposed into the lossy reconstruction obtained after compressing it with BPG
and the corresponding residual.  We then model the distribution of the residual
with a convolutional neural network-based probabilistic model that is
conditioned on the BPG reconstruction, and combine it with entropy coding to
losslessly encode the residual. Finally, the image is stored using the
concatenation of the bitstreams produced by BPG and the learned residual coder.
The resulting compression system achieves state-of-the-art performance in
learned lossless full-resolution image compression, outperforming previous
learned approaches as well as PNG, WebP, and JPEG2000.

# About the Code

The released code is very close to what we used when running experiments
for the paper. The codebase evolved from the [L3C repo](https://github.com/fab-jul/L3C-PyTorch/)
and contains code that is unused in this paper.

#### Naming

Files specific to this paper usually are marked "enhancement" or "enh" as 
this was the internal name. We frequently use the following terms:

- `x_r` / "raw" / "ground-truth" (gt) image: The input image, uncompressed. Note that this is inconsistent with the paper, where `x_r` is the residual
- `x_l` / "lossy" / "compressed" image: The image obtained by feeding the raw image through BPG.
- `res`: Residual between raw and lossy.

# Setup Environment

## Folder structure

The following is the suggested structure that is assumed by this README, but any
other structure is supported:

```
$RC_ROOT/
    datasets/       <-- See the "Preparing datasets" section.
    models/         <-- The result of get_models.sh, see below.
    RC-PyTorch/     <-- The result of a git clone of this repo.
        src/
        figs/
        README.md   <-- This file.
        ...
```

To set this up:

```bash
RC_ROOT="/path/to/wherever/you/want"
mkdir -p "$RC_ROOT"
mkdir -p "$RC_ROOT/datasets"
mkdir -p "$RC_ROOT/models"
pushd $RC_ROOT
git clone https://github.com/fab-jul/RC-PyTorch
```

## BPG

To run the code in this repo, you need BPG.
Follow the instractions on http://bellard.org/bpg/ to install it.

Afterwards, make sure `bpgenc` and `bpgdec` is in `$PATH` 
by running `bpgenc` in your console.
You can also run the following script to test:

```bash
pushd $RC_ROOT/RC-PyTorch/src
bash test_bpg_available.sh
```

## Python Environment

To run the code, you need Python 3.7 or newer, and PyTorch 1.1.0. 
The following assumes you use [conda](https://docs.conda.io/en/latest/miniconda.html)
to set this up:

```bash
NAME=pyt11  # You can change this to whatever you want.
conda create -n $NAME python==3.7 pip -y
conda activate $NAME

# Note that we're using PyTorch 1.1 and CUDA 10.0. 
# Other combinations may also work but have not been tested!
conda install pytorch==1.1.0 torchvision cudatoolkit==10.0.130 -c pytorch

# Install the pip requirements
pushd $RC_ROOT/RC-PyTorch/src
pip install -r requirements.txt
```

# Preparing Evaluation Datasets

We preprocess our datasets by compressing each image with a variety of
BPG quantization levels, as described in the following. Since the Q-classifier
has been trained for `Q \in {9, ..., 17}`, that's the range we use.
The pre-processing with all these Q is an artifact of the fact that this code
was used for experimentation also. In the "real world", we would run
BPG only with the Q that is output by the Q-classifier.

### CLIC.mobile and CLIC.pro

```bash
pushd "$RC_ROOT/datasets"

wget https://data.vision.ee.ethz.ch/cvl/clic/mobile_valid_2020.zip
unzip mobile_valid_2020.zip  # Note: if you don't have unzip, use "jar xf"
mv valid mobile_valid  # Give the extracted archive a unique name

wget https://data.vision.ee.ethz.ch/cvl/clic/professional_valid_2020.zip
unzip professional_valid_2020.zip
mv valid professional_valid  # Give the extracted archive a unique name
```

Preprocess with BPG.

```bash
pushd "$RC_ROOT/RC-PyTorch/src"
bash prep_bpg_ds.sh A9_17 $RC_ROOT/datasets/mobile_valid
bash prep_bpg_ds.sh A9_17 $RC_ROOT/datasets/professional_valid
```

### Open Images Validation 500

For Open Images, we use the same validation set that we used for [L3C](TODO):

```bash
pushd "$RC_ROOT/datasets"
wget http://data.vision.ee.ethz.ch/mentzerf/validation_sets_lossless/val_oi_500_r.tar.gz
mkdir val_oi_500_r && pushd val_oi_500_r
tar xvf ../val_oi_500_r.tar.gz

pushd "$RC_ROOT/RC-PyTorch/src"
bash prep_bpg_ds.sh A9_17 $RC_ROOT/datasets/val_oi_500_r
```

### DIV2K

```bash
pushd "$RC_ROOT/datasets"
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip
unzip DIV2K_valid_HR.zip

pushd "$RC_ROOT/RC-PyTorch/src"
bash prep_bpg_ds.sh A9_17 $RC_ROOT/datasets/DIV2K_valid_HR
```

# Running models used in the paper

### Download models

Download our models with `get_models.sh`:

```bash
MODELS_DIR="$RC_ROOT/models"
bash get_models.sh "$MODELS_DIR"
```

### Get bpsp on Open Images Validation 500

After downloading and preparing Open Images as above, and 
downloading the models, you can 
run our model on Open Images as follows, to test if all works:

``` 
# Running on Open Iamges 500
DATASET_DIR="$RC_ROOT/datasets"
MODELS_DIR="$RC_ROOT/models"

pushd "$RC_ROOT/RC-PyTorch/src"

# Note: depending on your environment, adapt CUDA_VISIBLE_DEVICES.
CUDA_VISIBLE_DEVICES=0 python -u run_test.py \
    "$MODELS_DIR" 1109_1715 "AUTOEXPAND:$DATASET_DIR/val_oi_500_r" \
    --restore_itr 1000000 \
    --tau \
    --clf_p "$MODELS_DIR/1115_1729*/ckpts/*.pt" \
    --qstrategy CLF_ONLY
```

The first three arguments are the location of the models,
the experiment ID (`1109_1715` here), and the dataset dir. The dataset dir
is special as we prefix it with `AUTOEXPAND`, which causes the tester
to get all the bpg folders that we created in the previous step (another
artifact of this being experimentation code). Here, you can also put any
other dataset that you preprocessed similar to the above.

If all goes well, you should see the 2.790 reported in the paper:

```
testset         exp         itr      Q           bpsp
val_oi_500...   1109_1715   998500   1.393e+01   2.790
```

### Get bpsp on all evaluation datasets from the paper

You can also pass multiple datasets by separating them with commas.
For example, to run on all datasets of the paper
(assuming you downloaded them as described above):

```
# Note: depending on your environment, adapt CUDA_VISIBLE_DEVICES.
CUDA_VISIBLE_DEVICES=0 python -u run_test.py \
    "$MODELS_DIR" 1109_1715 \
    "AUTOEXPAND:$DATASET_DIR/professional_valid,AUTOEXPAND:$DATASET_DIR/DIV2K_valid_HR,AUTOEXPAND:$DATASET_DIR/mobile_valid,AUTOEXPAND:$DATASET_DIR/val_oi_500_r" \
    --restore_itr 1000000 \
    --tau \
    --clf_p "$MODELS_DIR/1115_1729*/ckpts/*.pt" \
    --qstrategy CLF_ONLY
```

Expected Output:
```
testset                 exp         itr      Q           bpsp
DIV2K_valid_HR...       1109_1715   998500   1.378e+01   3.078
mobile_valid...         1109_1715   998500   1.325e+01   2.537
professional_valid...   1109_1715   998500   1.385e+01   2.932
val_oi_500_r...         1109_1715   998500   1.393e+01   2.790
```

## Sampling

To get the sampling figures shown in the paper, pass `--sample=some/dir`
to `run_test.py`.

<!-- ## Our TB

TODO: Link our tensorboard. -->


# Training your own models


## Prepare dataset

We uploaded our training set as 10 `.tar.gz` files, which you can
download and extract with the handy `training_set_helper.py` script:

```bash
pushd "$RC_ROOT/RC-PyTorch/src"

# Tars will be downloaded to $RC_ROOT/datasets/, and then the images will
# be in $RC_ROOT/datasets/train_oi_r.
TRAIN_DATA_DIR="$RC_ROOT/datasets"
python -u training_set_helper.py download_and_unpack "$TRAIN_DATA_DIR"
```

The tar files are downloaded sequentially and the unpacking then happens
in parallel with 10 processes.

Next, you will need to compress each of these with a random quality
factor using BPG, which is done with the following snippet.
This may take a significant amount of time on a single machine,
as you compress 300k+ images. We ran this on a CPU cluster, and the code
structure for the cluster is in `task_array.py`, but likely would need significant
adaptation for your setup. So instead, if you run the following as is, the work
is split over 16 processes on the current machine, which may be bearable. 
If you have a beefier CPU, adapt `MAX_PROCESS`. 
On our single CPU, 24-core test machine it took ~10h with `MAC_PROCESS=24`.

```bash
pushd "$RC_ROOT/RC-PyTorch/src"
MAX_PROCESS=16 bash prep_bpg_ds.sh R12_14 $RC_ROOT/datasets/train_oi_r
```

You will also need a validation set. We used `DIV2K_valid_HR_crop4`,
which you can get with:

```bash
pushd "$RC_ROOT/datasets"
wget http://data.vision.ee.ethz.ch/mentzerf/rc_data/DIV2K_valid_HR_rc.tar.gz
tar xvf DIV2K_valid_HR_rc.tar.gz
```

## Train your own models

The following command was used to train the models released above:
```bash

# NOTE: This assumes that $RC_ROOT is set and that 
# DIV2K_VALID_HR_crop4 and DIV2K_VALID_HR_crop4_bpg_q12 exist
# at $RC_ROOT/datasets, as described above. If this is not the case,
# adapt the config file at configs/dl/new_oi_q12_14_128.cf.

# You can set this to whatever you want
LOGS_DIR="$RC_ROOT/models"

# Note: depending on your environment, adapt CUDA_VISIBLE_DEVICES.
CUDA_VISIBLE_DEVICES=0 python -u train.py \
    configs/ms/gdn_wide_deep3.cf \
    configs/dl/new_oi_q12_14_128.cf \
    $LOGS_DIR  \
    -p unet_skip 
```

## Train Q-Classifier

#### Get the ground truth data.

To train the Q-Classifier, we need to figure out the optimal Qs
for all images in the training set.

For the model we published (downloaded by `get_model.sh`, see above),
the ground truth data is also avaiable and was downloaded by `get_model.sh`
into `$RC_ROOT/datasets/q_histories`. 

If you trained your own model,
you need to create `q_histories` as follows:

```bash
# First make create the subset of the training set needed.
pushd "$RC_ROOT/RC-PyTorch/src"
python -u make_clf_training_set.py "$RC_ROOT/datasets/train_oi_r"

# Now compress with every image with 7 quality factors.
# This step will take a long time (~7h on a test machine).
# Set MAX_PROCESS to the number of cores of your machine for speed.
MAX_PROCESS=16 bash prep_bpg_ds.sh A11_17 "$RC_ROOT/datasets/train_oi_r_subset_clf"

# Now, we determine the optimal Qs, given a trained model:
TODO
```

Then, you need to adapt `configs/dl/clf/model1715.cf` to point to this folder.

#### Train Classifier

Given `q_histories`, the classifier can be trained with:

```bash
# You can set this to whatever you want
LOGS_DIR="$RC_ROOT/models"

# Note: depending on your environment, adapt CUDA_VISIBLE_DEVICES.
CUDA_VISIBLE_DEVICES=0 python -u train.py \
   configs/ms/clf/down2_nonorm_down.cf \
   configs/dl/clf/model1715.cf \
   $LOGS_DIR
```
 

# Citation

If you use the work released here for your research, please cite the paper:

```
@InProceedings{mentzer2020learning,
  author = {Mentzer, Fabian and Gool, Luc Van and Tschannen, Michael},
  title = {Learning Better Lossless Compression Using Lossy Compression},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2020}
}
```
