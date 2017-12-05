#Alex Shah
#T2T translation script

#install tensor2tensor:

#pip3 install tensor2tensor --user
#pip3 install tensor2tensor[tensorflow_gpu] --user
#OR
#pip3 install tensor2tensor[tensorflow] --user

#view registered problems:
#t2t-trainer --registry_help

PROBLEM=translate_enfr_wmt_small8k
MODEL=transformer
HPARAMS=transformer_base_single_gpu

DATA_DIR=data/
TMP_DIR=temp/
TRAIN_DIR=train/
CUSTOM_DIR=usr/

mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR $CUSTOM_DIR

# Generate
t2t-datagen \
  --data_dir=$DATA_DIR \
  --tmp_dir=$TMP_DIR \
  --problem=$PROBLEM \
 # --t2t_usr_dir=$CUSTOM_DIR

# Train
# OOM: --hparams='batch_size=1024'.
t2t-trainer \
  --data_dir=$DATA_DIR \
  --problems=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR \
  #--t2t_usr_dir=$CUSTOM_DIR

# Decode
DECODE_FILE=$DATA_DIR/decode_this.txt

BEAM_SIZE=4
ALPHA=0.6

t2t-decoder \
  --data_dir=$DATA_DIR \
  --problems=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR \
  --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA" \
  --decode_from_file=$DECODE_FILE

cat $DECODE_FILE.$MODEL.$HPARAMS.beam$BEAM_SIZE.alpha$ALPHA.decodes
