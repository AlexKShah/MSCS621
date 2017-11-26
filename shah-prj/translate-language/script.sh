#pip3 install tensor2tensor --user
#pip3 install tensor2tensor[tensorflow_gpu] --user
#pip3 install tensor2tensor[tensorflow] --user

#t2t-trainer --registry_help

#HERE="~/git/MSCS692/shah-prj/working/"

PROBLEM=shah_en_es
MODEL=transformer
HPARAMS=transformer_base_single_gpu

DATA_DIR=data/
TMP_DIR=temp/
TRAIN_DIR=train/$PROBLEM/$MODEL-$HPARAMS/
CUSTOM_DIR=custom/$PROBLEM/

mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR $CUSTOM_DIR

# Generate
t2t-datagen \
  --data_dir=$DATA_DIR \
  --tmp_dir=$TMP_DIR \
  --problem=$PROBLEM \
  --t2t_usr_dir=$CUSTOM_DIR

# Train
# OOM: --hparams='batch_size=1024'.
t2t-trainer \
  --data_dir=$DATA_DIR \
  --problems=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR \
  --t2t_usr_dir=$CUSTOM_DIR
  
# Decode
DECODE_FILE=$DATA_DIR/decode_this.txt
echo "Hello world" >> $DECODE_FILE
echo "Goodbye world" >> $DECODE_FILE

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
