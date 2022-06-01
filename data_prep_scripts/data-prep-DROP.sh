#!/bin/env bash
#SBATCH --job-name=roberta_numnet_plus
source activate numnet_venv_1_0

set -xe

TMSPAN=$1
DATA_DIR=$2

BASE_DIR=.

CODE_DIR=${BASE_DIR}

if [ ${TMSPAN} = tag_mspan ]; then
  echo "Use tag_mspan model..."
  CACHED_TRAIN=${DATA_DIR}/tmspan_cached_roberta_drop_train.pkl
  CACHED_DEV=${DATA_DIR}/tmspan_cached_roberta_drop_dev.pkl
  MODEL_CONFIG="--tag_mspan"
  if [ \( ! -e "${CACHED_TRAIN}" \)  -o \( ! -e "${CACHED_DEV}" \) ]; then
  echo "Preparing cached data."
  python prepare_roberta_data.py --data_type drop --input_path ${DATA_DIR} --roberta_path ./synthetic_data/roberta.large --output_dir ${DATA_DIR} --tag_mspan
  fi
else
  echo "Use mspan model..."
  CACHED_TRAIN=${DATA_DIR}/cached_roberta_drop_train.pkl
  CACHED_DEV=${DATA_DIR}/cached_roberta_drop_dev.pkl
  MODEL_CONFIG=""
  if [ \( ! -e "${CACHED_TRAIN}" \)  -o \( ! -e "${CACHED_DEV}" \) ]; then
  echo "Preparing cached data."
  python prepare_roberta_data.py --data_type drop --input_path ${DATA_DIR} --roberta_path ./synthetic_data/roberta.large --output_dir ${DATA_DIR}
  fi
fi
