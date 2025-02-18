#!/bin/bash
#
#SBATCH --job-name=pos-eval
#SBATCH -p gpu --gres=gpu:titanrtx:1
#SBATCH --ntasks=1 --cpus-per-task=10 --mem=96000M

export TOKENIZERS_PARALLELISM=false
source ~/icml23-venv/bin/activate

export BASE_DIR=$1
export MODEL_DIRS=$(find final_models/${BASE_DIR}/*/* -maxdepth 0 -type d)

for model_dir in $MODEL_DIRS; do
  echo "Starting evaluation for $model_dir"
  python ../../evaluation/evaluate.py \
    --model_name_or_path=${model_dir} \
    --tokenizer_name=xlm-roberta-base \
    --eval_config=../../evaluation/configs/pos/tedtalks2020.json

  python ../../evaluation/evaluate.py \
    --model_name_or_path=${model_dir} \
    --tokenizer_name=xlm-roberta-base \
    --eval_config=../../evaluation/configs/pos/wikimatrix.json

  python ../../evaluation/evaluate.py \
    --model_name_or_path=${model_dir} \
    --tokenizer_name=xlm-roberta-base \
    --eval_config=../../evaluation/configs/pos/tatoeba.json
done
