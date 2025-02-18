#!/bin/bash
#
#SBATCH --job-name=tracin-xnli
#SBATCH -p gpu --gres=gpu:titanrtx:1
#SBATCH --ntasks=1 --cpus-per-task=10 --mem=60000M

export TOKENIZERS_PARALLELISM=false
source ~/icml23-venv/bin/activate

export BASE_DIR=$1
export MODEL_DIRS=$(find final_models/${BASE_DIR}/*/* -maxdepth 0 -type d)

for model_dir in $MODEL_DIRS; do
  echo "Starting evaluation for $model_dir"
  python tracin_xnli.py \
    --train_languages=es,fr,bg,vi,tr,hi,zh \
    --eval_languages=ar,de,ru,el,sw,th,ur \
    --max_seq_length=128 \
    --overwrite_cache \
    --model_name_or_path=xlm-roberta-base \
    --tokenizer_name=xlm-roberta-base \
    --max_train_samples=1000 \
    --base_dir=${model_dir}
done