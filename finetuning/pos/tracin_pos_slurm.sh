#!/bin/bash
#
#SBATCH --job-name=tracin-pos
#SBATCH -p gpu --gres=gpu:titanrtx:1
#SBATCH --ntasks=1 --cpus-per-task=10 --mem=96000M

export TOKENIZERS_PARALLELISM=false
source ~/icml23-venv/bin/activate

export BASE_DIR=$1
export MODEL_DIRS=$(find final_models/${BASE_DIR}/*/* -maxdepth 0 -type d)

for model_dir in $MODEL_DIRS; do
  echo "Starting evaluation for $model_dir"
  python tracin_pos.py \
    --data_dir=../../data/ud-treebanks-v2.8 \
    --train_languages=fr,it,zh,ja,pt,th,tr \
    --eval_languages=de,es,ar,hi,id,ko,ru \
    --max_seq_length=128 \
    --overwrite_cache \
    --model_name_or_path=xlm-roberta-base \
    --tokenizer_name=xlm-roberta-base \
    --use_pud \
    --per_lang_max_examples=1000 \
    --base_dir=${model_dir}
done