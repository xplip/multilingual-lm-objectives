#!/bin/bash
#
#SBATCH --job-name=xnli-finetuning-nodp
#SBATCH -p gpu --gres=gpu:titanrtx:1
#SBATCH --ntasks=1 --cpus-per-task=10 --mem=96000M

export WANDB_API_KEY=ANONYMIZED
export WANDB_PROJECT=ANONYMIZED

export TOKENIZERS_PARALLELISM=false
source ~/icml23-venv/bin/activate

while :; do
    case $1 in
    --langs)
        export LANGS="$2"
        shift
        ;;
    --bs)
        export BS="$2"
        shift
        ;;
    --lr)
        export LR="$2"
        shift
        ;;
    --ga)
        export GRADIENT_ACC="$2"
        shift
        ;;
    --train_layers)
        export TRAIN_LAYERS="$2"
        shift
        ;;
    --sigma)
        export SIGMA="$2"
        shift
        ;;
    --epsilon)
        export EPSILON="$2"
        shift
        ;;
    --max_grad_norm)
        export MAX_GRAD_NORM="$2"
        shift
        ;;
    --freeze_base_model)
        export FREEZE_BASE_MODEL=1
        ;;
    --use_opacus_sampler)
        export OPACUS_SAMPLER=1
        ;;
    --differentially_private)
        export DIFFERENTIALLY_PRIVATE=1
        ;;
    *) break ;;
    esac
    shift
done

export LANGS_STR=${LANGS//,/-}
export RUN_NAME="xnli${LANGS_STR}"

ARGS=""
if [[ "$FREEZE_BASE_MODEL" -eq 1 ]]; then
    ARGS+="--freeze_base_model "
    RUN_NAME+="-fr-${TRAIN_LAYERS}l"
else
    RUN_NAME+="-full"
fi

if [[ "$DIFFERENTIALLY_PRIVATE" -eq 1 ]]; then
    ARGS+="--differentially_private "
    RUN_NAME+="-dp"
else
    RUN_NAME+="-nodp"
fi
if [[ "$OPACUS_SAMPLER" -eq 1 ]]; then
    ARGS+="--use_opacus_sampler "
    RUN_NAME+="-osamp"
else
    RUN_NAME+="-rsamp"
fi
if [[ -n "$MAX_GRAD_NORM" ]]; then
    ARGS+="--max_per_sample_grad_norm ${MAX_GRAD_NORM} "
    RUN_NAME+="-gn${MAX_GRAD_NORM}"
fi
if [[ -n "$SIGMA" ]]; then
    ARGS+="--sigma ${SIGMA} "
    RUN_NAME+="-sig${SIGMA}"
fi
if [[ -n "$EPSILON" ]]; then
    ARGS+="--target_epsilon ${EPSILON} "
    RUN_NAME+="-eps${EPSILON}"
fi
if [[ -n "$DROPOUT_PROB" ]]; then
    ARGS+="--dropout_prob ${DROPOUT_PROB} "
    RUN_NAME+="-drop${DROPOUT_PROB}"
fi

#RUN_NAME+="-${LR}-${BS}-${GRADIENT_ACC}"
RUN_NAME+="-${BS}-${GRADIENT_ACC}"

echo "Run name: ${RUN_NAME}"

for lr in "1e-4" "2e-4" "9e-5"; do
  for seed in {1..3}; do
    echo "Starting experiment with lr ${lr} and random seed $seed"
    python run_xnli.py \
      --model_name_or_path=xlm-roberta-base \
      --tokenizer_name=xlm-roberta-base \
      --train_languages=es,fr,bg,vi,tr,hi,zh \
      --eval_languages=ar,de,ru,el,sw,th,ur \
      --seed=${seed} \
      --output_dir=final_models/${RUN_NAME}/${lr}/${seed} \
      --run_name=${RUN_NAME}-${lr}-s${seed} \
      --do_train \
      --do_eval \
      --do_predict \
      --per_device_train_batch_size=${BS} \
      --per_device_eval_batch_size=256 \
      --gradient_accumulation_steps=${GRADIENT_ACC} \
      --num_train_epochs=30 \
      --warmup_steps=0 \
      --evaluation_strategy=steps \
      --eval_steps=200 \
      --logging_steps=10 \
      --save_steps=100 \
      --save_total_limit=5 \
      --max_seq_length=128 \
      --block_size=128 \
      --learning_rate=${lr} \
      --weight_decay=0.01 \
      --adam_epsilon=1e-6 \
      --overwrite_output_dir \
      --overwrite_cache \
      --evaluate_during_training \
      --do_eval_sentence_retrieval \
      --trainable_encoder_layers=${TRAIN_LAYERS} \
      --eval_config_path=../../evaluation/configs/xnli/tedtalks2020.json \
      $(echo $ARGS)
  done
done