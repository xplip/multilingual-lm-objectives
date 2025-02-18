## Multilingual LM Objectives
This repository hosts the research code for our ICML 2023 paper [*Differential Privacy, Linguistic Fairness, and Training Data Influence: Impossibility and Possibility Theorems for Multilingual Language Models*](https://arxiv.org/abs/2308.08774).

Please note that this code is for archival purposes and is not actively maintained.

Below, we provide instructions to reproduce our experiments.

## Setup
```
# Setup virtual environment
python3 -m venv ~/icml23-venv
source ~/icml23-venv/bin/activate

pip install --upgrade pip
pip install git+https://github.com/lxuechen/private-transformers.git@d66b7ba
pip install -r requirements.txt

# Important step to resolve imports
export PYTHONPATH=$(pwd)
```

## Download data

The data can be prepared as described in [data/README.md](data/README.md).

## Fine-Tuning
The weights and biases (wandb) environment variables in the slurm scripts were anonymized for review purposes. Remember to deanonymize them to run the scripts.
Also: Adjust slurm's SBATCH parameters if necessary, or run the scripts directly in your shell without slurm.

### POS Tagging

```
cd finetuning/pos

# finetune private models with slurm
for epsilon in 1 3 8 15 30; do
  sbatch pos_slurm.sh \
  --bs 8 \
  --ga 12 \
  --train_layers 12 \
  --epsilon ${epsilon} \
  --max_grad_norm 0.1 \
  --differentially_private
done

# finetune non-private models with slurm
sbatch pos_slurm.sh \
  --bs 8 \
  --ga 12 \
  --train_layers 12 \
  --max_grad_norm 0.1
```

### XNLI

```
cd finetuning/xnli

# finetune private models with slurm
for epsilon in 1 3 8 15 30; do
  sbatch xnli_slurm.sh \
  --bs 8 \
  --ga 64 \
  --train_layers 12 \
  --epsilon ${epsilon} \
  --max_grad_norm 0.1 \
  --differentially_private
done

# finetune non-private models with slurm
sbatch xnli_slurm_nodp.sh \
  --bs 64 \
  --ga 8 \
  --train_layers 12 \
  --max_grad_norm 0.1
```

## Multilingual Compression Evaluation

This step having finetuned models as above.

### POS Tagging Models

```
cd finetuning/pos

# with slurm
export BASE_DIR=pos--full-nodp-rsamp-gn0.1-8-12 # set base dir like this as an example (should be a folder in final_models)
sbatch pos_evaluate_slurm.sh $BASE_DIR
```

### XNLI Models

```
cd finetuning/xnli

# with slurm
export BASE_DIR=xnli-full-dp-rsamp-gn0.1-eps1-8-64 # set base dir like this as an example (should be a folder in final_models)
sbatch xnli_evaluate_slurm.sh $BASE_DIR
```

## TracInCP / InfU Computation 

### POS Tagging Models

```
cd finetuning/pos

# with slurm
export BASE_DIR=pos--full-nodp-rsamp-gn0.1-8-12 # set base dir like this as an example (should be a folder in final_models)
sbatch tracin_pos_slurm.sh $BASE_DIR
```

### XNLI Models

```
cd finetuning/xnli

# with slurm
export BASE_DIR=xnli-full-dp-rsamp-gn0.1-eps1-8-64 # set base dir like this as an example (should be a folder in final_models)
sbatch tracin_xnli_slurm.sh $BASE_DIR
```


## Citation

```bibtex
@inproceedings{rust-soegaard-2023,
  title = {Differential Privacy, Linguistic Fairness, and Training Data Influence: Impossibility and Possibility Theorems for Multilingual Language Models},
  author = {Rust, Phillip and S{\o}gaard, Anders},
  booktitle = {Proceedings of the 40th International Conference on Machine Learning},
  year = {2023},
  url = {https://arxiv.org/abs/2308.08774},
}
```
