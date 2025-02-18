import copy
import glob
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from glob import glob
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from pos_dataset import Split
from run_pos import DataTrainingArguments, get_dataset
from scipy.special import softmax
from scipy.stats import entropy
from torch import nn
from torch.autograd import grad
from torch.utils.data import Dataset, SequentialSampler
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizer,
    TrainerState,
    default_data_collator,
    set_seed,
)
from utils_pos import UPOS_LABELS

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    base_dir: str = field(metadata={"help": "Directory containing model checkpoints"})
    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    do_lower_case: bool = field(
        default=False, metadata={"help": "Set to true when using an uncased tokenizer"}
    )


def compute_loss(outputs: torch.Tensor, targets: torch.Tensor):
    return F.cross_entropy(outputs, targets)


def get_grad_z(inputs_z: Dict, model: nn.Module, last_n_encoder_layers: int = 12):
    """Calculates the gradient z. One grad_z should be computed for each
    training sample.
    Arguments:
        inputs_z: torch tensor, training data points
            e.g. an input sequence, dictionary containing input_ids: torch.Tensor, attention_mask: torch.Tensor,
            label: torch.Tensor, and optionally token_type_ids: torch.Tensor
        model: Model used for evaluation
        last_n_encoder_layers: int, returns gradients for the last n encoder layers and the classification head
    Returns:
        grad_z: list of torch tensor, containing the gradients
            from model parameters to loss"""

    model.eval()

    for k, v in inputs_z.items():
        inputs_z[k] = v.to(model.device)

    loss = model(**inputs_z, return_dict=True)["loss"]

    params = [p for p in model.parameters() if p.requires_grad]
    return list(grad(loss, params))


def load_checkpoint(checkpoint_dir: str):
    logger.info(f"Loading checkpoint {checkpoint_dir}")
    model = AutoModelForTokenClassification.from_pretrained(checkpoint_dir).to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    trainer_state = TrainerState.load_from_json(
        os.path.join(checkpoint_dir, "trainer_state.json")
    )
    if "learning_rate" in trainer_state.log_history[-1]:
        learning_rate = trainer_state.log_history[-1]["learning_rate"]
    else:
        learning_rate = trainer_state.log_history[-2]["learning_rate"]
    logger.info(f"Checkpoint learning rate = {learning_rate}")

    return model, learning_rate


def get_dataloader(dataset: Dataset):
    return DataLoader(
        dataset,
        batch_size=1,
        sampler=SequentialSampler(data_source=dataset),
        collate_fn=default_data_collator,
        drop_last=False,
        num_workers=1,
    )


def do_checkpoint(train_dataloader, model, lr, influence_scores, test_dataloader=None):
    # get gradients
    grads_train = []
    grads_test = []
    for z in train_dataloader:
        grad_z = get_grad_z(z, model=model)
        #  for idx, t in enumerate(grad_z):
        #      grad_z[idx] = t.to("cpu")  # .detach()
        grads_train.append(grad_z)

    if not test_dataloader:
        grads_test = copy.deepcopy(grads_train)
    else:
        for z in tqdm(test_dataloader, desc="Test example"):
            grad_z = get_grad_z(z, model=model)
            #  for idx, t in enumerate(grad_z):
            #      grad_z[idx] = t.to("cpu")  # .detach()
            grads_test.append(grad_z)

    # Compute influence of each train datapoint z on each test datapoint z'
    for i, grad_test in enumerate(grads_test):
        for j, grad_train in enumerate(grads_train):
            # compute product of train and test gradient
            grad_dot_product = sum(
                [torch.sum(k * l).data for k, l in zip(grad_train, grad_test)]
            )

            # multiply by learning rate and add to influence score
            influence_scores[i][j] += lr * grad_dot_product


def compute_influence(
    checkpoints: List[Tuple[nn.Module, float]],
    train_dataloader: DataLoader,
    test_dataloader: Optional[DataLoader] = None,
):
    if not test_dataloader:
        influence_scores = np.zeros([len(train_dataloader), len(train_dataloader)])
    else:
        influence_scores = np.zeros([len(test_dataloader), len(train_dataloader)])
    for model, lr in checkpoints:
        do_checkpoint(train_dataloader, model, lr, influence_scores)

    return influence_scores


def log_top_k(
    tokenizer: PreTrainedTokenizer,
    example_dataset: List,
    example_index: int,
    influence_scores: np.array,
    k: int,
):
    input_ids = example_dataset[example_index].input_ids
    text = tokenizer.decode(input_ids, skip_special_tokens=True)
    logger.info(f" Influential examples for:  {text}")

    top_k = np.argsort(influence_scores)[::-1][-k:]
    for i, example in enumerate(top_k):
        input_ids = example_dataset[example].input_ids
        text = tokenizer.decode(input_ids, skip_special_tokens=True)
        logger.info(f"Rank {i + 1} (Influence = {influence_scores[example]}): {text}")


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.setLevel(logging.INFO)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Set seed before initializing model.
    set_seed(42)

    # Prepare for UD pos tagging task
    labels = UPOS_LABELS
    label_map: Dict[int, str] = {i: label for i, label in enumerate(labels)}
    num_labels = len(labels)

    checkpoint_dirs = sorted(
        glob(os.path.join(model_args.base_dir, "checkpoint-*")), reverse=True
    )[:3]

    config = AutoConfig.from_pretrained(
        (
            model_args.config_name
            if model_args.config_name
            else model_args.model_name_or_path
        ),
        num_labels=num_labels,
        id2label=label_map,
        label2id={label: i for i, label in enumerate(labels)},
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    tokenizer_name_or_path = (
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path
    )
    if config.model_type in {"gpt2", "roberta"}:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=True,
            do_lower_case=model_args.do_lower_case,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            add_prefix_space=True,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=True,
            do_lower_case=model_args.do_lower_case,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    checkpoints = [
        load_checkpoint(checkpoint_dir) for checkpoint_dir in checkpoint_dirs
    ]

    train_dataset = get_dataset(
        args=data_args,
        model_type=config.model_type,
        tokenizer=tokenizer,
        labels=labels,
        mode=Split.TRAIN,
    )
    lang_datasets = train_dataset.datasets
    example_datasets = [[] for _ in range(len(lang_datasets[0]))]
    for idx, example in enumerate(zip(*lang_datasets)):
        example_datasets[idx] = list(example)

    example_dataloaders = [get_dataloader(dataset=d) for d in example_datasets]

    influence_scores_full = []

    for dl in tqdm(example_dataloaders, desc="Example"):
        influence_scores = compute_influence(
            checkpoints=checkpoints, train_dataloader=dl
        )
        influence_scores_full.append(influence_scores)

    soft_entropies = []
    for example_idx, influence_scores in enumerate(influence_scores_full):
        for test_i, scores_i in enumerate(influence_scores):

            log_top_k(
                tokenizer=tokenizer,
                example_dataset=example_datasets[example_idx],
                example_index=test_i,
                influence_scores=scores_i,
                k=len(scores_i),
            )

            soft = softmax(scores_i)
            soft_entropies.append(entropy(soft, base=len(scores_i)))

    metrics = {
        "inf_u": np.mean(np.array(soft_entropies)),
    }

    logger.info(metrics)

    with open(
        os.path.join(model_args.base_dir, "tracin_results.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(
            metrics,
            f,
            sort_keys=True,
            separators=(",", ": "),
            ensure_ascii=False,
            indent=4,
        )


if __name__ == "__main__":
    main()
