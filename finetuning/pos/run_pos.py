#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for token classification.
"""
# You can also adapt this script on your own token classification task and datasets. Pointers for this are left as
# comments.

import logging
import os
import statistics
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import transformers
import wandb
from evaluation.evaluation_utils import EvaluationArguments, SentenceRetrievalEvaluator
from pos_dataset import POSDataset, Split
from seqeval.metrics import accuracy_score
from torch import nn
from torch.utils.data import ConcatDataset
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version
from utils.dp_training_utils import DPTrainer, DPTrainingArguments
from utils_pos import LANG_2_PUD, LANG_2_TREEBANK, UPOS_LABELS

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.9.0")

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

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
    dropout_prob: Optional[float] = field(
        default=None, metadata={"help": "Dropout probability"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_languages: str = field(
        metadata={"help": "Comma-separated list of languages for training"}
    )
    eval_languages: str = field(
        metadata={"help": "Comma-separated list of languages for validation"}
    )
    data_dir: str = field(metadata={"help": "Path to train, dev, and test data files."})
    use_pud: bool = field(
        default=False,
        metadata={"help": "Whether to use parallel PUD treebanks for training."},
    )
    per_lang_max_examples: Optional[int] = field(
        default=10000,
        metadata={
            "help": "Maximum number of examples to use per loaded POS tagging dataset"
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    max_seq_length: int = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. If set, sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )

    def __post_init__(self):
        if isinstance(self.train_languages, str):
            self.train_languages = self.train_languages.split(",")
        if isinstance(self.eval_languages, str):
            self.eval_languages = self.eval_languages.split(",")


def get_dataset(
    args: DataTrainingArguments,
    model_type: str,
    tokenizer: PreTrainedTokenizerFast,
    labels: List[str],
    mode: Split,
):
    def _dataset(file_path, m):
        ds = POSDataset(
            data_dir=file_path,
            tokenizer=tokenizer,
            labels=labels,
            model_type=model_type,
            max_seq_length=args.max_seq_length,
            overwrite_cache=args.overwrite_cache,
            mode=m,
        )
        if args.per_lang_max_examples:
            ds = ds[: args.per_lang_max_examples]
        logger.info("Size of dataset loaded from %s: %d", file_path, len(ds))
        return ds

    ds_list = []

    if mode == Split.TRAIN:
        languages = args.train_languages
    else:
        languages = args.eval_languages

    for lang in languages:
        if args.use_pud and (mode == Split.TRAIN or mode == Split.DEV):
            fp = os.path.join(args.data_dir, LANG_2_PUD[lang])
            ds_list.append(_dataset(fp, Split.TEST))
        else:
            fp = os.path.join(args.data_dir, LANG_2_TREEBANK[lang])
            ds_list.append(_dataset(fp, mode))

    if mode == Split.TRAIN or mode == Split.DEV:
        return ConcatDataset(ds_list)
    else:
        return ds_list


def setup_wandb(training_args: DPTrainingArguments):
    wandb.init(project="xpos")
    wandb.run.name = training_args.run_name
    wandb.run.save()


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (
            ModelArguments,
            DataTrainingArguments,
            DPTrainingArguments,
            EvaluationArguments,
        )
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, eval_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args, eval_args = (
            parser.parse_args_into_dataclasses()
        )

    if eval_args.eval_config_path:
        parser = HfArgumentParser(EvaluationArguments)
        (eval_args,) = parser.parse_json_file(json_file=eval_args.eval_config_path)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    if is_main_process(training_args.local_rank):
        logger.setLevel(logging.INFO)
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
        transformers.utils.logging.set_verbosity_info()

    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Prepare for UD pos tagging task
    labels = UPOS_LABELS
    label_map: Dict[int, str] = {i: label for i, label in enumerate(labels)}
    num_labels = len(labels)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
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
        hidden_dropout_prob=model_args.dropout_prob if model_args.dropout_prob else 0.1,
        attention_probs_dropout_prob=(
            model_args.dropout_prob if model_args.dropout_prob else 0.1
        ),
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

    model = AutoModelForTokenClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # model.init_weights()

    if model_args.dropout_prob:
        logger.info(f"Using dropout probability {model.config.hidden_dropout_prob}.")

    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
            "at https://huggingface.co/transformers/index.html#supported-frameworks to find the model types that meet this "
            "requirement"
        )

    train_dataset = (
        get_dataset(
            args=data_args,
            model_type=config.model_type,
            tokenizer=tokenizer,
            labels=labels,
            mode=Split.TRAIN,
        )
        if training_args.do_train
        else None
    )

    eval_dataset = (
        get_dataset(
            args=data_args,
            model_type=config.model_type,
            tokenizer=tokenizer,
            labels=labels,
            mode=Split.DEV,
        )
        if training_args.do_eval
        else None
    )

    def align_predictions(
        predictions: np.ndarray, label_ids: np.ndarray
    ) -> Tuple[List[list], List[list]]:
        preds = np.argmax(predictions, axis=2)

        batch_size, seq_len = preds.shape

        out_label_list = [[] for _ in range(batch_size)]
        preds_list = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            for j in range(seq_len):
                if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                    out_label_list[i].append(label_map[label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])

        return preds_list, out_label_list

    def compute_metrics(p: EvalPrediction) -> Dict:
        preds_list, out_label_list = align_predictions(p.predictions, p.label_ids)
        return {"acc": accuracy_score(out_label_list, preds_list)}

    if training_args.freeze_base_model:
        logger.info("Freezing base model parameters")

        trainable_layers = [
            model.base_model.encoder.layer[-training_args.trainable_encoder_layers :],
            model.classifier,
        ]
    else:
        trainable_layers = [model]

    total_params = 0
    trainable_params = 0

    for p in model.parameters():
        p.requires_grad = False
        total_params += p.numel()

    for layer in trainable_layers:
        for p in layer.parameters():
            p.requires_grad = True
            trainable_params += p.numel()

    logger.info(
        f"Total parameters count: {total_params}"
    )  # ~177M for mBERT, ~277M for XLM-R
    logger.info(f"Trainable parameters count: {trainable_params}")  # ~7M per layer

    if training_args.do_eval_sentence_retrieval and eval_args.retrieval_data_dir:
        eval_args.eval_output_dir = training_args.output_dir
        eval_args.eval_batch_size = training_args.per_device_eval_batch_size

        retrieval_evaluator = SentenceRetrievalEvaluator(
            tokenizer=tokenizer,
            eval_args=eval_args,
            save_plot=True,
            compute_isoscore=eval_args.compute_iso,
            compute_cka=eval_args.compute_cka,
            compute_rsa=eval_args.compute_rsa,
        )
    else:
        retrieval_evaluator = None

    # Initialize our Trainer
    trainer = DPTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        retrieval_evaluator=retrieval_evaluator,
    )

    # Training
    if training_args.do_train:
        # Evaluate pretrained model before training
        if retrieval_evaluator:
            setup_wandb(training_args)
            """
            eval_metrics = retrieval_evaluator.evaluate(model=model)
            eval_metrics["global_step"] = 0
            eval_metrics["epoch"] = 0
            trainer.log(eval_metrics)
            """

        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()  # Saves the tokenizer too for easy upload

        max_train_samples = (
            data_args.per_lang_max_examples * len(data_args.train_languages)
            if data_args.per_lang_max_examples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    elif training_args.do_eval_sentence_retrieval:
        setup_wandb(training_args)

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        eval_metrics = {}
        for lang_dataset, lang in zip(eval_dataset.datasets, data_args.eval_languages):
            do_retrieval = (
                False
                if training_args.do_train
                else training_args.do_eval_sentence_retrieval
            )
            metrics = trainer.evaluate(
                lang_dataset,
                do_retrieval=do_retrieval,
                metric_key_prefix=f"eval_{lang}",
            )

            max_eval_samples = (
                data_args.per_lang_max_examples
                if data_args.per_lang_max_examples is not None
                else len(lang_dataset)
            )
            metrics[f"eval_{lang}_samples"] = min(max_eval_samples, len(lang_dataset))
            eval_metrics.update(metrics)

        eval_metrics["eval_acc"] = np.mean(
            np.array([v for k, v in eval_metrics.items() if "acc" in k])
        )

        eval_metrics["eval_acc_computed"] = trainer.evaluate(do_retrieval=False)[
            "eval_acc"
        ]

        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

        if training_args.do_train and training_args.differentially_private:
            privacy_spent = trainer.optimizer.privacy_engine.get_privacy_spent(
                accounting_mode="all", lenient=False
            )
            logger.info(f"Privacy spent: {privacy_spent}")

    # Predict
    if training_args.do_predict:
        logger.info("*** Predict ***")

        test_datasets = get_dataset(
            args=data_args,
            model_type=config.model_type,
            tokenizer=tokenizer,
            labels=labels,
            mode=Split.TEST,
        )

        all_langs_acc = {}
        for language, test_dataset in zip(data_args.eval_languages, test_datasets):
            predictions, label_ids, metrics = trainer.predict(
                test_dataset,
                metric_key_prefix=f"predict_{language}",
                do_retrieval=False,
            )
            all_langs_acc[f"{language}_acc"] = metrics[f"predict_{language}_acc"]
            preds_list, _ = align_predictions(predictions, label_ids)

            trainer.log_metrics(
                "predict",
                {f"predict_{language}_acc": metrics[f"predict_{language}_acc"]},
            )

        output_predict_results_file = os.path.join(
            training_args.output_dir, f"predict_results.json"
        )
        with open(output_predict_results_file, "w", encoding="utf-8") as writer:
            writer.write("{\n")
            acc_list = []
            for k, v in all_langs_acc.items():
                writer.write(f'\t"{k}": "{v}",\n')
                acc_list.append(v)
            predict_mean_acc = statistics.mean(acc_list)
            writer.write(f'\t"mean_acc": "{predict_mean_acc}"\n')
            writer.write("}\n")

        trainer.log_metrics("predict", {f"predict_mean_acc": predict_mean_acc})


if __name__ == "__main__":
    main()
