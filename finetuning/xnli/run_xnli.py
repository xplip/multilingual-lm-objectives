#!/usr/bin/env python
# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" Finetuning multi-lingual models on XNLI (e.g. Bert, DistilBERT, XLM).
    Adapted from `examples/text-classification/run_glue.py`"""

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import wandb
from datasets import concatenate_datasets, load_dataset, load_metric
from evaluation.evaluation_utils import EvaluationArguments, SentenceRetrievalEvaluator
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
from utils.dp_training_utils import DPTrainer, DPTrainingArguments

check_min_version("4.9.2")

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/text-classification/requirements.txt",
)

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default=None,
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
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
    do_lower_case: Optional[bool] = field(
        default=False,
        metadata={
            "help": "arg to indicate if tokenizer should do lower case in AutoTokenizer.from_pretrained()"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
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
    dropout_prob: Optional[float] = field(
        default=0.1, metadata={"help": "Dropout probability"}
    )


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    train_languages: str = field(
        default=None, metadata={"help": "Train languages, comma-separated."}
    )
    eval_languages: str = field(
        default=None, metadata={"help": "Train languages, comma-separated."}
    )
    max_seq_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets or not."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )

    def __post_init__(self):
        if isinstance(self.train_languages, str):
            self.train_languages = self.train_languages.split(",")
        if isinstance(self.eval_languages, str):
            self.eval_languages = self.eval_languages.split(",")


def setup_wandb(training_args: DPTrainingArguments):
    wandb.init(project="xnli")
    wandb.run.name = training_args.run_name
    wandb.run.save()


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataArguments, DPTrainingArguments, EvaluationArguments)
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

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

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
        # transformers.utils.logging.enable_default_handler()
        # transformers.utils.logging.enable_explicit_format()
        # transformers.utils.logging.set_verbosity_info()

    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
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

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    # Downloading and loading xnli dataset from the hub.
    if training_args.do_train:
        train_dataset = concatenate_datasets(
            [
                load_dataset(
                    "xnli", language, split="validation", cache_dir=model_args.cache_dir
                )
                for language in data_args.train_languages
            ]
        )
        label_list_train = train_dataset.features["label"].names
    else:
        label_list_train = []

    if training_args.do_eval:
        eval_datasets = [
            load_dataset(
                "xnli", language, split="validation", cache_dir=model_args.cache_dir
            )
            for language in data_args.eval_languages
        ]
        label_list_dev = eval_datasets[0].features["label"].names
    else:
        label_list_dev = []

    if training_args.do_predict:
        predict_datasets = [
            load_dataset("xnli", language, split="test", cache_dir=model_args.cache_dir)
            for language in data_args.eval_languages
        ]
        label_list_test = predict_datasets[0].features["label"].names
    else:
        label_list_test = []

    # Labels
    num_labels = len(set(label_list_train + label_list_dev + label_list_test))

    # Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        (
            model_args.config_name
            if model_args.config_name
            else model_args.model_name_or_path
        ),
        num_labels=num_labels,
        finetuning_task="xnli",
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        hidden_dropout_prob=model_args.dropout_prob,
        attention_probs_dropout_prob=model_args.dropout_prob,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        (
            model_args.tokenizer_name
            if model_args.tokenizer_name
            else model_args.model_name_or_path
        ),
        do_lower_case=model_args.do_lower_case,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Preprocessing the datasets
    def preprocess_function(examples):
        # Tokenize the texts
        if "topic" in examples:
            del examples["topic"]

        return tokenizer(
            examples["premise"],
            examples["hypothesis"],
            padding="max_length",
            max_length=data_args.max_seq_length,
            truncation=True,
        )

    if training_args.do_train:
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
        # Log a few random samples from the training set:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    if training_args.do_eval:
        if data_args.max_eval_samples is not None:
            eval_datasets = [
                eval_dataset.select(range(data_args.max_eval_samples))
                for eval_dataset in eval_datasets
            ]
        eval_datasets = [
            eval_dataset.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )
            for eval_dataset in eval_datasets
        ]

    if training_args.do_predict:
        if data_args.max_predict_samples is not None:
            predict_datasets = [
                predict_dataset.select(range(data_args.max_predict_samples))
                for predict_dataset in predict_datasets
            ]
        predict_datasets = [
            predict_dataset.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )
            for predict_dataset in predict_datasets
        ]

    # Get the metric function
    metric = load_metric("xnli")

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        return metric.compute(predictions=preds, references=p.label_ids)

    if training_args.freeze_base_model:
        logger.info("Freezing base model parameters")

        trainable_layers = [
            model.base_model.encoder.layer[-training_args.trainable_encoder_layers :],
            model.classifier,
        ]
        total_params = 0
        trainable_params = 0

        for p in model.parameters():
            p.requires_grad = False
            total_params += p.numel()

        for layer in trainable_layers:
            for p in layer.parameters():
                p.requires_grad = True
                trainable_params += p.numel()

        logger.info(f"Total parameters count: {total_params}")  # ~177M for mBERT
        logger.info(f"Trainable parameters count: {trainable_params}")  # ~7M per layer

    if (
        training_args.do_eval_sentence_retrieval
        and eval_args.retrieval_data_dir
        and eval_args.languages
    ):
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
        eval_dataset=(
            concatenate_datasets(eval_datasets) if training_args.do_eval else None
        ),
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        retrieval_evaluator=retrieval_evaluator,
    )

    # Training
    if training_args.do_train:
        if training_args.do_train:
            # Evaluate pretrained model before training
            if retrieval_evaluator:
                setup_wandb(training_args)
                eval_metrics = retrieval_evaluator.evaluate(model=model, log_wandb=True)
                eval_metrics["global_step"] = 0
                eval_metrics["epoch"] = 0
                trainer.log(eval_metrics)

        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    elif training_args.do_eval_sentence_retrieval:
        setup_wandb(training_args)

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        eval_metrics = {}
        for eval_dataset, language in zip(eval_datasets, data_args.eval_languages):
            metrics = trainer.evaluate(
                eval_dataset=eval_dataset, metric_key_prefix=f"eval_{language}"
            )

            max_eval_samples = (
                data_args.max_eval_samples
                if data_args.max_eval_samples is not None
                else len(eval_dataset)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

            eval_metrics.update(metrics)

        eval_metrics["eval_accuracy"] = np.mean(
            np.array([v for k, v in eval_metrics.items() if "accuracy" in k])
        )

        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

        if training_args.do_train and training_args.differentially_private:
            privacy_spent = trainer.optimizer.privacy_engine.get_privacy_spent(
                accounting_mode="all", lenient=False
            )
            logger.info(f"Privacy spent: {privacy_spent}")

    # Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")
        predict_metrics = {}
        for predict_dataset, language in zip(
            predict_datasets, data_args.eval_languages
        ):
            predictions, labels, metrics = trainer.predict(
                predict_dataset, metric_key_prefix=f"predict_{language}"
            )

            max_predict_samples = (
                data_args.max_predict_samples
                if data_args.max_predict_samples is not None
                else len(predict_dataset)
            )
            metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

            predictions = np.argmax(predictions, axis=1)
            output_predict_file = os.path.join(
                training_args.output_dir, f"predictions_{language}.txt"
            )
            if trainer.is_world_process_zero():
                with open(output_predict_file, "w") as writer:
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        item = label_list_test[item]
                        writer.write(f"{index}\t{item}\n")

            predict_metrics.update(metrics)

        predict_metrics["predict_accuracy"] = np.mean(
            np.array([v for k, v in predict_metrics.items() if "accuracy" in k])
        )

        trainer.log_metrics("predict", predict_metrics)
        trainer.save_metrics("predict", predict_metrics)


if __name__ == "__main__":
    main()
