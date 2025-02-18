import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from evaluation.evaluation_utils import EvaluationArguments, SentenceRetrievalEvaluator
from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
)
from transformers.trainer_utils import set_seed

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to use for inference.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "The model checkpoint for weights initialization."},
    )
    output_dir: Optional[str] = field(
        default="",
        metadata={"help": "Directory that evaluation results are written to."},
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "If using config from scratch, pass a model type from the list: "
            + ", ".join(MODEL_TYPES)
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
        metadata={"help": "Pretrained tokenizer name or path."},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from s3"
        },
    )
    do_lower_case: bool = field(
        default=False, metadata={"help": "Set to true when using an uncased tokenizer"}
    )
    use_fast: bool = field(
        default=False, metadata={"help": "Whether to use fast tokenizer."}
    )


def main():
    parser = HfArgumentParser((ModelArguments, EvaluationArguments))
    model_args, eval_args = parser.parse_args_into_dataclasses()

    if eval_args.eval_config_path:
        parser = HfArgumentParser(EvaluationArguments)
        (eval_args,) = parser.parse_json_file(json_file=eval_args.eval_config_path)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%d/%m/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.setLevel(logging.INFO)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Set seed
    set_seed(42)

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_args.output_dir == "" and os.path.isdir(model_args.model_name_or_path):
        eval_args.eval_output_dir = model_args.model_name_or_path
    else:
        eval_args.eval_output_dir = model_args.output_dir

    # Make output dir
    os.makedirs(eval_args.eval_output_dir, exist_ok=True)

    if model_args.config_name:
        config = AutoConfig.from_pretrained(
            model_args.config_name, cache_dir=model_args.cache_dir
        )
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path, cache_dir=model_args.cache_dir
        )
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name,
            cache_dir=model_args.cache_dir,
            config=config,
            do_lower_case=model_args.do_lower_case,
            use_fast=model_args.use_fast,
        )
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            config=config,
            do_lower_case=model_args.do_lower_case,
            use_fast=model_args.use_fast,
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch."
            "This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name"
        )

    if model_args.model_name_or_path:
        model = AutoModel.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModel.from_config(config)

    model.to(device)

    retrieval_evaluator = SentenceRetrievalEvaluator(
        tokenizer=tokenizer,
        eval_args=eval_args,
        save_plot=True,
        compute_isoscore=eval_args.compute_iso,
        compute_cka=eval_args.compute_cka,
        compute_rsa=eval_args.compute_rsa,
    )

    eval_metrics = retrieval_evaluator.evaluate(model=model)

    logger.info("***** EVAL METRICS *****")
    logger.info(eval_metrics)

    with open(
        os.path.join(
            eval_args.eval_output_dir,
            f"{retrieval_evaluator.dataset_name}_eval_results.json",
        ),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            eval_metrics,
            f,
            sort_keys=True,
            separators=(",", ": "),
            ensure_ascii=False,
            indent=4,
        )


if __name__ == "__main__":
    main()
