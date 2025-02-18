import logging
import os
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import wandb
from evaluation.cka import CudaCKA
from IsoScore import IsoScore
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.dataset import ConcatDataset, Dataset
from tqdm import tqdm
from transformers import (
    AutoModel,
    DataCollatorWithPadding,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluationArguments:
    eval_config_path: Optional[str] = field(
        default="", metadata={"help": "Path to eval config .json file"}
    )
    eval_pretrained_model_name_or_path: Optional[str] = field(
        default="xlm-roberta-base",
        metadata={"help": "Pretrained model to compute mean cosine distance against."},
    )
    eval_output_dir: Optional[str] = field(
        default="",
        metadata={"help": "Directory that evaluation results are written to."},
    )
    eval_batch_size: Optional[int] = field(
        default=256, metadata={"help": "Per-device evaluation batch size (default 256)"}
    )

    retrieval_data_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to directory containing data for sentence retrieval."},
    )
    retrieval_max_examples: Optional[int] = field(
        default=None,
        metadata={
            "help": "Maximum number of examples to use per loaded sentence retrieval dataset"
        },
    )
    languages: Optional[str] = field(
        default=None, metadata={"help": "A comma-separated list of languages"}
    )
    layers: str = field(
        default="0,8",
        metadata={"help": "Layer to extract the embeddings from (0 to 12)."},
    )
    block_size: int = field(
        default=-1,
        metadata={
            "help": "Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    compute_iso: bool = field(
        default=False,
        metadata={
            "help": "Whether to compute the IsoScore with the sentence retrieval evaluator"
        },
    )
    compute_cka: bool = field(
        default=False,
        metadata={
            "help": "Whether to compute the centered kernel alignment (CKA) with the evaluator"
        },
    )
    compute_rsa: bool = field(
        default=False,
        metadata={
            "help": "Whether to perform representational similarity analysis (RSA) with the evaluator"
        },
    )

    def __post_init__(self):
        if isinstance(self.languages, str):
            self.languages = self.languages.split(",")
        if isinstance(self.layers, str):
            self.layers = [int(s) for s in self.layers.split(",")]


def remove_padding(tensor: torch.FloatTensor, padding_mask: torch.BoolTensor):
    if tensor is None:
        return []
    result = []
    for elem, mask in zip(tensor, padding_mask):
        if len(elem[mask]) == 0 and len(elem) > 2:
            # add the first token after [CLS] if entry is too short
            result.append(elem[1:2])
        else:
            result.append(elem[mask])
    return result


# Modified from https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/util.py
def get_distances(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    X_norm = torch.nn.functional.normalize(X, p=2, dim=1)
    Y_norm = torch.nn.functional.normalize(Y, p=2, dim=1)
    cos_sims = torch.mm(X_norm, Y_norm.transpose(0, 1))

    # 1.0 - cosine_similarity(X, Y) without copy
    cos_sims *= -1
    cos_sims += 1
    torch.clip(cos_sims, 0, 2, out=cos_sims)
    if X is Y or Y is None:
        # Ensure that distances between vectors and themselves are set to 0.0.
        # This may not be the case due to floating point rounding errors.
        cos_sims.fill_diagonal_(0.0)

    return cos_sims


# torch-based version of https://github.com/pdufter/minimult/blob/master/evaluate.py#L113
def evaluate_retrieval(device, vectors: List[torch.FloatTensor]):
    # for each sentence pair get vectors and alignments (using NNs)
    n = len(vectors)
    if n % 2 != 0:
        raise ValueError("Something's wrong.")
    vectors_e, vectors_f = vectors[: n // 2], vectors[n // 2 :]
    vectors_e = torch.stack([x.to(device).mean(dim=0) for x in vectors_e], dim=0)
    vectors_f = torch.stack([x.to(device).mean(dim=0) for x in vectors_f], dim=0)

    dist = get_distances(vectors_e, vectors_f)
    if dist.shape[0] != dist.shape[1]:
        print("Number of sentences is different?")
    # get different p@k
    nns = torch.argsort(dist, dim=1)[:, :10]
    gt = torch.reshape(torch.arange(dist.shape[0], device=device), (-1, 1))
    p = {}
    for considern in [1, 5, 10]:
        hits1 = torch.sum(torch.sum((nns[:, :considern] == gt), dim=1) > 0)
        p[considern] = hits1 / dist.shape[0]
    nns = torch.argsort(dist, dim=0)[:10, :].T
    gt = torch.arange(dist.shape[0], dtype=torch.int32, device=device).reshape(-1, 1)
    pinv = {}
    for considern in [1, 5, 10]:
        hits1 = torch.sum(torch.sum(nns[:, :considern] == gt, dim=1) > 0)
        pinv[considern] = hits1 / dist.shape[0]

    return p, pinv


def compute_cka(device, vectors: List[torch.FloatTensor]):
    """
    Computes linear centered kernel alignment (CKA) for a pair of contextualized representations
    Reference: https://arxiv.org/pdf/1905.00414.pdf
    """
    n = len(vectors)
    if n % 2 != 0:
        raise ValueError("Something's wrong.")
    vectors_e, vectors_f = vectors[: n // 2], vectors[n // 2 :]
    vectors_e = torch.stack([x.to(device).mean(dim=0) for x in vectors_e], dim=0)
    vectors_f = torch.stack([x.to(device).mean(dim=0) for x in vectors_f], dim=0)

    return CudaCKA(device=device).linear_CKA(vectors_e, vectors_f)


# Calculates the representational geometry of a set of embeddings
def calculate_geometry(device, embeddings: torch.Tensor):
    sim_mat = spearmanr(embeddings.to(device), axis=1)[0]
    dissim_mat = torch.ones(sim_mat.shape).to(device) - sim_mat
    return dissim_mat[torch.triu_indices(embeddings.shape[0], 1)].reshape(-1)


def compute_rsa(device, vectors: List[torch.Tensor]):
    n = len(vectors)
    if n % 2 != 0:
        raise ValueError("Something's wrong.")
    vectors_e, vectors_f = vectors[: n // 2], vectors[n // 2 :]
    vectors_e = torch.stack([x.to(device).mean(dim=0) for x in vectors_e], dim=0)
    vectors_f = torch.stack([x.to(device).mean(dim=0) for x in vectors_f], dim=0)

    geometry_e = calculate_geometry(device, vectors_e)
    geometry_f = calculate_geometry(device, vectors_f)

    return spearmanr(geometry_e, geometry_f, axis=1)[0]


def compute_iso_score(embeddings: List[torch.Tensor], layer: int):
    embeddings = torch.cat(embeddings, dim=0).detach().numpy()
    logger.info(
        f"Calculating IsoScore for layer {layer} embeddings with shape {embeddings.shape}"
    )
    iso_score = IsoScore.IsoScore(np.transpose(embeddings))
    logger.info(f"IsoScore: {iso_score}")
    return iso_score


class LineByLineTextDataset(Dataset):
    """
    Modified from:
    https://github.com/huggingface/transformers/blob/master/src/transformers/data/datasets/language_modeling.py
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        file_path: str,
        block_size: int,
        max_examples: Optional[int] = None,
    ):
        if os.path.isfile(file_path) is False:
            raise ValueError(f"Input file path {file_path} not found")
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info(f"Creating features from dataset file at {file_path}")

        with open(file_path, encoding="utf-8") as f:
            lines = [
                line
                for line in f.read().splitlines()
                if (len(line) > 0 and not line.isspace())
            ]

        if max_examples:
            max_examples = min(max_examples, len(lines))
            lines = lines[:max_examples]

        logger.info(f"{tokenizer.num_special_tokens_to_add()}")
        test = tokenizer.convert_tokens_to_ids(
            tokenizer.tokenize("I like cheese", add_special_tokens=True)
        )
        logger.info(f"{test}")
        batch_encoding = tokenizer(
            lines, add_special_tokens=True, truncation=True, max_length=block_size
        )

        self.examples = batch_encoding["input_ids"]
        self.examples = [
            {"input_ids": torch.tensor(e, dtype=torch.long)} for e in self.examples
        ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]


class SentenceRetrievalEvaluator:
    def __init__(
        self,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        eval_args,
        save_plot: bool = True,
        compute_isoscore: bool = True,
        compute_cka: bool = True,
        compute_rsa: bool = True,
    ):
        self.tokenizer = tokenizer
        self.eval_args = eval_args
        self.save_plot = save_plot
        self.compute_isoscore = compute_isoscore
        self.compute_cka = compute_cka

        self.compute_rsa = compute_rsa

        self._num_langs = len(self.eval_args.languages)
        self.eval_args.languages.sort()
        self.lang_to_idx = {
            lang: idx for idx, lang in enumerate(self.eval_args.languages)
        }

        self.language_pairs = [
            (lang_a, lang_b)
            for lang_a in self.eval_args.languages
            for lang_b in [
                lang_b for lang_b in self.eval_args.languages if lang_a < lang_b
            ]
        ]

        self.dataset_name = os.path.splitext(
            os.path.basename(eval_args.eval_config_path)
        )[0] or os.path.basename(eval_args.retrieval_data_dir)
        self.datasets = [
            self.get_dataset(
                [
                    os.path.join(
                        self.eval_args.retrieval_data_dir,
                        f"{lang_a}-{lang_b}.txt.{lang_a}",
                    ),
                    os.path.join(
                        self.eval_args.retrieval_data_dir,
                        f"{lang_a}-{lang_b}.txt.{lang_b}",
                    ),
                ]
            )
            for (lang_a, lang_b) in self.language_pairs
        ]
        self.data_collator = DataCollatorWithPadding(
            tokenizer=tokenizer, padding="max_length", max_length=eval_args.block_size
        )

        self.results_matrices = self._clear_matrices()
        self.cka_matrices = self._clear_matrices()
        self.rsa_matrices = self._clear_matrices()

    def get_dataset(self, file_paths: List[str]):
        def _dataset(args, file_path, t):
            logger.info("Loading dataset from %s", file_path)
            ds = LineByLineTextDataset(
                tokenizer=t,
                file_path=file_path,
                block_size=args.block_size,
                max_examples=args.retrieval_max_examples,
            )
            logger.info("Size of dataset loaded from %s: %d", file_path, len(ds))
            return ds

        return ConcatDataset(
            [_dataset(self.eval_args, f, self.tokenizer) for f in file_paths]
        )

    def _matrix_as_df(self, matrix: np.array):
        df_keys = self.eval_args.languages + ["avg"]
        return pd.DataFrame(data=matrix, index=df_keys, columns=df_keys)

    def _clear_matrices(self):
        matrices = {
            layer: np.zeros((self._num_langs + 1, self._num_langs + 1))
            for layer in self.eval_args.layers
        }
        for k, v in matrices.items():
            np.fill_diagonal(v, np.nan)
        return matrices

    def store_as_list(self, df: pd.DataFrame, save_path: str):
        np_df = df.to_numpy()[: self._num_langs, : self._num_langs]
        np_df = np_df.flatten()
        df_list = [str(e) for e in np_df if not np.isnan(e)]
        with open(f"{save_path}.txt", "w", encoding="utf-8") as f:
            f.write(",".join(df_list))

    def evaluate(self, model: PreTrainedModel, log_wandb: bool = False):

        num_langs = self._num_langs
        tokenizer = self.tokenizer

        if model.device == "cpu" and torch.cuda.is_available():
            model.to("cuda")
        model.eval()

        logger.info("***** Starting sentence retrieval *****")

        embeds_lists_dict = defaultdict(list)
        for idx, language_pair in enumerate(tqdm(self.language_pairs)):

            lang_a, lang_b = language_pair

            dataset = self.datasets[idx]
            sampler = SequentialSampler(dataset)
            data_loader = DataLoader(
                dataset,
                sampler=sampler,
                batch_size=self.eval_args.eval_batch_size,
                collate_fn=self.data_collator,
            )

            embeds_list_dict = defaultdict(list)
            for inputs in tqdm(
                data_loader,
                desc=f"Embedding sentences for ({lang_a.upper()},{lang_b.upper()})",
            ):
                for k, v in inputs.items():
                    inputs[k] = v.to(model.device)
                with torch.no_grad():
                    output = model.base_model(
                        **inputs, output_hidden_states=True, return_dict=False
                    )[2]
                    padding_mask = (
                        (inputs["input_ids"] == tokenizer.pad_token_id)
                        | (inputs["input_ids"] == tokenizer.cls_token_id)
                        | (inputs["input_ids"] == tokenizer.sep_token_id)
                    )
                    padding_mask = ~padding_mask
                    for layer in self.eval_args.layers:
                        layer_output = output[layer].cpu()
                        layer_output = remove_padding(layer_output, padding_mask)
                        embeds_list_dict[layer].extend(layer_output)

            for layer in self.eval_args.layers:
                embeds_list = embeds_list_dict[layer]

                a_b, b_a = evaluate_retrieval(model.device, embeds_list)
                self.results_matrices[layer][
                    self.lang_to_idx[lang_a], self.lang_to_idx[lang_b]
                ] = (a_b[10].item() * 100)
                self.results_matrices[layer][
                    self.lang_to_idx[lang_b], self.lang_to_idx[lang_a]
                ] = (b_a[10].item() * 100)

                if self.compute_cka:
                    lang_pair_cka = compute_cka(
                        device=model.device, vectors=embeds_list
                    )
                    self.cka_matrices[layer][
                        self.lang_to_idx[lang_a], self.lang_to_idx[lang_b]
                    ] = lang_pair_cka
                    self.cka_matrices[layer][
                        self.lang_to_idx[lang_b], self.lang_to_idx[lang_a]
                    ] = np.nan

                if self.compute_rsa:
                    lang_pair_rsa = compute_rsa(device="cpu", vectors=embeds_list)
                    self.rsa_matrices[layer][
                        self.lang_to_idx[lang_a], self.lang_to_idx[lang_b]
                    ] = lang_pair_rsa
                    self.rsa_matrices[layer][
                        self.lang_to_idx[lang_b], self.lang_to_idx[lang_a]
                    ] = np.nan

                embeds_lists_dict[layer].extend(embeds_list)

        outputs = {}
        for layer in self.eval_args.layers:
            for idx, lang in enumerate(self.eval_args.languages):
                self.results_matrices[layer][num_langs][idx] = np.nanmean(
                    self.results_matrices[layer][:num_langs, idx]
                )
                self.results_matrices[layer][idx][num_langs] = np.nanmean(
                    self.results_matrices[layer][idx, :num_langs]
                )

                if self.compute_cka:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        self.cka_matrices[layer][num_langs][idx] = np.nanmean(
                            self.cka_matrices[layer][:num_langs, idx]
                        )
                        self.cka_matrices[layer][idx][num_langs] = np.nanmean(
                            self.cka_matrices[layer][idx, :num_langs]
                        )

                if self.compute_rsa:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        self.rsa_matrices[layer][num_langs][idx] = np.nanmean(
                            self.rsa_matrices[layer][:num_langs, idx]
                        )
                        self.rsa_matrices[layer][idx][num_langs] = np.nanmean(
                            self.rsa_matrices[layer][idx, :num_langs]
                        )

            self.results_matrices[layer][num_langs][num_langs] = np.nanmean(
                self.results_matrices[layer][:num_langs][:num_langs]
            )
            if self.compute_cka:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    self.cka_matrices[layer][num_langs][num_langs] = np.nanmean(
                        self.cka_matrices[layer][:num_langs][:num_langs]
                    )

            if self.compute_rsa:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    self.rsa_matrices[layer][num_langs][num_langs] = np.nanmean(
                        self.rsa_matrices[layer][:num_langs][:num_langs]
                    )

            std = np.nanstd(self.results_matrices[layer][:num_langs][:num_langs])
            worst_score = np.nanmin(
                self.results_matrices[layer][:num_langs][:num_langs]
            )
            best_score = np.nanmax(self.results_matrices[layer][:num_langs][:num_langs])

            if self.save_plot:
                results_df = self._matrix_as_df(self.results_matrices[layer])
                logger.info(f"\n{results_df}")

                if self.compute_cka:
                    cka_df = self._matrix_as_df(self.cka_matrices[layer])
                    logger.info(f"\n{cka_df}")

                if self.compute_rsa:
                    rsa_df = self._matrix_as_df(self.rsa_matrices[layer])
                    logger.info(f"\n{rsa_df}")

            if self.save_plot:
                c_map = sns.cubehelix_palette(start=0.5, rot=-0.5, as_cmap=True)
                out_path = os.path.join(
                    self.eval_args.eval_output_dir,
                    f"{self.dataset_name}_retrieval_results_layer_{layer}",
                )
                plot = sns.heatmap(
                    results_df,
                    annot=True,
                    fmt="0.2f",
                    xticklabels=True,
                    yticklabels=True,
                    cmap=c_map,
                    vmin=0,
                    vmax=100,
                )
                plot.figure.savefig(f"{out_path}.png")
                self.store_as_list(results_df, out_path)

                if log_wandb:
                    wandb.log({f"layer_{layer}_retrieval_heatmap": wandb.Image(plot)})
                logger.info(
                    f"Saved layer {layer} sentence retrieval results heatmap to %s",
                    out_path,
                )

                plot.figure.clf()

                if self.compute_cka:
                    out_path = os.path.join(
                        self.eval_args.eval_output_dir,
                        f"{self.dataset_name}_cka_layer_{layer}",
                    )
                    plot = sns.heatmap(
                        cka_df,
                        annot=True,
                        fmt="0.2f",
                        xticklabels=True,
                        yticklabels=True,
                        cmap=c_map,
                        vmin=0,
                        vmax=1,
                    )
                    plot.figure.savefig(f"{out_path}.png")
                    self.store_as_list(cka_df, out_path)

                    if log_wandb:
                        wandb.log({f"{layer}_cka_heatmap": wandb.Image(plot)})
                    logger.info("Saved cka heatmap to %s", out_path)

                    plot.figure.clf()

                if self.compute_rsa:
                    out_path = os.path.join(
                        self.eval_args.eval_output_dir,
                        f"{self.dataset_name}_rsa_layer_{layer}",
                    )
                    plot = sns.heatmap(
                        rsa_df,
                        annot=True,
                        fmt="0.2f",
                        xticklabels=True,
                        yticklabels=True,
                        cmap=c_map,
                        vmin=0,
                        vmax=1,
                    )
                    plot.figure.savefig(f"{out_path}.png")
                    self.store_as_list(rsa_df, out_path)

                    if log_wandb:
                        wandb.log({f"layer_{layer}_rsa_heatmap": wandb.Image(plot)})
                    logger.info("Saved rsa heatmap to %s", out_path)

                    plot.figure.clf()

            output = {
                f"layer_{layer}_eval_retrieval_mean_precision": self.results_matrices[
                    layer
                ][num_langs][num_langs],
                f"layer_{layer}_eval_retrieval_std_precision": std,
                f"layer_{layer}_eval_retrieval_min_precision": worst_score,
                f"layer_{layer}_eval_retrieval_max_precision": best_score,
            }

            if self.compute_cka:
                output[f"layer_{layer}_eval_mean_cka"] = self.cka_matrices[layer][
                    num_langs
                ][num_langs]

            if self.compute_rsa:
                output[f"layer_{layer}_eval_mean_rsa"] = self.rsa_matrices[layer][
                    num_langs
                ][num_langs]

            if self.compute_isoscore:
                output[f"layer_{layer}_eval_iso_score"] = compute_iso_score(
                    embeds_lists_dict[layer], layer=layer
                )

            outputs.update(output)

        outputs.update(
            {
                f"mean_eval_retrieval_mean_precision": np.mean(
                    np.array(
                        [
                            outputs[f"layer_{layer}_eval_retrieval_mean_precision"]
                            for layer in self.eval_args.layers
                        ]
                    )
                ),
            }
        )
        if self.compute_isoscore:
            outputs.update(
                {
                    f"mean_eval_iso_score": np.mean(
                        np.array(
                            [
                                outputs[f"layer_{layer}_eval_iso_score"]
                                for layer in self.eval_args.layers
                            ]
                        )
                    ),
                }
            )
        if self.compute_cka:
            outputs.update(
                {
                    f"mean_eval_mean_cka": np.mean(
                        np.array(
                            [
                                outputs[f"layer_{layer}_eval_mean_cka"]
                                for layer in self.eval_args.layers
                            ]
                        )
                    ),
                }
            )
        if self.compute_rsa:
            outputs.update(
                {
                    f"mean_eval_mean_rsa": np.mean(
                        np.array(
                            [
                                outputs[f"layer_{layer}_eval_mean_rsa"]
                                for layer in self.eval_args.layers
                            ]
                        )
                    ),
                }
            )

        self.results_matrices = self._clear_matrices()
        self.cka_matrices = self._clear_matrices()
        self.rsa_matrices = self._clear_matrices()

        return outputs
