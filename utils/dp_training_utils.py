import collections
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from evaluation.evaluation_utils import SentenceRetrievalEvaluator
from opacus.utils.uniform_sampler import UniformWithReplacementSampler
from private_transformers import PrivacyEngine
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from tqdm import tqdm
from transformers import (
    CONFIG_NAME,
    WEIGHTS_NAME,
    PretrainedConfig,
    PreTrainedTokenizerBase,
    TrainerCallback,
    TrainerState,
    __version__,
    is_apex_available,
)
from transformers.data.data_collator import DataCollator
from transformers.integrations import hp_params
from transformers.modeling_utils import PreTrainedModel
from transformers.models.bert.modeling_bert import BertEmbeddings, BertLMPredictionHead
from transformers.trainer import Trainer
from transformers.trainer_utils import (
    EvalPrediction,
    PredictionOutput,
    TrainOutput,
    get_last_checkpoint,
    set_seed,
    speed_metrics,
)
from transformers.training_args import TrainingArguments
from transformers.utils import logging

if is_apex_available():
    from apex import amp

logger = logging.get_logger(__name__)


@dataclass
class DPTrainingArguments(TrainingArguments):
    differentially_private: bool = field(
        default=False,
        metadata={
            "help": "Whether to train a differentially private model using DP-SGD."
        },
    )
    sigma: Optional[float] = field(
        default=None,
        metadata={
            "help": "Noise multiplier (automatically determined by target eps and num_epochs if None)"
        },
    )
    max_per_sample_grad_norm: float = field(
        default=1.0,
        metadata={"help": "Clip per-sample gradients to this norm (default 1.0)"},
    )
    delta: Optional[float] = field(default=None, metadata={"help": "Target delta"})
    target_epsilon: Optional[float] = field(
        default=None, metadata={"help": "Target epsilon"}
    )
    sample_rate: Optional[float] = field(
        default=None, metadata={"help": "Sample rate used for batch construction"}
    )
    secure_rng: bool = field(
        default=False,
        metadata={
            "help": "Enable Secure RNG to have trustworthy privacy guarantees. Comes at a performance cost"
        },
    )
    clip_per_layer: bool = field(
        default=False,
        metadata={"help": "Clip gradient norms per layer"},
    )
    ghost_clipping: bool = field(
        default=False,
        metadata={"help": "Whether to use ghost clipping technique to save memory"},
    )
    accounting_mode: str = field(
        default="rdp_cks",
        metadata={"help": "One of `rdp`, `gdp`, `rdp_cks`, `glw`, `all`."},
    )
    evaluate_during_training: bool = field(
        default=False,
        metadata={"help": "Run evaluation during training at each logging step."},
    )
    store_best_model: bool = field(
        default=False, metadata={"help": "Whether to store best model during training."}
    )
    metric_score: Optional[str] = field(
        default=None,
        metadata={"help": "Metric used to determine best model during training."},
    )
    do_eval_sentence_retrieval: bool = field(
        default=False,
        metadata={
            "help": "Whether to perform sentence retrieval during evaluation loop"
        },
    )
    new_embeddings: bool = field(
        default=False,
        metadata={"help": "Set True if Embeddings should be reinitialized"},
    )
    freeze_base_model: bool = field(
        default=False,
        metadata={"help": "Set True if base model weights should be frozen"},
    )
    do_language_modeling: bool = field(
        default=False,
        metadata={"help": "Set True if doing MLMing"},
    )
    trainable_encoder_layers: int = field(
        default=1, metadata={"help": "Number of encoder layers that will be trained"}
    )
    sampler_seed: int = field(
        default=42,
        metadata={
            "help": "Random seed for the generator used in UniformWithReplacementSampler."
        },
    )
    use_opacus_sampler: bool = field(
        default=False,
        metadata={
            "help": "Whether to use the UniformWithReplacementSampler provided by Opacus for data loading."
        },
    )


def convert_mlm_head(mlm_head: BertLMPredictionHead, training_complete: bool):
    pass
    """
    if not training_complete:
        mlm_head.predictions.decoder.bias.data = mlm_head.predictions.bias.data.clone()
        del mlm_head.predictions.bias
    else:
        mlm_head.predictions.bias = copy.deepcopy(mlm_head.predictions.decoder.bias)
    """


class DPTrainer(Trainer):
    """
    Modification of HuggingFace trainer for differentially private training
    """

    args: DPTrainingArguments

    def __init__(
        self,
        model: PreTrainedModel,
        args: DPTrainingArguments,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Callable[[], PreTrainedModel] = None,
        retrieval_evaluator: Optional[SentenceRetrievalEvaluator] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        **kwargs,
    ):
        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            optimizers,
            **kwargs,
        )

        if self.args.do_eval_sentence_retrieval and retrieval_evaluator:
            self.retrieval_evaluator = retrieval_evaluator
        else:
            self.retrieval_evaluator = None

        if self.args.do_train:
            self.sample_rate = self.args.sample_rate or (
                self.args.train_batch_size / len(self.train_dataset)
            )  # * self.args.gradient_accumulation_steps
            self.delta = self.args.delta or (1e-4 / len(self.train_dataset))
        else:
            self.sample_rate = 0
            self.delta = 0

        # for finding the best model.
        # TODO: assumes higher is better
        self.best_score = 0.0
        self.eval_steps_without_improvement = 0

    def get_train_dataloader(self, generator) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        if self.args.use_opacus_sampler:
            data_loader = DataLoader(
                self.train_dataset,
                batch_sampler=UniformWithReplacementSampler(
                    num_samples=len(self.train_dataset),
                    sample_rate=self.sample_rate,
                    generator=generator,
                ),
                collate_fn=self.data_collator,
            )
        else:
            data_loader = DataLoader(
                self.train_dataset,
                batch_size=self.args.train_batch_size,
                sampler=RandomSampler(self.train_dataset),
                collate_fn=self.data_collator,
                drop_last=True,
                num_workers=self.args.dataloader_num_workers,
            )

        return data_loader

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        sampler = SequentialSampler(eval_dataset)

        data_loader = DataLoader(
            eval_dataset,
            sampler=sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
        )

        return data_loader

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:

        sampler = SequentialSampler(test_dataset)

        data_loader = DataLoader(
            test_dataset,
            sampler=sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
        )

        return data_loader

    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
    ):
        """
        Main training entry point.
        Args:
            model_path (:obj:`str`, `optional`):
                Local path to the model if the model to train has been instantiated from a local path. If present,
                training will resume from the optimizer/scheduler states loaded here.
            trial (:obj:`optuna.Trial` or :obj:`Dict[str, Any]`, `optional`):
                The trial run or the hyperparameter dictionary for hyperparameter search.
        """

        resume_from_checkpoint = (
            None if not resume_from_checkpoint else resume_from_checkpoint
        )

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        args = self.args

        self.is_in_train = True

        # do_train is not a reliable argument, as it might not be set and .train() still called, so
        # the following is a workaround:
        if args.fp16_full_eval and not args.do_train:
            self._move_model_to_device(self.model, args.device)

        # This might change the seed so needs to run first.
        self._hp_search_setup(trial)

        # Model re-init
        model_reloaded = False
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            set_seed(args.seed)
            self.model = self.call_model_init(trial)
            model_reloaded = True
            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        # Load potential model checkpoint
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(
                    f"No valid checkpoint found in output directory ({args.output_dir})"
                )

        if resume_from_checkpoint is not None:
            if not os.path.isfile(os.path.join(resume_from_checkpoint, WEIGHTS_NAME)):
                raise ValueError(
                    f"Can't find a valid checkpoint at {resume_from_checkpoint}"
                )

            logger.info(f"Loading model from {resume_from_checkpoint}).")

            if os.path.isfile(os.path.join(resume_from_checkpoint, CONFIG_NAME)):
                config = PretrainedConfig.from_json_file(
                    os.path.join(resume_from_checkpoint, CONFIG_NAME)
                )
                checkpoint_version = config.transformers_version
                if checkpoint_version is not None and checkpoint_version != __version__:
                    logger.warning(
                        f"You are resuming training from a checkpoint trained with {checkpoint_version} of "
                        f"Transformers but your current version is {__version__}. This is not recommended and could "
                        "yield to errors or unwanted behaviors."
                    )

            if args.deepspeed:
                # will be resumed in deepspeed_init
                pass
            else:
                # We load the model state dict on the CPU to avoid an OOM error.
                state_dict = torch.load(
                    os.path.join(resume_from_checkpoint, WEIGHTS_NAME),
                    map_location="cpu",
                )
                # If the model is on the GPU, it still works!
                self._load_state_dict_in_model(state_dict)

                # release memory
                del state_dict

        # If model was re-initialized, put it on the right device and update self.model_wrapped
        if model_reloaded:
            if self.place_model_on_device:
                self._move_model_to_device(self.model, args.device)
            self.model_wrapped = self.model

        # Keeping track whether we can can len() on the dataset or not
        train_dataset_is_sized = isinstance(self.train_dataset, collections.abc.Sized)

        if self.args.secure_rng:
            try:
                import torchcsprng as prng
            except ImportError as e:
                msg = (
                    "To use secure RNG, you must install the torchcsprng package! "
                    "Check out the instructions here: https://github.com/pytorch/csprng#installation"
                )
                raise ImportError(msg) from e

            generator = prng.create_random_device_generator("/dev/urandom")

        else:
            generator = torch.Generator()
            generator.manual_seed(self.args.sampler_seed)

        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader(generator=generator)

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        if train_dataset_is_sized:
            num_update_steps_per_epoch = (
                len(train_dataloader) // self.args.gradient_accumulation_steps
            )
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            if self.args.max_steps > 0:
                max_steps = self.args.max_steps
                num_train_epochs = (
                    self.args.max_steps // num_update_steps_per_epoch
                    + int(self.args.max_steps % num_update_steps_per_epoch > 0)
                )
            else:
                max_steps = math.ceil(
                    self.args.num_train_epochs * num_update_steps_per_epoch
                )
                num_train_epochs = math.ceil(self.args.num_train_epochs)
        else:
            # see __init__. max_steps is set when the dataset has no __len__
            max_steps = self.args.max_steps
            num_train_epochs = 1
            num_update_steps_per_epoch = max_steps

        self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        model = self._wrap_model(self.model_wrapped)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        model.train()

        total_train_batch_size = (
            self.args.train_batch_size * self.args.gradient_accumulation_steps
        )

        num_examples = (
            self.num_examples(train_dataloader)
            if train_dataset_is_sized
            else total_train_batch_size * self.args.max_steps
        )

        if self.args.differentially_private:
            num_layers = len(
                list((n, p) for n, p in model.named_parameters() if p.requires_grad)
            )

            logger.info("Using differential privacy")
            logger.info(f"  Clip per layer = {self.args.clip_per_layer}")
            logger.info(f"  Secure RNG = {self.args.secure_rng}")
            logger.info(f"  Noise multiplier = {self.args.sigma}")
            logger.info(f"  Max grad norm = {self.args.max_per_sample_grad_norm}")
            logger.info(f"  Delta = {self.delta}")
            logger.info(f"  Num layers = {num_layers}")
            if self.args.clip_per_layer:
                max_grad_norm = num_layers * [self.args.max_per_sample_grad_norm]
            else:
                max_grad_norm = self.args.max_per_sample_grad_norm

            """
            privacy_engine = PrivacyEngine(
                model,
                sample_rate=self.sample_rate * self.args.gradient_accumulation_steps,
                sample_size=len(self.train_dataset),
                alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
                noise_multiplier=self.args.sigma,
                max_grad_norm=max_grad_norm,
                secure_rng=self.args.secure_rng,
                target_delta=self.delta,
                poisson=True,
            )
            """
            privacy_engine = PrivacyEngine(
                module=model,
                batch_size=self.args.per_device_train_batch_size
                * self.args.gradient_accumulation_steps,
                sample_size=len(self.train_dataset),
                epochs=num_train_epochs,
                max_grad_norm=max_grad_norm,
                noise_multiplier=self.args.sigma,
                target_epsilon=self.args.target_epsilon,
                target_delta=self.delta,
                # sample_rate=self.sample_rate * self.args.gradient_accumulation_steps,
                # alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
                accounting_mode=self.args.accounting_mode,
                ghost_clipping=self.args.ghost_clipping,  # The only change you need to make!
            )

            privacy_engine.attach(self.optimizer)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", num_examples)
        logger.info("  Num Epochs = %d", num_train_epochs)
        if self.args.differentially_private:
            logger.info("  Sample rate = %0.8f", self.sample_rate)
        logger.info(
            "  Instantaneous batch size per device = %d",
            self.args.per_device_train_batch_size,
        )
        logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d",
            total_train_batch_size,
        )
        logger.info(
            "  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps
        )
        logger.info("  Total optimization steps = %d", max_steps)

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, "trainer_state.json")
        ):
            self.state = TrainerState.load_from_json(
                os.path.join(resume_from_checkpoint, "trainer_state.json")
            )
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = (
                    self.state.global_step % num_update_steps_per_epoch
                )
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info(
                "  Continuing training from checkpoint, will skip to saved global_step"
            )
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(
                f"  Continuing training from global step {self.state.global_step}"
            )
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "
                    "flag to your launch command, but you will resume the training on data already seen by your model."
                )
                if self.is_local_process_zero() and not args.disable_tqdm:
                    steps_trained_progress_bar = tqdm(
                        total=steps_trained_in_current_epoch
                    )
                    steps_trained_progress_bar.set_description(
                        "Skipping the first batches"
                    )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        self.state.trial_name = (
            self.hp_name(trial) if self.hp_name is not None else None
        )
        self.state.trial_params = hp_params(trial) if trial is not None else None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        self._total_flos = self.state.total_flos

        self.control = self.callback_handler.on_train_begin(
            self.args, self.state, self.control
        )

        for epoch in range(epochs_trained, num_train_epochs):
            model.zero_grad(set_to_none=True)

            epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if train_dataset_is_sized
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(
                args, self.state, self.control
            )

            for step, inputs in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(
                        args, self.state, self.control
                    )

                losses = self.training_step(model, inputs)
                tr_loss += losses["scalar_loss"]

                self._total_flos += self.floating_point_ops(inputs)

                if (step + 1) % self.args.gradient_accumulation_steps == 0:

                    if not self.args.differentially_private:
                        torch.nn.utils.clip_grad_norm_(
                            (
                                amp.master_params(self.optimizer)
                                if self.use_apex
                                else model.parameters()
                            ),
                            self.args.max_grad_norm,
                        )
                        self.optimizer.step()
                    else:
                        self.optimizer.step(loss=losses.get("vector_loss"))

                    self.lr_scheduler.step()
                    model.zero_grad(set_to_none=True)

                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(
                        self.args, self.state, self.control
                    )

                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch)
                else:
                    if self.args.differentially_private:
                        self.optimizer.virtual_step(loss=losses.get("vector_loss"))
                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

            self.control = self.callback_handler.on_epoch_end(
                self.args, self.state, self.control
            )
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch)

            if self.control.should_training_stop:
                break

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info(
            "\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n"
        )
        if (
            self.args.load_best_model_at_end
            and self.state.best_model_checkpoint is not None
        ):
            logger.info(
                f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric})."
            )
            if isinstance(model, PreTrainedModel):
                self.model = model.from_pretrained(self.state.best_model_checkpoint)
                self.model = self.model.to(self.args.device)
            else:
                state_dict = torch.load(
                    os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME)
                )
                self.model.load_state_dict(state_dict)

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_examples,
            num_steps=self.state.max_steps,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        self.control = self.callback_handler.on_train_end(
            args, self.state, self.control
        )

        return TrainOutput(self.state.global_step, train_loss, metrics)

    def compute_loss(
        self, model, inputs, return_outputs=False, return_vector_loss=False
    ):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs["logits"]
        if len(logits.shape) == 3:
            logits = logits.permute(0, 2, 1)  # Unpack.
            seq_lens = (labels != -100).sum(dim=1)
            loss = F.cross_entropy(logits, labels, reduction="none")
            loss = (
                loss.sum(dim=1) / seq_lens
            )  # This can cause NaNs in case a sequence contains only -100 labels
        else:
            loss = F.cross_entropy(logits, labels, reduction="none")
        if not return_vector_loss:
            loss = loss.mean(dim=0)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        return (loss, (loss, outputs["logits"])) if return_outputs else loss

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> dict:
        model.train()
        inputs = self._prepare_inputs(inputs)
        loss = self.compute_loss(
            model, inputs, return_vector_loss=True
        )  # (batch_size,).

        vector_loss = loss
        scalar_loss = loss.mean(dim=0) / self.args.gradient_accumulation_steps

        if not self.args.differentially_private:
            if self.use_apex:
                with amp.scale_loss(scalar_loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                scalar_loss.backward()

        scalar_loss = scalar_loss.detach()
        return dict(vector_loss=vector_loss, scalar_loss=scalar_loss)

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch):
        if self.control.should_log:
            logs: Dict[str, float] = {}
            tr_loss_scalar = tr_loss.item()
            tr_loss -= tr_loss

            logs["loss"] = round(
                tr_loss_scalar
                / (self.state.global_step - self._globalstep_last_logged),
                4,
            )
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            if self.args.differentially_private:
                logs.update(self.optimizer.privacy_engine.get_training_stats())
                logs.update(
                    self.optimizer.privacy_engine.get_privacy_spent(
                        accounting_mode="all", lenient=False
                    )
                )

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate()
            self._report_to_hp_search(trial, epoch, metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(
                self.args, self.state, self.control
            )

    def save_model(self, output_dir: Optional[str] = None):
        """
        Saving best-practices: if you use default names for the model,
        you can reload it using from_pretrained().

        """
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, PreTrainedModel):
            raise ValueError("Trainer.model appears to not be a PreTrainedModel")

        """
        model_copy = copy.deepcopy(self.model)
        if self.args.differentially_private and self.args.do_language_modeling:
            convert_mlm_head(model_copy.cls, training_complete=True)
        model_copy.save_pretrained(output_dir)
        """
        self.model.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        do_retrieval: bool = True,
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.
        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init :obj:`compute_metrics` argument).
        You can also subclass and override this method to inject custom behavior.
        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`,
                columns not accepted by the ``model.forward()`` method are automatically removed. It must implement the
                :obj:`__len__` method.
            ignore_keys (:obj:`Lst[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (:obj:`str`, `optional`, defaults to :obj:`"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)
            do_retrieval: (:obj:`bool`, `optional`, defaults to :obj:`"True"`):
                Pass true if you wish to use the sentence retrieval evaluator
        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        output = self.evaluation_loop(
            eval_dataloader,
            description="Evaluation",
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        if self.args.store_best_model:
            self.store_best_model(output)

        if self.retrieval_evaluator and do_retrieval:
            output.metrics.update(
                self.retrieval_evaluator.evaluate(model=self.model, log_wandb=True)
            )
            # When evaluating sentence retrieval (during MLM), we're also interested in the perplexity
            if self.args.do_language_modeling:
                output.metrics["eval_perplexity"] = math.exp(
                    output.metrics["eval_loss"]
                )

        self.log(output.metrics)

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, output.metrics
        )
        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics

    def predict(
        self,
        test_dataset: Dataset,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "test",
        do_retrieval: bool = True,
    ) -> PredictionOutput:
        """
        Run prediction and returns predictions and potential metrics.
        Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method
        will also return metrics, like in :obj:`evaluate()`.
        Args:
            test_dataset (:obj:`Dataset`):
                Dataset to run the predictions on. If it is an :obj:`datasets.Dataset`, columns not accepted by the
                ``model.forward()`` method are automatically removed. Has to implement the method :obj:`__len__`
            ignore_keys (:obj:`Lst[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (:obj:`str`, `optional`, defaults to :obj:`"test"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "test_bleu" if the prefix is "test" (default)
            do_retrieval: (:obj:`bool`, `optional`, defaults to :obj:`"True"`):
                Pass true if you wish to use the sentence retrieval evaluator
        .. note::
            If your predictions or labels have different sequence length (for instance because you're doing dynamic
            padding in a token classification task) the predictions will be padded (on the right) to allow for
            concatenation into one array. The padding index is -100.
        Returns: `NamedTuple` A namedtuple with the following keys:
            - predictions (:obj:`np.ndarray`): The predictions on :obj:`test_dataset`.
            - label_ids (:obj:`np.ndarray`, `optional`): The labels (if the dataset contained some).
            - metrics (:obj:`Dict[str, float]`, `optional`): The potential dictionary of metrics (if the dataset
              contained labels).
        """

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()
        eval_dataloader = self.get_test_dataloader(test_dataset)
        start_time = time.time()

        output = self.evaluation_loop(
            eval_dataloader,
            description="Prediction",
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        if self.args.store_best_model:
            self.store_best_model(output)

        if self.retrieval_evaluator and do_retrieval:
            output.metrics.update(
                self.retrieval_evaluator.evaluate(model=self.model, log_wandb=True)
            )
            # When evaluating sentence retrieval (during MLM), we're also interested in the perplexity
            if self.args.do_language_modeling:
                output.metrics["test_perplexity"] = math.exp(
                    output.metrics["test_loss"]
                )

        self.log(output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return PredictionOutput(
            predictions=output.predictions,
            label_ids=output.label_ids,
            metrics=output.metrics,
        )

    def store_best_model(self, output):

        if self.args.metric_score not in output.metrics:
            raise Exception(
                "Metric %s not in output.\nThe following output was generated: %s",
                str(self.args.metric_score),
                str(output),
            )

        if output.metrics[self.args.metric_score] > self.best_score:

            self.best_score = output.metrics[self.args.metric_score]
            self.eval_steps_without_improvement = 0
            # Save model checkpoint
            self.save_model(os.path.join(self.args.output_dir, "best_model"))
            with open(
                os.path.join(self.args.output_dir, "best_model", "output.txt"), "w"
            ) as f:
                f.write(str(output.metrics))
        else:
            self.eval_steps_without_improvement += 1
            logger.info(
                "Number of eval steps without improvement: %d",
                self.eval_steps_without_improvement,
            )
            if self.eval_steps_without_improvement > 4:
                logger.info(
                    "%d eval steps have passed without improvement. Early stopping training",
                    self.eval_steps_without_improvement,
                )


def map_a_special_embeddings_into_b(
    a_embeddings, a_tokenizer, b_embeddings, b_tokenizer
):
    a_vocab = a_tokenizer.get_vocab()
    counter = 0
    for token, b_position in zip(
        b_tokenizer.all_special_tokens, b_tokenizer.all_special_ids
    ):
        if token in a_vocab:
            counter += 1
            a_position = a_vocab[token]
            b_embeddings.word_embeddings.weight.data[b_position] = (
                a_embeddings.word_embeddings.weight[a_position].data.clone()
            )
    logger.info(
        f"We have found {counter} original tokens and replaced their representations."
    )
    return b_embeddings


def overwrite_embeddings(model: PreTrainedModel, a_tokenizer=None, b_tokenizer=None):
    new_embeddings = BertEmbeddings(model.config)
    new_embeddings.apply(model._init_weights)
    if a_tokenizer is not None and b_tokenizer is not None:
        a_embeddings = model.embeddings
        new_embeddings = map_a_special_embeddings_into_b(
            a_embeddings=a_embeddings,
            a_tokenizer=a_tokenizer,
            b_embeddings=new_embeddings,
            b_tokenizer=b_tokenizer,
        )

    if a_tokenizer is not None and b_tokenizer is not None:
        new_embeddings.position_embeddings = model.embeddings.position_embeddings
        new_embeddings.token_type_embeddings = model.embeddings.token_type_embeddings
        new_embeddings.LayerNorm = model.embeddings.LayerNorm
        new_embeddings = new_embeddings.to(model.device)

    model.embeddings = new_embeddings
    model.vocab_size = b_tokenizer.vocab_size


class QuestionAnsweringTrainer(DPTrainer):
    def __init__(self, *args, eval_examples=None, post_process_function=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function

    def evaluate(
        self,
        eval_dataset=None,
        eval_examples=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ):
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            output = self.evaluation_loop(eval_dataloader, description="Evaluation")
        finally:
            self.compute_metrics = compute_metrics

        if self.post_process_function is not None and self.compute_metrics is not None:
            eval_preds = self.post_process_function(
                eval_examples, eval_dataset, output.predictions
            )
            metrics = self.compute_metrics(eval_preds)

            if self.retrieval_evaluator:
                metrics.update(
                    self.retrieval_evaluator.evaluate(args=self.args, model=self.model)
                )

            # Prefix all keys with metric_key_prefix + '_'
            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

            self.log(metrics)
        else:
            metrics = {}

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )
        return metrics

    def predict(
        self,
        predict_dataset,
        predict_examples,
        ignore_keys=None,
        metric_key_prefix: str = "test",
    ):
        predict_dataloader = self.get_test_dataloader(predict_dataset)

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            output = self.evaluation_loop(predict_dataloader, description="Prediction")
        finally:
            self.compute_metrics = compute_metrics

        if self.post_process_function is None or self.compute_metrics is None:
            return output

        predictions = self.post_process_function(
            predict_examples, predict_dataset, output.predictions, "predict"
        )
        metrics = self.compute_metrics(predictions)

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return PredictionOutput(
            predictions=predictions.predictions,
            label_ids=predictions.label_ids,
            metrics=metrics,
        )
