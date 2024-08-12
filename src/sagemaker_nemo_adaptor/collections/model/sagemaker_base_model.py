from contextlib import nullcontext
from typing import Any, Dict, Optional, Union

import torch
import torch.distributed as dist
import transformer_engine
import transformers
from accelerate import init_empty_weights
from nemo.collections.nlp.models.nlp_model import NLPModel
from omegaconf import open_dict
from omegaconf.dictconfig import DictConfig
from packaging import version as pversion
from pytorch_lightning import Trainer
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformer_engine.common.recipe import DelayedScaling, Format
from transformers import AutoModelForCausalLM

from sagemaker_nemo_adaptor.utils.fsdp_utils import (
    get_auto_wrap_policy,
    get_backward_fetch_policy,
    get_sharding_strategy,
    get_transformer_layer,
    set_mixed_precision_recipe,
)
from sagemaker_nemo_adaptor.utils.log_utils import Logger
from sagemaker_nemo_adaptor.utils.train_utils import (
    compute_num_params,  # TODO: Find a more integrated way to compute num params, probably using lightning ModelSummary: https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelSummary.html#lightning.pytorch.callbacks.ModelSummary
)
from sagemaker_nemo_adaptor.utils.train_utils import apply_activation_checkpoint

TF_VERSION = pversion.parse(transformers.__version__)

_logger = Logger().get_logger()


class SageMakerNLPBaseModel(NLPModel):
    """
    General Lightning Model class for SageMaker adaptor, it deals with general model/optimizer setup
    and training/eval behaviors.
    User will need to either consume the provided inheritors or inherit and implement their own model class.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer, use_smp=True, no_lm_init=True):
        """
        no_lm_init: Will only be False for BERT model
        """
        self._cfg = cfg
        self.use_smp = use_smp
        self.model = None
        self.grad_norm = None
        super().__init__(cfg, trainer, no_lm_init)
        # Only use custom grad clipping if using smp optimizations
        if self.use_smp:
            self.configure_gradient_clipping = self._smp_configure_gradient_clipping

    def get_model_config(self):
        """
        Get model config to build the model, should be implemented in specific model class
        """
        raise NotImplementedError(f"get_model_config is not implemented for {typr(self).__name__}")

    def setup(self, stage=None):
        """PTL hook that is executed after setup environment. chetodo: check whether it's called before or after strategy setup
            See https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#setup for more information.
        Args:
            stage (str, optional): Can be 'fit', 'validate', 'test' or 'predict'. Defaults to None.
        """
        # Building model
        model_config = self.get_model_config()
        if self._cfg.delayed_param:
            # TODO: revisit this when implementing fine tuning
            if self._cfg.finetune_with_pretrained_weights and dist.get_rank() == 0:
                # create model with pretrained weights on one rank even if we want to use
                # delayed param, param init on other ranks will still be delayed
                self.build_model(model_config)
            else:
                with init_empty_weights():
                    self.build_model(model_config)
            # TODO: check why this needed, and revisit for finetune case
            if self._cfg.do_finetune:
                dist.barrier()
        else:
            self.build_model(model_config)

        # Transfer model to SMP model
        if self.use_smp:
            moe_config = None  # TODO: Add moe support
            load_state_dict_from_rank0 = self._cfg.finetune_with_pretrained_weights
            from torch.sagemaker import transform

            self.model = transform(
                self.model,
                config=moe_config,
                load_state_dict_from_rank0=load_state_dict_from_rank0,
            )

        self.fp8_recipe = None
        if self._cfg.fp8 and self.use_smp:
            self.fp8_recipe = DelayedScaling(
                fp8_format=Format.HYBRID,
                amax_history_len=self._cfg.fp8_amax_history_len,
                amax_compute_algo=self._cfg.fp8_amax_compute_algo,
            )

        self.setup_fsdp()

    def setup_fsdp(self):
        """
        setup smp enabled FSDP
        """
        transformer_layer = get_transformer_layer(self._cfg.model_type, self.use_smp, self._cfg.moe)
        auto_wrap_policy = get_auto_wrap_policy(self._cfg.auto_wrap_policy, transformer_layer)
        mixed_precision_policy = set_mixed_precision_recipe(precision=self._cfg.precision, use_smp=self.use_smp)
        sharding_strategy = get_sharding_strategy(self._cfg.sharding_strategy)
        backward_prefetch = get_backward_fetch_policy(self._cfg.backward_fetch_policy)

        if self._cfg.delayed_param:
            param_init_fn, post_param_init_fn, model_context = self.delayed_param_init_fn()

        with model_context:
            self.model = FSDP(
                module=self.model,
                auto_wrap_policy=auto_wrap_policy,
                mixed_precision=mixed_precision_policy,
                sharding_strategy=sharding_strategy,
                backward_prefetch=backward_prefetch,
                forward_prefetch=self._cfg.forward_prefetch,
                limit_all_gathers=self._cfg.limit_all_gathers,
                device_id=torch.cuda.current_device(),
                use_orig_params=self._cfg.use_orig_param,
                param_init_fn=param_init_fn,
                sync_module_states=self._cfg.finetune_with_pretrained_weights,
            )
        if self._cfg.activation_checkpointing:
            apply_activation_checkpoint(
                model=self.model,
                model_type=self._cfg.model_type,
                use_smp=self.use_smp,
                fp8=self._cfg.fp8,
                moe=self._cfg.moe,
            )

        if self._cfg.offload_activations:
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                offload_wrapper,
            )

            model = offload_wrapper(model)
            if (
                self.use_smp and self._cfg.model_type == "gpt_neox" and self._cfg.patch_neox_rope
            ):  # TODO: add a check to the model type and use enum for it
                patch_neox_rope(model)

    def delayed_param_init_fn(self):
        """
        Initializes model on torch meta devices
        """
        param_init_fn = None
        post_param_init_fn = None
        model_context = nullcontext()
        delayed_param_initer = None
        # If using SMP use sagemaker optimized DelayedParamIniter function
        if self.use_smp:
            from torch.sagemaker.delayed_param import DelayedParamIniter

            # Depending on training job type, define DelayedParamIniter function across all ranks or subset of ranks
            if self._cfg.finetune_with_pretrained_weights:
                if self.global_rank != 0:
                    delayed_param_initer = DelayedParamIniter(self.model)
            else:
                delayed_param_initer = DelayedParamIniter(self.model)

            # If delayed_param_initer is defined in current rank return the pre and post init functions
            if delayed_param_initer:
                param_init_fn = delayed_param_initer.get_param_init_fn()
                post_param_init_fn = delayed_param_initer.get_post_param_init_fn()

                if not self._cfg.finetune_with_pretrained_weights:
                    model_context = delayed_param_initer.validate_params_and_buffers_inited()
        else:
            # Pulled param initialization function from open source meta/llama training recipes
            # https://github.com/meta-llama/llama-recipes/blob/f531d17287bf11d2cc2a5992e9282c77a70b2f51/src/llama_recipes/finetuning.py#L186C13-L186C103
            param_init_fn = lambda module: module.to_empty(device=torch.device("cuda"), recurse=False)
        return param_init_fn, post_param_init_fn, model_context

    def build_model(self, model_config):
        """
        Initialize model and configure flash attention
        """
        if self._cfg.pretrained_model_name_or_path:
            _logger.info("Loading pretrained weights from %s.", self._cfg.pretrained_model_name_or_path)

            if TF_VERSION < pversion.parse("4.37.1") or not self._cfg.use_flash_attention:

                self.model = AutoModelForCausalLM.from_pretrained(
                    self._cfg.pretrained_model_name_or_path, config=model_config
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self._cfg.pretrained_model_name_or_path,
                    attn_implementation="flash_attention_2",
                    config=model_config,
                )
        else:
            if TF_VERSION < pversion.parse("4.37.1") or not self._cfg.use_flash_attention:
                self.model = AutoModelForCausalLM.from_config(model_config)
            else:
                self.model = AutoModelForCausalLM.from_config(model_config, attn_implementation="flash_attention_2")

    def configure_flash_attn(self):
        """
        Configure flash attention, should be implemented in specific model class
        """
        raise NotImplementedError(f"configure_flash_attn is not implemented for {type(self).__name__}")

    def training_step(self, batch, batch_idx):
        """
        General training forward steps, backward/optimizer step will be done by PTL
        User can also skip auto optimization with self.automatic_optimization=False
        """
        input_ids, _, labels = self.trainer.datamodule.get_batch(batch)

        # uses default causal mask
        if self._cfg.fp8 and self.use_smp:
            import torch.sagemaker as tsm

            with transformer_engine.pytorch.fp8_autocast(
                enabled=self._cfg.fp8, fp8_recipe=self.fp8_recipe, fp8_group=tsm.state.world_process_group
            ):
                loss = self.model(input_ids=input_ids, attention_mask=None, labels=labels)["loss"]
        else:
            loss = self.model(input_ids=input_ids, attention_mask=None, labels=labels)["loss"]

        self.loss = loss
        return loss

    def setup_optimization(
        self,
        optim_config: Optional[Union[DictConfig, Dict]] = None,
        optim_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Inherit from nemo with removing megatron specific optimization implementation, i.e. distributed adam
        """
        # Ensure `max_steps` is set correctly
        optim_config_cp = self._optim_config_copy(optim_config) or {}
        max_steps = optim_config_cp.get("sched", {}).get("max_steps")

        if "sched" in optim_config_cp and max_steps is None:
            with open_dict(optim_config_cp):
                optim_config_cp.sched.max_steps = self._get_max_steps()

        optim_kwargs = {} if optim_kwargs is None else optim_kwargs.copy()

        return super().setup_optimization(optim_config=optim_config_cp, optim_kwargs=optim_kwargs)

    def configure_optimizers(self):
        """
        Inherit from nemo with removing megatron specific optimization implementation, i.e. distributed adam
        TODO: check lr and curve between this and rubik naive opt implementation
        """
        self.setup_optimization()

        if getattr(self._cfg.optim, "sched", None) is not None and self._scheduler is None:
            # The error below refers in particular to logs from `prepare_lr_scheduler()` (when it retunrs `None`).
            raise AssertionError(
                "A scheduler config exists but no scheduler was instantiated! Previous logs may help identify the "
                "root cause of this issue."
            )

        if self._scheduler is None:
            return self._optimizer
        else:
            return [self._optimizer], [self._scheduler]

    def _smp_configure_gradient_clipping(self, *args, **kwargs):
        """
        Cutomized gradient clipping for SMP
        TODO: figure out whether we should still use this for non-SMP usecase
        """
        from torch.sagemaker.grad_norm import clip_grad_norm_

        self.grad_norm = clip_grad_norm_(self.model, self._cfg.grad_clip)
        return

    def on_train_batch_end(self, *args, **kwargs):
        """
        Hook called at the end of each training batch, do logging here
        """
        loss_scalar = self._process_loss()
        self.log(
            "loss",
            loss_scalar,
            prog_bar=True,
        )

    def _process_loss(self):
        """General function to process loss after train/eval"""
        if self._cfg.log_reduced_training_loss:
            loss_detached = self.loss.detach()
            dist.all_reduce(loss_detached)
            loss_scalar = loss_detached.item() / dist.get_world_size()
            return loss_scalar
        else:
            loss_scalar = self.loss.item()
            return loss_scalar

    def _get_max_steps(self):
        """
        Compute the maximum number of training steps (-1 if it cannot be computed).
        Over write nemo's _get_max_steps with
        1. Override max step from config lr_decay_iters
        2. Get data loader length from datamodule
        """
        if self._cfg.lr_decay_iters is not None:
            return self._cfg.lr_decay_iters

        if getattr(self, "_trainer", None) is None:
            _logger.warning("Cannot compute `max_steps` as no trainer is set")
            return -1

        if self._trainer.max_steps >= 0:
            # Note that when `trainer.max_steps` is defined, we ignore `max_epochs` (even if training may end
            # before `max_steps` is reached due to `max_epochs`). This is for backward compatibility with older
            # versions of NeMo.
            if self._trainer.max_epochs is not None and self._trainer.max_epochs >= 0:
                _logger.warning(
                    "Ignoring `trainer.max_epochs` when computing `max_steps` because `trainer.max_steps` is already "
                    f"set to {self._trainer.max_steps}."
                )
            return self._trainer.max_steps

        if self._trainer.max_epochs is None or self._trainer.max_epochs < 0:
            _logger.warning("Cannot compute `max_steps` if neither `trainer.max_steps` nor `trainer.max_epochs` is set")
            return -1

        if getattr(self, "_train_dl", None) is None:
            _logger.warning("Cannot compute `max_steps` from the number of epochs as the train dataloader is not set")
            return -1

        # The number of training step per epoch is typically the number of global batches in the training set...
        num_global_batches = len(self.datamodule._train_dl)
        steps_per_epoch = num_global_batches

        # ... unless it is constrained by the `limit_train_batches` option.
        limit_batches = self._trainer.limit_train_batches
        if limit_batches is not None:
            if isinstance(limit_batches, float):
                limit_batches = int(limit_batches * num_global_batches)

            steps_per_epoch = min(num_global_batches, limit_batches)

        return steps_per_epoch * self._trainer.max_epochs

    def list_available_models(self):
        """Override Nemo's abstract class"""
        return None

    def setup_training_data(self):
        """We're using Data Module for data pipelining"""
        return None

    def setup_validation_data(self):
        """We're using Data Module for data pipelining"""
        return None
