from typing import Any, Dict, Optional, Union

import torch
import torch.distributed as dist
import torch.sagemaker as tsm
import transformer_engine
import transformers
from accelerate import init_empty_weights
from nemo.collections.nlp.models.nlp_model import NLPModel
from omegaconf import open_dict
from omegaconf.dictconfig import DictConfig
from packaging import version as pversion
from pytorch_lightning import Trainer
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.sagemaker import transform
from torch.sagemaker.delayed_param import DelayedParamIniter
from torch.sagemaker.grad_norm import clip_grad_norm_
from torch.sagemaker.logger import get_logger
from transformer_engine.common.recipe import DelayedScaling, Format
from transformers import AutoModelForCausalLM

from sagemaker_nemo_adaptor.utils.fsdp_utils import (
    get_auto_wrap_policy,
    get_backward_fetch_policy,
    get_sharding_strategy,
    get_transformer_layer,
    set_mixed_precision_recipe,
)
from sagemaker_nemo_adaptor.utils.train_utils import (
    compute_num_params,  # TODO: Find a more integrated way to compute num params, probably using lightning ModelSummary: https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelSummary.html#lightning.pytorch.callbacks.ModelSummary
)
from sagemaker_nemo_adaptor.utils.train_utils import apply_activation_checkpoint

_logger = get_logger()


class SageMakerNLPBaseModel(NLPModel):
    """
    General Lightning Model class for SageMaker adaptor, it deals with general model/optimizer setup
    and training/eval behaviors.
    User will need to either consume the provided inheritors or inherite and implement their own model class.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer, no_lm_init=True):
        """
        no_lm_init: Will only be False for BERT model
        """

        self.use_smp = cfg.use_smp
        self.cfg = cfg

        self.grad_norm = None
        super().__init__(cfg, trainer, no_lm_init)

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
        if self.cfg.delayed_param:
            # TODO: revisit this when implementing fine tuning
            if self.cfg.finetune_with_pretrained_weights and dist.get_rank() == 0:
                # create model with pretrained weights on one rank even if we want to use
                # delayed param, param init on other ranks will still be delayed
                self.build_model(model_config)
            else:
                with init_empty_weights():
                    self.build_model(model_config)
            # TODO: check why this needed, and revisit for finetune case
            if self.cfg.do_finetune:
                dist.barrier()
        else:
            self.build_model(model_config)

        # Transfer model to SMP model
        if self.cfg.use_smp:
            moe_config = None  # TODO: Add moe support
            load_state_dict_from_rank0 = self.cfg.finetune_with_pretrained_weights
            self.model = transform(
                self.model,
                config=moe_config,
                load_state_dict_from_rank0=load_state_dict_from_rank0,
            )

        self.fp8_recipe = None
        if self.cfg.fp8 and self.cfg.use_smp:
            self.fp8_recipe = DelayedScaling(
                fp8_format=Format.HYBRID,
                amax_history_len=self.cfg.fp8_amax_history_len,
                amax_compute_algo=self.cfg.fp8_amax_compute_algo,
            )
        self.setup_fsdp()

    def setup_fsdp(self):
        """
        setup FSDP
        """
        transformer_layer = get_transformer_layer(self.cfg.model_type, self.cfg.use_smp, self.cfg.moe)
        auto_wrap_policy = get_auto_wrap_policy(self.cfg.auto_wrap_policy, transformer_layer)
        mixed_precision_policy = set_mixed_precision_recipe(
            precision=self.cfg.trainer.precision, use_smp=self.cfg.use_smp
        )
        sharding_strategy = get_sharding_strategy(self.cfg.sharding_strategy)
        backward_prefetch = get_backward_fetch_policy(self.cfg.backward_fetch_policy)

        if self.cfg.delayed_param:
            if self.cfg.finetune_with_pretrained_weights:
                if self.global_rank != 0:
                    delayed_param_initer = DelayedParamIniter(self.model)
            else:
                delayed_param_initer = DelayedParamIniter(self.model)

        if delayed_param_initer:
            param_init_fn = delayed_param_initer.get_param_init_fn()
            post_param_init_fn = delayed_param_initer.get_post_param_init_fn()
        else:
            param_init_fn = None
            post_param_init_fn = None

        with (
            delayed_param_initer.validate_params_and_buffers_inited()
            if (delayed_param_initer and not self.cfg.finetune_with_pretrained_weights)
            else nullcontext()
        ):
            self.model = FSDP(
                module=self.model,
                auto_wrap_policy=auto_wrap_policy,
                mixed_precision=mixed_precision_policy,
                sharding_strategy=sharding_strategy,
                backward_prefetch=backward_prefetch,
                forward_prefetch=self.cfg.forward_prefetch,
                limit_all_gathers=self.cfg.limit_all_gathers,
                device_id=torch.cuda.current_device(),
                use_orig_params=self.cfg.use_orig_param,
                param_init_fn=param_init_fn,
                post_param_init_fn=post_param_init_fn,
                sync_module_states=self.cfg.finetune_with_pretrained_weights,
            )

        apply_activation_checkpoint(
            model=self.model,
            model_type=self.cfg.model_type,
            use_smp=self.cfg.use_smp,
            fp8=self.cfg.fp8,
            moe=self.cfg.moe,
        )

        if self.cfg.offload_activations:
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                offload_wrapper,
            )

            model = offload_wrapper(model)
            if self.cfg.use_smp and self.cfg.model_type == "gpt_neox" and self.cfg.patch_neox_rope:
                patch_neox_rope(model)

        if torch.distributed.get_rank() == 0:
            print(f"che calling model setup finished")

    def build_model(self, model_config):
        """
        Initialize model and configure flash attention
        """
        if self.cfg.pretrained_model_weights:
            _logger.info("Loading pretrained weights from %s.", self.cfg.pretrained_model_weights)
            if pversion.parse(transformers.__version__) < pversion.parse("4.37.1"):
                self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_weights, config=model_config)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    pretrained_model_weights, attn_implementation="flash_attention_2", config=model_config
                )
        else:
            if pversion.parse(transformers.__version__) < pversion.parse("4.37.1"):
                self.model = AutoModelForCausalLM.from_config(model_config)
            else:
                self.model = AutoModelForCausalLM.from_config(model_config, attn_implementation="flash_attention_2")

        if self.cfg.use_smp_flash_attn:
            self.configure_flash_attn()

    def configure_flash_attn(self):
        """
        Configure flash attention, should be implemented in specific model class
        """
        raise NotImplementedError(f"configure_flash_attn is not implemented for {typr(self).__name__}")

    def training_step(self, batch, batch_idx):
        """
        General training forward steps, backward/optimizer step will be done by PTL
        User can also skip auto optimization with self.automatic_optimization=False
        """
        input_ids, _, labels = self.trainer.datamodule.get_batch(batch)
        # uses default causal mask
        if self.cfg.fp8 and self.cfg.use_smp:
            with transformer_engine.pytorch.fp8_autocast(
                enabled=self.cfg.fp8, fp8_recipe=self.fp8_recipe, fp8_group=tsm.state.world_process_group
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
        optim_config = self._optim_config_copy(optim_config)
        if optim_config is not None and "sched" in optim_config and optim_config.sched.get("max_steps") is None:
            with open_dict(optim_config):
                optim_config.sched.max_steps = self._get_max_steps()

        optim_kwargs = {} if optim_kwargs is None else optim_kwargs.copy()

        return super().setup_optimization(optim_config=optim_config, optim_kwargs=optim_kwargs)

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

        if torch.distributed.get_rank() == 0:
            print(f"che calling opt setup fiished")

    def configure_gradient_clipping(self, *args, **kwargs):
        """
        Cutomized gradient clipping for SMP
        TODO: figure out whether we should still use this for non-SMP usecase
        """
        self.grad_norm = clip_grad_norm_(self.model, self.cfg.grad_clip)
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
        if self.cfg.log_reduced_training_loss:
            loss_detached = self.loss.detach()
            dist.all_reduce(loss_detached)
            loss_scalar = loss_detached.item() / dist.get_world_size()
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
        if self.cfg.lr_decay_iters is not None:
            return self.cfg.lr_decay_iters
        else:
            if getattr(self, "_trainer", None) is None:
                logging.warning("Cannot compute `max_steps` as no trainer is set")
                return -1

            if self._trainer.max_steps >= 0:
                # Note that when `trainer.max_steps` is defined, we ignore `max_epochs` (even if training may end
                # before `max_steps` is reached due to `max_epochs`). This is for backward compatibility with older
                # versions of NeMo.
                if self._trainer.max_epochs is not None and self._trainer.max_epochs >= 0:
                    logging.warning(
                        "Ignoring `trainer.max_epochs` when computing `max_steps` because `trainer.max_steps` is already "
                        f"set to {self._trainer.max_steps}."
                    )
                return self._trainer.max_steps

            if self._trainer.max_epochs is None or self._trainer.max_epochs < 0:
                logging.warning(
                    "Cannot compute `max_steps` if neither `trainer.max_steps` nor `trainer.max_epochs` is set"
                )
                return -1

            if getattr(self, "_train_dl", None) is None:
                logging.warning(
                    "Cannot compute `max_steps` from the number of epochs as the train dataloader is not set"
                )
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
