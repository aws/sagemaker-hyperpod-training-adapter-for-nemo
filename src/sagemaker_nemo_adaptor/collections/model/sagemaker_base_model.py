import os
from typing import Any, Dict, Optional, Union

import torch
import torch.distributed as dist
import torch.sagemaker as tsm
import transformer_engine
import transformers
from accelerate import init_empty_weights
from nemo.core.classes.modelPT import ModelPT
from omegaconf import open_dict
from omegaconf.dictconfig import DictConfig
from packaging import version as pversion
from peft import LoraConfig, get_peft_model
from pytorch_lightning import Trainer
from torch.sagemaker import transform
from transformer_engine.common.recipe import DelayedScaling, Format
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from sagemaker_nemo_adaptor.utils.log_utils import Logger

TF_VERSION = pversion.parse(transformers.__version__)

_logger = Logger().get_logger()


class SageMakerNLPBaseModel(ModelPT):
    """
    General Lightning Model class for SageMaker adaptor, it deals with general model/optimizer setup
    and training/eval behaviors.
    User will need to either consume the provided inheritors or inherit and implement their own model class.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer, use_smp=True):
        self._cfg = cfg
        self.model = None
        self.use_smp = use_smp
        super().__init__(cfg, trainer)

    def get_model_config(self):
        """
        Get model config to build the model, should be implemented in specific model class
        """
        cls = type(self).__name__
        raise NotImplementedError(f"{cls}.get_model_config not implemented")

    @property
    def use_peft(self):
        return self._cfg.peft.peft_type is not None

    @property
    def do_finetune(self):
        """
        Returns true if finetuning
        """
        return self._cfg.get("pretrained_model_name_or_path", None) is not None

    @property
    def do_finetune_with_pretrained_weights(self):
        """
        Returns true for start of finetuning only, meaning we don't have a checkpoint and need to load pretrained weights
        """
        return self.do_finetune and self._cfg.get("resume_from_checkpoint", None) is None

    def setup(self, *a, **kw):
        if self.do_finetune:
            from transformers import AutoConfig

            # Using config from the pretrained model
            model_cfg = AutoConfig.from_pretrained(self._cfg.pretrained_model_name_or_path)
            # Disable KV cache for HF models
            if hasattr(model_cfg, "use_cache"):
                model_cfg.use_cache = False
        else:
            model_cfg = self.get_model_config()
        model = self._setup_delayed_param(model_cfg)
        if self.do_finetune_with_pretrained_weights:
            dist.barrier()
        self.model = self._transform(model)
        self.fp8_recipe = self._fp8_delayed_scaling()

    def param_init_fn(self, module):
        _logger.warning(
            f"A _param_init_fn has not been implemented for the current model class. Proceeding to train with delayed_param={self._cfg.delayed_param} will lead to convergence issues."
        )
        return module.to_empty(device=torch.device("cuda"), recurse=False)

    def _fp8_delayed_scaling(self):
        if self.use_smp and self._cfg.fp8:
            return DelayedScaling(
                fp8_format=Format.HYBRID,
                amax_history_len=self._cfg.fp8_amax_history_len,
                amax_compute_algo=self._cfg.fp8_amax_compute_algo,
            )

    def _transform(self, model):
        if not self.use_smp:
            return model

        moe_config = None  # TODO: Add moe support
        load_state_dict_from_rank0 = self.do_finetune_with_pretrained_weights
        return transform(
            model,
            config=moe_config,
            load_state_dict_from_rank0=load_state_dict_from_rank0,
        )

    def _setup_delayed_param(self, model_cfg):
        if not self._cfg.delayed_param:
            return self.build_model(model_cfg)
        if self.do_finetune_with_pretrained_weights and dist.get_rank() == 0:
            return self.build_model(model_cfg)
        with init_empty_weights():
            return self.build_model(model_cfg)

    def build_model(self, model_cfg):
        if self.use_peft:
            return self._build_model_from_pretrain_peft(model_cfg)
        if self.do_finetune_with_pretrained_weights and dist.get_rank() == 0:
            return self._build_model_from_pretrain(model_cfg)
        return self._build_model(model_cfg)

    def _build_model_from_pretrain_peft(self, model_cfg):
        assert not self.use_smp, "Must set use_smp=False to use PEFT"
        assert not self._cfg.delayed_param, "Must set delayed_param=False to use PEFT"
        assert self.do_finetune, "Must provide pretrained weights to use PEFT"

        # set env vars for efficient HF model loading (PEFT does not use SMP delayed param)
        # see https://tiny.amazon.com/15r3rmil3/githhuggtranblob2790srctran
        os.environ["ACCELERATE_USE_FSDP"] = "True"
        os.environ["FSDP_CPU_RAM_EFFICIENT_LOADING"] = "True"

        if self._cfg.peft.peft_type == "qlora_4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_quant_storage=torch.bfloat16,
            )
        else:
            quantization_config = None

        model = self._build_model_from_pretrain(
            model_cfg, torch_dtype=torch.bfloat16, quantization_config=quantization_config
        )

        lora_config = LoraConfig(
            target_modules="all-linear",
            # Alpha parameter for LoRA scaling
            lora_alpha=self._cfg.peft.alpha,
            # Dropout probability for LoRA layers
            lora_dropout=self._cfg.peft.dropout,
            # LoRA attention dimension
            r=self._cfg.peft.rank,
            bias="none",
            task_type="CAUSAL_LM",
            inference_mode=False,
        )

        model.enable_input_require_grads()
        model = get_peft_model(model, lora_config)
        if dist.get_rank() == 0:
            model.print_trainable_parameters()
        return model

    def _build_model_from_pretrain(self, model_cfg, torch_dtype=None, quantization_config=None):
        path = self._cfg.pretrained_model_name_or_path
        _logger.info("Loading pretrained weights from %s.", path)
        use_flash_attn = self._cfg.use_flash_attention
        attn = "flash_attention_2"
        if TF_VERSION < pversion.parse("4.37.1") or not use_flash_attn:
            return AutoModelForCausalLM.from_pretrained(
                path, config=model_cfg, torch_dtype=torch_dtype, quantization_config=quantization_config
            )
        return AutoModelForCausalLM.from_pretrained(
            path,
            attn_implementation=attn,
            config=model_cfg,
            torch_dtype=torch_dtype,
            quantization_config=quantization_config,
        )

    def _build_model(self, model_cfg):
        use_flash_attn = self._cfg.use_flash_attention
        attn = "flash_attention_2"
        if TF_VERSION < pversion.parse("4.37.1") or not use_flash_attn:
            return AutoModelForCausalLM.from_config(model_cfg)
        return AutoModelForCausalLM.from_config(
            model_cfg,
            attn_implementation=attn,
        )

    def _training_step_fp8(self, batch, batch_idx, *a, **kw):
        fp8 = self._cfg.fp8
        fp8_recipe = self.fp8_recipe
        fp8_group = tsm.state.world_process_group
        input_ids, _, labels = self.trainer.datamodule.get_batch(batch)
        with transformer_engine.pytorch.fp8_autocast(
            enabled=fp8,
            fp8_recipe=fp8_recipe,
            fp8_group=fp8_group,
        ):
            return self(
                *a,
                input_ids=input_ids,
                attention_mask=None,
                labels=labels,
                **kw,
            )["loss"]

    def _training_step(self, batch, batch_idx, *a, **kw):
        input_ids, _, labels = self.trainer.datamodule.get_batch(batch)
        return self(
            *a,
            input_ids=input_ids,
            attention_mask=None,
            labels=labels,
            **kw,
        )["loss"]

    def training_step(self, batch, batch_idx, *a, **kw):
        """
        General training forward steps, backward/optimizer step will be done by
        PTL can also skip auto optimization with self.automatic_optimization=False
        """
        if self._cfg.fp8 and self.use_smp:
            self.loss = self._training_step_fp8(batch, batch_idx, *a, **kw)
        else:
            self.loss = self._training_step(batch, batch_idx, *a, **kw)
        return self.loss

    def setup_optimization(
        self,
        optim_config: Optional[Union[DictConfig, Dict]] = None,
        optim_kwargs: Optional[Dict[str, Any]] = None,
    ):
        optim_config_cp = self._optim_config_copy(optim_config) or {}
        max_steps = optim_config_cp.get("sched", {}).get("max_steps")
        if "sched" in optim_config_cp and max_steps is None:
            with open_dict(optim_config_cp):
                optim_config_cp.sched.max_steps = self._get_max_steps()

        optim_kwargs = {} if optim_kwargs is None else optim_kwargs.copy()
        return super().setup_optimization(
            optim_config=optim_config_cp,
            optim_kwargs=optim_kwargs,
        )

    def configure_optimizers(self):
        self.setup_optimization()
        if getattr(self._cfg.optim, "sched", None) and self._scheduler is None:
            # The error below refers in particular to logs from
            # `prepare_lr_scheduler()` (when it retunrs `None`).
            raise AssertionError(
                "A scheduler config exists but no scheduler was instantiated!"
                "Previous logs may help identify the root cause of this issue."
            )
        if self._scheduler is None:
            return self._optimizer
        return [self._optimizer], [self._scheduler]

    def forward(self, *a, **kw):
        return self.model(*a, **kw)

    def on_train_batch_end(self, *args, **kwargs):
        """
        Hook called at the end of each training batch, do logging here
        """
        loss_scalar = self._process_loss()
        self.log("loss", loss_scalar, prog_bar=True)

    def _process_loss(self):
        """General function to process loss after train/eval"""
        if self._cfg.log_reduced_training_loss:
            loss_detached = self.loss.detach()
            dist.all_reduce(loss_detached)
            loss_scalar = loss_detached.item() / dist.get_world_size()
            return loss_scalar

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
            # Note that when `trainer.max_steps` is defined, we ignore
            # `max_epochs` (even if training may end before `max_steps` is
            # reached due to `max_epochs`). This is for backward compatibility
            # with older versions of NeMo.
            if self._trainer.max_epochs is not None and self._trainer.max_epochs >= 0:
                _logger.warning(
                    "Ignoring `trainer.max_epochs` when computing `max_steps` "
                    "because `trainer.max_steps` is already "
                    f"set to {self._trainer.max_steps}."
                )
            return self._trainer.max_steps

        if self._trainer.max_epochs is None or self._trainer.max_epochs < 0:
            _logger.warning("Cannot compute `max_steps` if neither `trainer.max_steps` nor `trainer.max_epochs` is set")
            return -1

        if getattr(self, "_train_dl", None) is None:
            _logger.warning("Cannot compute `max_steps` from the number of epochs as the train dataloader is not set")
            return -1

        # The number of training step per epoch is typically the number of
        # global batches in the training set...
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
