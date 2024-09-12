"""
WIP data module for megatron data pipelining, TODO: test/refine it when implementing fine-tuning
"""


from nemo.collections.nlp.data.language_modeling.megatron.base_dataset_utils import (
    get_datasets_weights_and_num_samples,
)
from nemo.collections.nlp.data.language_modeling.megatron.blendable_dataset import (
    MemoryEfficientBlendableDataset,
)
from nemo.collections.nlp.data.language_modeling.megatron.megatron_batch_samplers import (
    MegatronPretrainingBatchSampler,
    MegatronPretrainingRandomBatchSampler,
)
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.utils import logging

from sagemaker_nemo_adaptor.collections.data import BaseDataModule, SkipDataLoader
from sagemaker_nemo_adaptor.collections.data.datasets import (
    build_train_valid_test_datasets,
)


class MegatronDataModule(BaseDataModule):
    def setup(self, stage=None):
        super().setup()
        if hasattr(self.cfg.model.data, "tokenizer"):
            # build tokenizer (defaults to nemo supported tokenizers)
            self._build_tokenizer()

            # manipulate vocabulary (e.g., pad vocabulary for better efficiency)
            self._build_vocab()

        if stage == "predict":
            return
        elif self.cfg.model.data.get("fine_tuning", False):
            self.build_sft_datasets()
        else:
            # TODO: consider adding a ModelPT guard to check if model is being restored.
            # allowing restored models to optionally setup datasets
            self.build_train_valid_test_datasets()

    def train_dataloader(self):
        if hasattr(self, "_train_ds"):
            consumed_samples = self.compute_consumed_samples(0)
            logging.info(
                f"Setting up train dataloader with len(len(self._train_ds)): {len(self._train_ds)} and consumed samples: {consumed_samples}"
            )
            if self.cfg.model.data.get("fine_tuning", False):
                return self.build_fine_tuning_data_loader(self._train_ds, cfg.train_ds)
            else:
                return self.build_pretraining_data_loader(self._train_ds, consumed_samples)

    def val_dataloader(self):
        if hasattr(self, "_validation_ds"):
            consumed_samples = 0
            logging.info(
                f"Setting up validation dataloader with len(len(self._validation_ds)): {len(self._validation_ds)} and consumed samples: {consumed_samples}"
            )

            drop_last = True
            if not self.cfg.model.data.get("validation_drop_last", True):
                logging.info(f"Drop last in validation dataset is set to False")
                drop_last = False
            pad_samples_to_global_batch_size = False
            if self.cfg.model.data.get("pad_samples_to_global_batch_size", False):
                logging.info("pad_samples_to_global_batch_size set to True")
                pad_samples_to_global_batch_size = True
            if self.cfg.model.data.get("fine_tuning", False):
                return self.build_fine_tuning_data_loader(self._validation_ds, self.cfg.model.data.validation_ds)
            else:
                return self.build_pretraining_data_loader(
                    self._validation_ds, consumed_samples, "validation", drop_last, pad_samples_to_global_batch_size
                )

    def predict_dataloader(self):
        if hasattr(self, "_test_ds"):
            consumed_samples = 0
            logging.info(
                f"Setting up test dataloader with len(len(self._test_ds)): {len(self._test_ds)} and consumed samples: {consumed_samples}"
            )
            if self.cfg.model.data.get("fine_tuning", False):
                return self.build_fine_tuning_data_loader(self._test_ds, self.cfg.model.data.test_ds)
            else:
                return self.build_pretraining_data_loader(self._test_ds, consumed_samples)

    def build_train_valid_test_datasets(self):
        logging.info("Building GPT datasets.")
        if self.trainer.limit_val_batches > 1.0 and isinstance(self.trainer.limit_val_batches, float):
            raise ValueError("limit_val_batches must be an integer or float less than or equal to 1.0.")
        global_batch_size = self.cfg.model.data.global_batch_size
        max_train_steps = self.trainer.max_steps
        eval_iters = (max_train_steps // self.trainer.val_check_interval + 1) * self.trainer.limit_val_batches
        test_iters = self.trainer.limit_test_batches

        train_valid_test_num_samples = [
            max_train_steps * global_batch_size,
            eval_iters * global_batch_size,
            test_iters * global_batch_size,
        ]

        if self.trainer.limit_val_batches <= 1.0 and isinstance(self.trainer.limit_val_batches, float):
            train_valid_test_num_samples[
                1
            ] = 1  # This is to make sure we only have one epoch on every validation iteration

        self._train_ds, self._validation_ds, self._test_ds = build_train_valid_test_datasets(
            cfg=self.cfg,
            trainer=self.trainer,
            data_prefix=self.cfg.model.data.data_prefix,
            data_impl=self.cfg.model.data.data_impl,
            splits_string=self.cfg.model.data.splits_string,
            train_valid_test_num_samples=train_valid_test_num_samples,
            seq_length=self.cfg.model.data.seq_length,
            seed=self.cfg.model.seed,
            skip_warmup=self.cfg.model.data.get("skip_warmup", True),
            tokenizer=self.tokenizer,
        )
        if self._train_ds is not None:
            logging.info(f"Length of train dataset: {len(self._train_ds)}")
        if self._validation_ds is not None:
            logging.info(f"Length of val dataset: {len(self._validation_ds)}")
        if self._test_ds is not None:
            logging.info(f"Length of test dataset: {len(self._test_ds)}")
        logging.info(f"Finished building GPT datasets.")
        xm.rendezvous("finished_dataloader")

        return self._train_ds, self._validation_ds, self._test_ds

    def build_pretraining_data_loader(
        self, dataset, consumed_samples, dataset_type=None, drop_last=True, pad_samples_to_global_batch_size=False
    ):
        """Buld dataloader given an input dataset."""

        logging.info(f"Building dataloader with consumed samples: {consumed_samples}")
        # Megatron sampler
        if hasattr(self.cfg.model.data, "dataloader_type") and self.cfg.model.data.dataloader_type is not None:
            if self.cfg.model.data.dataloader_type == "single":
                batch_sampler = MegatronPretrainingBatchSampler(
                    total_samples=len(dataset),
                    consumed_samples=consumed_samples,
                    micro_batch_size=self.cfg.model.data.micro_batch_size,
                    global_batch_size=self.cfg.model.data.global_batch_size,
                    data_parallel_rank=self.dp_rank,
                    data_parallel_size=self.dp_size,
                    drop_last=drop_last,
                    pad_samples_to_global_batch_size=pad_samples_to_global_batch_size,
                )
            elif self.cfg.model.data.dataloader_type == "cyclic":
                batch_sampler = MegatronPretrainingRandomBatchSampler(
                    total_samples=len(dataset),
                    consumed_samples=consumed_samples,
                    micro_batch_size=self.cfg.model.data.micro_batch_size,
                    global_batch_size=self.cfg.model.data.global_batch_size,
                    data_parallel_rank=self.dp_rank,
                    data_parallel_size=self.dp_size,
                    drop_last=self.cfg.model.data.get("drop_last", True),
                    pad_samples_to_global_batch_size=pad_samples_to_global_batch_size,
                )
            else:
                raise ValueError('cfg.model.data.dataloader_type must be "single" or "cyclic"')  # TODO: Change to Enum
        else:
            raise ValueError('cfg.model.data.dataloader_type not found. Must be "single" or "cyclic"')

        return SkipDataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=self.cfg.model.data.num_workers,
            pin_memory=False,
            prefetch_factor=1,
        )

    def build_sft_datasets(self):
        if self.trainer.limit_val_batches > 1.0 and isinstance(self.trainer.limit_val_batches, float):
            raise ValueError("limit_val_batches must be an integer or float less than or equal to 1.0.")

        if (
            hasattr(self.cfg.model.data, "validation_ds")
            and self.cfg.model.data.validation_ds.get("file_names", None) is not None
        ):
            logging.info("Building GPT SFT validation datasets.")
            # Wrap this in a list since the general finetuning parent class supports multi-validation.
            self._validation_ds = self._build_dataset(
                self.cfg.model.data.validation_ds, is_train=False, is_validation=True
            )
            logging.info(f"Length of val dataset: {len(self._validation_ds)}")

        if hasattr(self.cfg.model.data, "test_ds") and self.cfg.model.data.test_ds.get("file_names", None) is not None:
            logging.info("Building GPT SFT test datasets.")
            # Wrap this in a list since the general finetuning parent class supports multi-validation.
            self._test_ds = self._build_dataset(self.cfg.model.data.test_ds, is_train=False)
            logging.info(f"Length of test dataset: {len(self._test_ds[0])}")

        logging.info("Building GPT SFT training datasets.")
        self._train_ds = self._build_dataset(self.cfg.model.data.train_ds)
        logging.info(f"Length of train dataset: {len(self._train_ds)}")

    def _build_dataset(self, data_cfg, is_train=True, is_validation=False):
        datasets = []
        # Determine if we are using a single dataset or a list of datasets.
        is_list_config = isinstance(data_cfg.file_names, ListConfig)
        if not is_list_config:
            raise ValueError(f"SFT train/validation datasets must be provided as a list of individual JSONL files.")
        max_steps = 0
        if is_train:
            max_steps = self.trainer.max_steps
        elif is_validation:
            max_steps = (self.trainer.max_steps // self.trainer.val_check_interval + 1) * self.trainer.limit_val_batches

        # Construct the data prefix list for `get_datasets_weights_and_num_samples()`
        # that is of the format [weight1,file_name1,weight2,file_name2,...]
        if data_cfg.concat_sampling_probabilities is None or not isinstance(
            data_cfg.concat_sampling_probabilities, ListConfig
        ):
            raise ValueError(
                (
                    f"concat_sampling_probabilities must be a ListConfig with the same number of files in file_names."
                    f"Found: {data_cfg.concat_sampling_probabilities}"
                )
            )

        if len(data_cfg.get("concat_sampling_probabilities", None)) != len(data_cfg.file_names):
            raise ValueError(
                (
                    f"concat_sampling_probabilities must be of the same size as file_names.",
                    f"Provided size {len(data_cfg.concat_sampling_probabilities)}, number of datasets {len(data_cfg.file_names)}",
                )
            )

        data_prefix = []
        for weight, prefix in zip(data_cfg.concat_sampling_probabilities, data_cfg.file_names):
            data_prefix.append(weight)
            data_prefix.append(prefix)

        num_train_samples = [max_steps * self.cfg.model.data.global_batch_size]
        _, _, num_train_samples_per_dataset = get_datasets_weights_and_num_samples(data_prefix, num_train_samples)
        num_train_samples_after_blend = sum([x[0] for x in num_train_samples_per_dataset])

        # Check dataset max_seq_length and max_position_embeddings size
        if (
            self.cfg.model.get("position_embedding_type", None) in [None, "learned_absolute"]
            and data_cfg.max_seq_length > self.cfg.model.max_position_embeddings
        ):
            logging.warning(
                f"Set dataset max_seq_length to max_position_embeddings {self.cfg.model.max_position_embeddings} if using learned_absolute position embedding"
            )
            data_cfg.max_seq_length = self.cfg.model.max_position_embeddings

        for file_path, num_samples in zip(data_cfg.file_names, num_train_samples_per_dataset):
            if data_cfg.get("chat", False):
                dataset_cls = GPTSFTChatDataset
            else:
                dataset_cls = GPTSFTDataset

            dataset = dataset_cls(
                file_path=file_path,
                tokenizer=self.tokenizer,
                max_seq_length=data_cfg.max_seq_length,
                min_seq_length=data_cfg.min_seq_length,
                add_bos=data_cfg.get("add_bos", False),
                add_eos=data_cfg.get("add_eos", True),
                add_sep=data_cfg.get("add_sep", False),
                sep_id=None,
                max_num_samples=num_samples[0],
                seed=data_cfg.get("seed", 1234),
                label_key=data_cfg.get("label_key", "answer"),
                answer_only_loss=data_cfg.get("answer_only_loss", True),
                truncation_field=data_cfg.get("truncation_field", "text"),
                pad_to_max_length=data_cfg.get("pad_to_max_length", True),
                index_mapping_dir=data_cfg.get("index_mapping_dir", None),
                prompt_template=data_cfg.get("prompt_template", None),
                virtual_tokens=0,
                tokens_to_generate=data_cfg.get(
                    "tokens_to_generate", 0
                ),  # used at inference time to allocate tensor positions for tokens that will be generated by inf procedure.
                memmap_workers=data_cfg.get(
                    "memmap_workers", None
                ),  # used to set num. of workers to create the memmap index files
                hf_dataset=data_cfg.get(
                    "hf_dataset", False
                ),  # Whether to load the json file with the HuggingFace dataset. otherwise, will load the jsonl file with the JSONLMemMapDataset.
                truncation_method=data_cfg.get(
                    "truncation_method", "right"
                ),  # used to choose truncation method. Options: ['random', 'left', 'right']
            )
            datasets.append(dataset)

        dataset = MemoryEfficientBlendableDataset(
            datasets=datasets, weights=data_cfg.concat_sampling_probabilities, size=num_train_samples_after_blend
        )
        return dataset

    def build_fine_tuning_data_loader(self, dataset, data_cfg, consumed_samples=0):
        """Buld fine tuning dataloader given an input dataset."""

        logging.info(f"Building fine tuning dataloader with consumed samples: {consumed_samples}")

        collate_fn = dataset.datasets[0].collate_fn

        batch_sampler = MegatronPretrainingBatchSampler(
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=self.cfg.model.data.micro_batch_size,
            global_batch_size=self.cfg.model.data.global_batch_size,
            data_parallel_rank=self.dp_rank,
            data_parallel_size=self.dp_size,
            drop_last=True,
            pad_samples_to_global_batch_size=False,
        )
        return SkipDataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            num_workers=data_cfg.num_workers,
            pin_memory=False,
            prefetch_factor=1,
        )

    def _build_tokenizer(self):
        """
        Default tokenizer is based on available nemo tokenizers.
        Override this method to use an external tokenizer.
        All tokenizers are expected to provide compatible interface.
        Override default Encoder-decoder tokenizer to use legacy=True for sentencepiece.
        """
        if hasattr(self.cfg.model.data.tokenizer, "sentencepiece_legacy"):
            legacy = self.cfg.model.data.tokenizer.sentencepiece_legacy
        else:
            legacy = True if self.cfg.model.data.tokenizer.library == "sentencepiece" else False
        self.tokenizer = get_nmt_tokenizer(
            library=self.cfg.model.data.tokenizer.library,
            model_name=self.cfg.model.data.tokenizer.type,
            tokenizer_model=self.trainer.model.register_artifact(
                "tokenizer.model", self.cfg.model.data.tokenizer.model
            ),
            vocab_file=self.trainer.model.register_artifact(
                "tokenizer.vocab_file", self.cfg.model.data.tokenizer.vocab_file
            ),
            merges_file=self.trainer.model.register_artifact(
                "tokenizer.merge_file", self.cfg.model.data.tokenizer.merge_file
            ),
            delimiter=self.cfg.model.data.tokenizer.get("delimiter", None),
            legacy=legacy,
        )

    def get_batch_length(self, batch):
        return len(batch["tokens"])

    def process_global_batch(self, global_batch, global_batch_size=None):
        """Prepares the global batch for apex fwd/bwd functions.
        Global batch is a list of micro batches.
        """
        return [
            global_batch["tokens"],
            global_batch["labels"],
            global_batch["loss_mask"],
            global_batch["attention_mask"],
            global_batch["position_ids"],
        ]
