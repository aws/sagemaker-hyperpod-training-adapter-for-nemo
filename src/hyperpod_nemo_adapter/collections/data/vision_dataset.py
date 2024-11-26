# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

import copy
import random
from itertools import islice

import numpy as np
import torch
import torch.distributed as dist
from datasets import load_dataset
from torch.utils.data import DistributedSampler
from transformers import default_data_collator
from transformers.data import DataCollatorForSeq2Seq


# check system prompt token seq or user prompt token seq is in the current token list
def check_header(targets, seq):
    for i in range(len(seq) - 3):
        if seq[i : i + 3] in targets:
            return True
    return False


def replace_target(target, seq):
    for i in range(len(seq) - 3):
        if seq[i : i + 3] == target:
            seq[i], seq[i + 1], seq[i + 2] = -100, -100, -100
    return seq


def tokenize_dialogs(dialogs, images, processor):
    text_prompt = processor.apply_chat_template(dialogs)
    batch = processor(images=images, text=text_prompt, padding=True, return_tensors="pt")
    label_list = []
    for i in range(len(batch["input_ids"])):
        dialog_tokens = batch["input_ids"][i].tolist()
        labels = copy.copy(dialog_tokens)
        eot_indices = [i for i, n in enumerate(labels) if n == 128009]
        last_idx = 0
        # system prompt header "<|start_header_id|>system<|end_header_id|>" has been tokenized to [128006, 9125, 128007]
        # user prompt header "<|start_header_id|>user<|end_header_id|>" has been tokenized to [128006, 882, 128007]
        prompt_header_seqs = [[128006, 9125, 128007], [128006, 882, 128007]]
        for n, idx in enumerate(eot_indices):
            current_seq = labels[last_idx : idx + 1]
            if check_header(prompt_header_seqs, current_seq):
                # found prompt header, indicating that this seq should be masked
                labels[last_idx : idx + 1] = [-100] * (idx - last_idx + 1)
            else:
                last_idx = idx + 1
            #  Mask all the assistant header prompt <|start_header_id|>assistant<|end_header_id|>, which has been tokenized to [128006, 78191, 128007]
        assistant_header_seq = [128006, 78191, 128007]
        labels = replace_target(assistant_header_seq, labels)
        # Mask the padding token and image token 128256
        for i in range(len(labels)):
            if labels[i] == processor.tokenizer.pad_token_id or labels[i] == 128256:  #  128256 is image token index
                labels[i] = -100
        label_list.append(labels)
    batch["labels"] = torch.tensor(label_list)
    return batch


def get_custom_dataset(split, split_ratio=0.9):
    # load_dataset will return DatasetDict that contains all the data in the train set
    dataset_dict = load_dataset("HuggingFaceM4/the_cauldron", name="ocrvqa")
    dataset = dataset_dict["train"]
    # Comment out the following line to use the full dataset, for quick testing only use 5000 samples
    # dataset = dataset.select(range(5000))
    dataset = dataset.train_test_split(test_size=1 - split_ratio, shuffle=True, seed=42)[split]
    return dataset


class OCRVQADataCollator:
    def __init__(self, processor):
        self.processor = processor
        self.processor.tokenizer.padding_side = "right"  # during training, one always uses padding on the right

    def __call__(self, samples):
        dialogs, images = [], []
        for sample in samples:
            image_list, sample_list = sample["images"], sample["texts"]
            if len(image_list) > 1:
                raise ValueError("Only support one image per sample")
            image = image_list[0].convert("RGB")  # only use the first image
            dialog = []
            for sample_dict in sample_list:
                if not dialog:
                    # only append image to the first sentence
                    dialog += [
                        {
                            "role": "user",
                            "content": [{"type": "image"}, {"type": "text", "text": sample_dict["user"].strip()}],
                        },
                        {"role": "assistant", "content": [{"type": "text", "text": sample_dict["assistant"].strip()}]},
                    ]

                else:
                    dialog += [
                        {"role": "user", "content": [{"type": "text", "text": sample_dict["user"].strip()}]},
                        {"role": "assistant", "content": [{"type": "text", "text": sample_dict["assistant"].strip()}]},
                    ]
            dialogs.append(dialog)
            images.append([image])
        return tokenize_dialogs(dialogs, images, self.processor)


class LengthBasedBatchSampler(torch.utils.data.BatchSampler):
    def __init__(self, data_source, batch_size: int, drop_last: bool, shuffle: bool = True) -> None:
        if isinstance(next(iter(data_source)), dict):
            first_key = next(iter(next(iter(data_source)).keys()))
            self.lengths = [len(d[first_key]) for d in data_source]
        else:
            self.lengths = [len(d) for d in data_source]
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

    def __iter__(self):
        ids = np.argsort(self.lengths, kind="mergesort")
        if self.drop_last:
            ids = ids[: len(ids) // self.batch_size * self.batch_size]

        batches = [ids[i : i + self.batch_size] for i in range(0, len(ids), self.batch_size)]

        if self.shuffle:
            random.shuffle(batches)

        for b in batches:
            yield b

    def __len__(self):
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        else:
            return len(self.lengths) // self.batch_size + (len(self.lengths) % self.batch_size > 0)


class DistributedLengthBasedBatchSampler(torch.utils.data.BatchSampler):
    def __init__(
        self, data_source, batch_size: int, num_replicas: int, rank: int, shuffle: bool = True, seed: int = 0
    ) -> None:
        random.seed(seed)
        self.batch_sampler = LengthBasedBatchSampler(
            data_source, batch_size=batch_size, drop_last=True, shuffle=shuffle
        )
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self):
        max_length = len(self.batch_sampler) // self.num_replicas * self.num_replicas
        return islice(self.batch_sampler, self.rank, max_length, self.num_replicas)

    def __len__(self):
        return len(self.batch_sampler) // self.num_replicas


def get_dataloader_kwargs(train_config, dataset, dataset_processer, mode):
    kwargs = {}
    batch_size = train_config.batch_size_training if mode == "train" else train_config.val_batch_size
    if train_config.batching_strategy == "padding":
        if train_config.enable_fsdp:
            kwargs["batch_sampler"] = DistributedLengthBasedBatchSampler(
                dataset,
                batch_size=batch_size,
                rank=dist.get_rank(),
                num_replicas=dist.get_world_size(),
                shuffle=mode == "train",
            )
        else:
            kwargs["batch_sampler"] = LengthBasedBatchSampler(
                dataset, batch_size, drop_last=True, shuffle=mode == "train"
            )
        kwargs["collate_fn"] = DataCollatorForSeq2Seq(dataset_processer)
    elif train_config.batching_strategy == "packing":
        if train_config.enable_fsdp:
            kwargs["sampler"] = DistributedSampler(
                dataset,
                rank=dist.get_rank(),
                num_replicas=dist.get_world_size(),
                shuffle=mode == "train",
                drop_last=True,
            )
        kwargs["batch_size"] = batch_size
        kwargs["drop_last"] = True
        kwargs["collate_fn"] = default_data_collator
    else:
        raise ValueError(f"Unknown batching strategy: {train_config.batching_strategy}")
    return kwargs
