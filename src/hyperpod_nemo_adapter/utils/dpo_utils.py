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
# Portions taken from https://github.com/huggingface/trl, Copyright 2025 HuggingFace Team

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers.data.data_collator import DataCollatorMixin


def compute_dpo_loss(
    model: torch.nn.Module,
    ref_model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    chosen_ids: torch.Tensor,
    chosen_mask: torch.Tensor,
    rejected_ids: torch.Tensor,
    rejected_mask: torch.Tensor,
    max_length: int,
    beta: float,
    label_smoothing: float,
    peft: bool,
    bf16: bool,
    *a,
    **kw,
) -> torch.Tensor:
    """
    Returns mean DPO loss over the batch.
    """
    batch = {
        "prompt_input_ids": prompt_ids,
        "prompt_attention_mask": prompt_mask,
        "chosen_input_ids": chosen_ids,
        "chosen_attention_mask": chosen_mask,
        "rejected_input_ids": rejected_ids,
        "rejected_attention_mask": rejected_mask,
    }
    padding_value = getattr(model, "pad_token_id", 0)
    concatenated_batch = concatenated_inputs(batch, padding_value=padding_value)
    model_output = concatenated_forward(model, concatenated_batch, max_length=max_length, *a, **kw)

    chosen_logps = model_output["chosen_logps"]
    rejected_logps = model_output["rejected_logps"]
    ref_chosen_logps, ref_rejected_logps = compute_ref_log_probs(
        model, concatenated_batch, ref_model=ref_model, peft=peft, bf16=bf16
    )

    losses, _, _ = dpo_loss(
        model,
        chosen_logps,
        rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
        beta=beta,
        label_smoothing=label_smoothing,
    )
    return losses.mean()


def concatenated_inputs(batch: dict, padding_value: int) -> dict:
    """
    Concatenate the `chosen` and `rejected` inputs from the batch into a single tensor for both the prompt
    and completion sequences.
    """
    output = {}

    output["prompt_input_ids"] = torch.cat([batch["prompt_input_ids"], batch["prompt_input_ids"]], dim=0)
    output["prompt_attention_mask"] = torch.cat([batch["prompt_attention_mask"], batch["prompt_attention_mask"]], dim=0)

    max_completion_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])
    output["completion_input_ids"] = torch.cat(
        (
            pad_to_length(batch["chosen_input_ids"], max_completion_length, pad_value=padding_value),
            pad_to_length(batch["rejected_input_ids"], max_completion_length, pad_value=padding_value),
        ),
    )
    output["completion_attention_mask"] = torch.cat(
        (
            pad_to_length(batch["chosen_attention_mask"], max_completion_length, pad_value=0),
            pad_to_length(batch["rejected_attention_mask"], max_completion_length, pad_value=0),
        ),
    )
    return output


def concatenated_forward(model: torch.nn.Module, batch: dict, max_length: int = 2048, *a, **kw) -> dict:
    """
    Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.
    Avoid doing two forward passes, because it's faster for FSDP.
    """
    num_examples = batch["prompt_input_ids"].shape[0] // 2
    prompt_input_ids = batch["prompt_input_ids"]
    prompt_attention_mask = batch["prompt_attention_mask"]
    completion_input_ids = batch["completion_input_ids"]
    completion_attention_mask = batch["completion_attention_mask"]

    # Concatenate the prompt and completion inputs
    input_ids = torch.cat((prompt_input_ids, completion_input_ids), dim=1)
    attention_mask = torch.cat((prompt_attention_mask, completion_attention_mask), dim=1)
    # Mask the prompt but not the completion for the loss
    loss_mask = torch.cat((torch.zeros_like(prompt_attention_mask), completion_attention_mask), dim=1)
    # Flush left to reduce the memory usage
    attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)

    # Truncate right and keep end
    input_ids = input_ids[:, -max_length:]
    attention_mask = attention_mask[:, -max_length:]
    loss_mask = loss_mask[:, -max_length:]

    # Forward pass
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    # Offset the logits by one to align with the labels
    labels = torch.roll(input_ids, shifts=-1, dims=1)
    loss_mask = torch.roll(loss_mask, shifts=-1, dims=1).bool()

    # Ensure logits and labels align (truncate logits if needed)
    if logits.shape[:2] != labels.shape[:2]:
        # for llava, the returned logits include the image tokens (placed before the text tokens)
        seq_len = labels.shape[1]
        logits = logits[:, -seq_len:]

    # Compute the log probabilities of the labels
    labels[~loss_mask] = 0  # dummy token
    per_token_logps = selective_log_softmax(logits, labels)
    per_token_logps[~loss_mask] = 0  # dummy token

    per_token_logps = torch.roll(per_token_logps, shifts=1, dims=1)
    all_logps = per_token_logps.sum(-1)  # dummy token ignored here

    output = {
        "chosen_logps": all_logps[:num_examples],
        "rejected_logps": all_logps[num_examples:],
    }
    return output


def dpo_loss(
    model,
    chosen_logps: torch.FloatTensor,
    rejected_logps: torch.FloatTensor,
    ref_chosen_logps: torch.FloatTensor,
    ref_rejected_logps: torch.FloatTensor,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> tuple:
    """
    Compute the DPO loss for a batch.
    """
    logratios = chosen_logps - rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = logratios - ref_logratios  # logits = logratios if reference free
    # Use sigmoid loss
    losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing
    # Rewards for logging (in future testing)
    chosen_rewards = beta * (chosen_logps - ref_chosen_logps).detach()
    rejected_rewards = beta * (rejected_logps - ref_rejected_logps).detach()
    return losses, chosen_rewards, rejected_rewards


def compute_ref_log_probs(
    model: torch.nn.Module,
    batch: dict,
    ref_model: torch.nn.Module = None,
    bf16: bool = False,
    peft: bool = False,
):
    """
    Computes log probabilities of the reference model for a single padded batch of a DPO specific dataset.
    """
    context_manager = torch.amp.autocast("cuda", dtype=torch.bfloat16) if bf16 and peft else nullcontext()
    with torch.inference_mode(), context_manager:
        if peft and ref_model is None:
            with model.model.disable_adapter():
                ref_model_output = concatenated_forward(model, batch)
        else:
            ref_model_output = concatenated_forward(ref_model, batch)
    return ref_model_output["chosen_logps"], ref_model_output["rejected_logps"]


# From @https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py
@dataclass
class DataCollatorForPreference(DataCollatorMixin):
    pad_token_id: int
    return_tensors: str = "pt"

    def torch_call(self, examples: list[Union[list[int], Any, dict[str, Any]]]) -> dict[str, Any]:
        # Convert to tensor
        prompt_input_ids = [torch.tensor(example["prompt_input_ids"]) for example in examples]
        prompt_attention_mask = [torch.ones_like(input_ids) for input_ids in prompt_input_ids]
        chosen_input_ids = [torch.tensor(example["chosen_input_ids"]) for example in examples]
        chosen_attention_mask = [torch.ones_like(input_ids) for input_ids in chosen_input_ids]
        rejected_input_ids = [torch.tensor(example["rejected_input_ids"]) for example in examples]
        rejected_attention_mask = [torch.ones_like(input_ids) for input_ids in rejected_input_ids]
        if "pixel_values" in examples[0]:
            pixel_values = [torch.tensor(example["pixel_values"]) for example in examples]
        if "pixel_attention_mask" in examples[0]:
            pixel_attention_mask = [torch.tensor(example["pixel_attention_mask"]) for example in examples]
        if "ref_chosen_logps" in examples[0] and "ref_rejected_logps" in examples[0]:
            ref_chosen_logps = torch.tensor([example["ref_chosen_logps"] for example in examples])
            ref_rejected_logps = torch.tensor([example["ref_rejected_logps"] for example in examples])

        # Pad
        output = {}
        output["prompt_input_ids"] = pad(prompt_input_ids, padding_value=self.pad_token_id, padding_side="left")
        output["prompt_attention_mask"] = pad(prompt_attention_mask, padding_value=0, padding_side="left")
        output["chosen_input_ids"] = pad(chosen_input_ids, padding_value=self.pad_token_id)
        output["chosen_attention_mask"] = pad(chosen_attention_mask, padding_value=0)
        output["rejected_input_ids"] = pad(rejected_input_ids, padding_value=self.pad_token_id)
        output["rejected_attention_mask"] = pad(rejected_attention_mask, padding_value=0)
        if "pixel_values" in examples[0]:
            output["pixel_values"] = pad(pixel_values, padding_value=0.0)
        if "pixel_attention_mask" in examples[0]:
            output["pixel_attention_mask"] = pad(pixel_attention_mask, padding_value=0)
        if "image_sizes" in examples[0]:
            output["image_sizes"] = torch.tensor([example["image_sizes"] for example in examples])
        if "ref_chosen_logps" in examples[0] and "ref_rejected_logps" in examples[0]:
            output["ref_chosen_logps"] = ref_chosen_logps
            output["ref_rejected_logps"] = ref_rejected_logps

        return output


# From @https://github.com/huggingface/trl/blob/main/trl/trainer/utils.py
def pad(tensors: list[torch.Tensor], padding_value: int = 0, padding_side: str = "right") -> torch.Tensor:
    # Determine the maximum shape for each dimension
    output_shape = np.max([t.shape for t in tensors], 0).tolist()

    # Create an output tensor filled with the padding value
    output = torch.full((len(tensors), *output_shape), padding_value, dtype=tensors[0].dtype, device=tensors[0].device)

    for i, t in enumerate(tensors):
        # Determine the slice for the sequence dimension
        if padding_side == "left":
            seq_slice = slice(output_shape[0] - t.shape[0], output_shape[0])
        elif padding_side == "right":
            seq_slice = slice(0, t.shape[0])
        else:
            raise ValueError("padding_side must be 'left' or 'right'")

        slices = (seq_slice,) + tuple(slice(0, s) for s in t.shape[1:])
        output[i][slices] = t

    return output


# From @https://github.com/huggingface/trl/blob/main/trl/trainer/utils.py
def pad_to_length(tensor: torch.Tensor, length: int, pad_value: Union[int, float], dim: int = -1) -> torch.Tensor:
    if tensor.size(dim) >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat(
            [
                tensor,
                pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device),
            ],
            dim=dim,
        )


# From @https://github.com/huggingface/trl/blob/main/trl/trainer/utils.py
def selective_log_softmax(logits, index):
    if logits.dtype in [torch.float32, torch.float64]:
        selected_logits = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
        # loop to reduce peak mem consumption
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
        per_token_logps = selected_logits - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficent approach
        per_token_logps = []
        for row_logits, row_labels in zip(logits, index):  # loop to reduce peak mem consumption
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)
    return per_token_logps


# From @https://github.com/huggingface/trl/blob/main/trl/trainer/utils.py
def flush_left(mask: torch.Tensor, *tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    # Create copy of mask and tensors
    mask = mask.clone()
    tensors = [t.clone() for t in tensors]

    # Shift non-zero values to the left
    for i in range(mask.size(0)):
        first_one_idx = torch.nonzero(mask[i])[0].item()
        mask[i] = torch.roll(mask[i], shifts=-first_one_idx)
        for tensor in tensors:
            tensor[i] = torch.roll(tensor[i], shifts=-first_one_idx)

    # Get the first column idx that is all zeros and remove every column after that
    empty_cols = torch.sum(mask, dim=0) == 0
    first_empty_col = torch.nonzero(empty_cols)[0].item() if empty_cols.any() else mask.size(1)
    mask = mask[:, :first_empty_col]
    for i, tensor in enumerate(tensors):
        tensors[i] = tensor[:, :first_empty_col]

    if not tensors:
        return mask
    else:
        return mask, *tensors
