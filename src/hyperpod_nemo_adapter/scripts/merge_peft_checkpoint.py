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

import argparse
import os

from peft import PeftModel
from transformers import AutoModelForCausalLM

from hyperpod_nemo_adapter.collections.model.nlp.custom_models.configuration_deepseek import (
    DeepseekV3Config,
)
from hyperpod_nemo_adapter.collections.model.nlp.custom_models.modeling_deepseek import (
    DeepseekV3ForCausalLM,
)


def run(args):
    print("Loading the HF model...")

    if args.deepseek_v3:
        model_config = DeepseekV3Config.from_pretrained(
            args.hf_model_name_or_path, token=args.hf_access_token, trust_remote_code=True
        )
        if hasattr(model_config, "quantization_config"):
            delattr(model_config, "quantization_config")
        model = DeepseekV3ForCausalLM.from_pretrained(
            args.hf_model_name_or_path,
            torch_dtype="auto",
            device_map="auto",
            token=args.hf_access_token,
            config=model_config,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.hf_model_name_or_path,
            torch_dtype="auto",
            device_map="auto",
            token=args.hf_access_token,
        )

    print("Loading the PEFT adapter checkpoint...")
    model = PeftModel.from_pretrained(model, args.peft_adapter_checkpoint_path)

    print("Merging the PEFT adapter with the base model...")
    model = model.merge_and_unload(progressbar=True)

    print(f"Saving the merged model to {args.output_model_path}...")
    if not os.path.exists(args.output_model_path):
        os.makedirs(args.output_model_path)
    model.save_pretrained(args.output_model_path)
    print("Model saved successfully.")


def main():
    parser = argparse.ArgumentParser(
        description="Script for merging a Hugging Face model with a PEFT adapter checkpoint"
    )

    parser.add_argument(
        "--hf_model_name_or_path",
        type=str,
        required=True,
        help="The Hugging Face model name or path to load the model from.",
    )
    parser.add_argument(
        "--peft_adapter_checkpoint_path", type=str, required=True, help="Path to the PEFT adapter checkpoint."
    )
    parser.add_argument(
        "--output_model_path", type=str, required=True, help="Path where the merged model will be saved."
    )
    parser.add_argument(
        "--hf_access_token", type=str, default=None, help="Optional Hugging Face access token for authentication."
    )
    parser.add_argument("--deepseek_v3", type=bool, default=False, help="Whether the model is DeepSeek V3 model.")

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
