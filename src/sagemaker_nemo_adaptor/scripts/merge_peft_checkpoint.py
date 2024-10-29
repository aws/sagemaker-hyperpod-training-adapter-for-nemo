import argparse
import os

from peft import PeftModel
from transformers import AutoModelForCausalLM


def run(args):
    print("Loading the HF model...")
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

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
