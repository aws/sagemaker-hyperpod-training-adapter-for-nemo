# HyperPodNemoAdapter

HyperPodNemoAdapter is a generative AI framework built on top of [NVIDIA's NeMo](https://github.com/NVIDIA/NeMo)
framework and [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning)
for Large Language Models (LLMs).

This solution enables you to leverage existing resources for common language
model pre-training tasks, supporting popular models such as LLaMA, Mixtral, and
Mistral. Additionally, our framework incorporates standard fine-tuning techniques,
including Supervised Fine-Tuning (SFT) and Parameter-Efficient Fine-Tuning (PEFT)
using LoRA or QLoRA.

HyperPodNemoAdapter streamlines the development and deployment of LLMs, making
it easier for researchers and developers to work with state-of-the-art language
models. For more detailed information on distributed training capabilities, please
refer to our comprehensive documentation: "HyperPod recipes for distributed training."

## Building HyperPodNemoAdapter

If you want to create an installable package (wheel) for the HyperPodNemoAdapter
library from its source code, you need to execute the command:

```bash
python setup.py bdist_wheel
```

from the root directory of the repository. Once the build is complete a `/dist`
folder will be generated and populated with the resulting `.whl` object.

## Installing HyperPodNemoAdapter

### Pip

HyperPodNemoAdapter can be installed using the Python package installer (pip)
by running the command

```bash
pip install hyperpod-nemo-adapter[all]
```

Please note that this library requires Python version 3.11 or later to function
correctly. Alternatively, you have the option to install the library from its
source code

## HyperPod recipes

SageMaker HyperPod Recipes offers a launcher for running training scripts built
on HyperPodNemoAdapter. You can use this launcher on various cluster types,
including Slurm, Kubernetes, or SageMaker Training Jobs. The recipes also include
many useful templates for pre-training or fine-tuning models. For more information,
please refer to the [SageMaker HyperPod Recipes](https://github.com/aws/sagemaker-hyperpod-recipes).

## Testing

Follow the instructions on the "Installing HyperPodNemoAdapter" then use the command below to install the testing dependencies:

```bash
pip install hyperpod-nemo-adapter[test]
```

### Unit Tests
To run the unit tests navigate to the root directory and use the command
```pytest``` plus any desired flags.

The `myproject.toml` file defines additional options that are always appended to the `pytest` command:
```
[tool.pytest.ini_options]
...
addopts = [
    "--cache-clear",
    "--quiet",
    "--durations=0",
    "--cov=src/hyperpod_nemo_adapter/",
    # uncomment this line to see a detailed HTML test coverage report instead of the usual summary table output to stdout.
    # "--cov-report=html",
    "tests/hyperpod_nemo_adapter/",
]
```

### Non-synthetic Tests
To run a non-synthetic test change ```use_synthetic_data``` in your ```model-config.yaml``` file from ```False``` to ```True```. Make sure ```dataset_type: hf``` and that ```train_dir``` and ```val_dir``` point to valid datasets in ```/fsx/datasets```
Example (c4 dataset pre-tokenized with llama3 tokenizer):
```
data:
    train_dir: ["/fsx/datasets/c4/en/hf-tokenized/llama3/train"]
    val_dir: ["/fsx/datasets/c4/en/hf-tokenized/llama3/val"]
    dataset_type: hf
    use_synthetic_data: True
```
Additional considerations:
1. Make sure you have a ```vocab_size``` that fits your model.
2. Make sure your ```max_context_width``` aligns with the sequence length that the dataset was tokenized at.

## Contributing

### Formatting code

To format the code, run following command before committing your changes:
```
pip install pre-commit
pre-commit run --all-files
```

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the [Apache-2.0 License](LICENSE).
