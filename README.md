# Amazon SageMaker HyperPod training adapter for NeMo

Amazon SageMaker HyperPod training adapter for NeMo is a generative AI framework built on top of [NVIDIA's NeMo](https://github.com/NVIDIA/NeMo)
framework and [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning).

This adapter enables you to leverage existing resources for common language
model pre-training tasks, supporting popular models such as LLaMA, Mixtral, and
Mistral. Additionally, the framework incorporates standard fine-tuning techniques,
including Supervised Fine-Tuning (SFT) and Parameter-Efficient Fine-Tuning (PEFT)
using LoRA or QLoRA.

Amazon SageMaker HyperPod training adapter for NeMo makes it
it easier to work with state-of-the-art large language
models. For more detailed information on distributed training capabilities, please
refer to our documentation: [HyperPod recipes](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-recipes.html).

## Building Amazon SageMaker HyperPod training adapter for NeMo

If you want to create an installable package (wheel) for the Amazon SageMaker HyperPod training adapter for NeMo
from source code, you will need to run:

```bash
python setup.py bdist_wheel
```

from the root directory of the repository. Once the build is complete a `/dist`
folder will be generated and populated with the resulting `.whl` object.

## Installing Amazon SageMaker HyperPod training adapter for NeMo

### Pip

The Amazon SageMaker HyperPod training adapter for NeMo can be installed using the Python package installer (pip)
by running the command

```bash
pip install hyperpod-nemo-adapter[all]
```

Please note that this library requires Python version 3.11 or later to function
correctly. Alternatively, you have the option to install the library from its
source code.

## Amazon SageMaker HyperPod recipes

Amazon SageMaker SageMaker HyperPod recipes offers launch scripts built on the Amazon SageMaker HyperPod training adapter for NeMo.
You can use this launcher on Amazon SageMaker HyperPod (with Slurm or Amazon EKS orchestrator), or Amazon SageMaker training jobs.
The recipes also include templates for pre-training or fine-tuning models. For more information,
please refer to [Amazon SageMaker HyperPod recipes](https://github.com/aws/sagemaker-hyperpod-recipes).

## Testing

Follow the instructions on the "Installing Amazon SageMaker HyperPod training adapter for NeMo" then use the command below to install the testing dependencies:

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
2. Make sure your ```max_context_width``` aligns with the sequence length of your tokenized dataset.

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
