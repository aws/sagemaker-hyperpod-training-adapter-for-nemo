# SageMakerNeMoAdaptor

SageMakerNeMoAdaptor is a NeMo based framework solution to help users running end-to-end training workload with minimal effort.

## Building SageMakerNeMoAdaptor
To build a pip wheel from source execute ```python setup.py bdist_wheel``` from the root directory of the repository.
Once the build is complete a ```/dist``` folder will be generated and populated with the resulting ```.whl``` object.

## Installing SageMakerNeMoAdaptor
You can install SageMakerNeMoAdaptor 1 of 4 ways.

Use the command below to install only the sagemaker-nemo-adaptor library without the dependencies.

```pip install sagemaker-nemo-adaptor```

Use the command below to install the sagemaker-nemo-adaptor library along with all nemo dependencies.

```pip install sagemaker-nemo-adaptor[nemo]```

Use the command below to install the sagemaker-nemo-adaptor library along with all pytorch lightning dependencies.

```pip install sagemaker-nemo-adaptor[lightning]```

Use the command below to install the sagemaker-nemo-adaptor library along with all its dependencies.

```pip install sagemaker-nemo-adaptor[all]```

## Running jobs
```
cd scripts
sbatch -N 4 ./run_llama.sh
```

## Testing

Follow the instructions on the "Installing SageMakerNeMoAdaptor" then use the command below to install the testing dependencies:

```pip install sagemaker-nemo-adaptor[test]```

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
    "--cov=src/sagemaker_nemo_adaptor/",
    # uncomment this line to see a detailed HTML test coverage report instead of the usual summary table output to stdout.
    # "--cov-report=html",
    "tests/sagemaker_nemo_adaptor/",
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
    zipped_data: False
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
