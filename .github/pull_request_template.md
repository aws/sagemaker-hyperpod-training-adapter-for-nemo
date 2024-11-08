## Description

### Motivation
Explain the motivation

### Changes
* List your changes

### Testing
Explain how the changes were tested

## Merge Checklist
Put an x in the boxes that apply. If you're unsure about any of them, don't hesitate to ask. We're here to help! This is simply a reminder of what we are going to look for before merging your pull request.

### General
 - [ ] I have read the [CONTRIBUTING](https://github.com/aws/private-sagemaker-training-adapter-for-nemo-staging/blob/main/CONTRIBUTING.md) doc
 - [ ] I have run `pre-commit run --all-files` on my code. It will check for [this configuration](https://github.com/aws/private-sagemaker-training-adapter-for-nemo-staging/blob/main/.pre-commit-config.yaml).
 - [ ] I have updated any necessary documentation, including [READMEs](https://github.com/aws/private-sagemaker-training-adapter-for-nemo-staging/blob/main/README.md) and API docs (if appropriate)
 - [ ] I have verified the licenses used in the license-files artifact generated in the Python License Scan CI check. If the license workflow fails, kindly check the licenses used in the artifact.

### Tests
 - [ ] I have run `pytest` on my code and all unit tests passed.
 - [ ] I have added tests that prove my fix is effective or that my feature works (if appropriate)
 - [ ] I have ran a training job with real data (huggingface dataloader instead of synthetic data) and validated that the loss value goes down.

By submitting this pull request, I confirm that my contribution is made under the terms of the Apache 2.0 license.
