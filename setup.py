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
# Portions taken from https://github.com/NVIDIA/NeMo, Copyright Nvidia Corporation

import os

from setuptools import PEP420PackageFinder, setup

# Source the package version as __version__
exec(open("src/hyperpod_nemo_adapter/version/__version__.py").read())
package_name = "hyperpod-nemo-adapter"
package_keywords = "aws sagemaker"


# This implementation of a requirements.txt loader is base off of the linked function from NVIDIA NeMo setup.py: https://github.com/NVIDIA/NeMo/blob/5a9000fbb858edfd5d156adf5453ea2b8342e4d2/setup.py#L71C24-L71C45
def parse_requirements(filename, folder="requirements"):
    with open(os.path.join(folder, filename), encoding="utf-8") as f:
        content = f.readlines()
    # Added the split at # character to ignore any commented text in a given line of the requirements file
    return [x.split("#")[0].strip() for x in content]


def extra_requirements():
    return {
        "nemo": parse_requirements("requirements_nemo.txt"),
        "lightning": parse_requirements("requirements_lightning.txt"),
        "test": parse_requirements("requirements_test.txt"),
        "profiling": parse_requirements("requirements_profiling.txt"),
        "all": parse_requirements("requirements_nemo.txt")
        + parse_requirements("requirements_lightning.txt")
        + parse_requirements("requirements_test.txt")
        + parse_requirements("requirements_profiling.txt"),
    }


setup(
    name=package_name,
    version=__version__,
    license="Apache 2.0",
    license_files=("LICENSE",),
    keywords=package_keywords,
    include_package_data=True,
    packages=PEP420PackageFinder.find(where="src"),
    package_dir={"": "src"},
    package_data={"hyperpod_nemo_adapter": ["conf/*.yaml"]},
    install_requires=parse_requirements("requirements.txt"),
    extras_require=extra_requirements(),
    python_requires=">= 3.10",
    entry_points={
        "console_scripts": [
            "merge-peft-checkpoint = hyperpod_nemo_adapter.scripts.merge_peft_checkpoint:main",
        ],
    },
)
