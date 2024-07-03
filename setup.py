from setuptools import setup, PEP420PackageFinder
import os

# Source the package version as __version__
exec(open("src/sagemaker_nemo_adaptor/version/__version__.py").read())
package_name = "sagemaker-nemo-adaptor"
package_keywords = "aws sagemaker"


# This implementation of a requirements.txt loader is base off of the linked function from NVIDIA NeMo setup.py: https://github.com/NVIDIA/NeMo/blob/5a9000fbb858edfd5d156adf5453ea2b8342e4d2/setup.py#L71C24-L71C45
def parse_requirements(filename, folder="requirements"):
    with open(os.path.join(folder, filename), encoding='utf-8') as f:
        content = f.readlines()
    # Added the split at # character to ignore any commented text in a given line of the requirements file
    return [x.split("#")[0].strip() for x in content]

def extra_requirements():
   return {
       'nemo': parse_requirements("requirements_nemo.txt"),
       'lightning': parse_requirements("requirements_lightning.txt"),
       'all': parse_requirements("requirements_nemo.txt") + parse_requirements("requirements_lightning.txt"),
   }

setup(
    name=package_name,
    version=__version__,
    keywords=package_keywords,
    packages=PEP420PackageFinder.find(where="src"),
    package_dir={"": "src"},
    install_requires=parse_requirements('requirements.txt'),
    extras_require=extra_requirements(),
    python_requires=">=3.9",
)