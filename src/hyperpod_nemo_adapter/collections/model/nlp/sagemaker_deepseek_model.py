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

from transformers import LlamaConfig, Qwen2Config

from hyperpod_nemo_adapter.utils.config_utils import get_hf_config_from_name_or_path
from hyperpod_nemo_adapter.utils.log_utils import Logger

from .sagemaker_llama_model import SageMakerLlamaModel
from .sagemaker_qwen_model import SageMakerQwenModel

_logger = Logger().get_logger()

# TODO add a model class for the first-party DeepSeek models (DeepSeek-R1, DeepSeek-V3...)


class SageMakerDeepSeekDistilledLlamaModel(SageMakerLlamaModel):
    """
    Lightning Model class for DeepSeek-r1 Distilled Llama3
    """

    predefined_model = True

    def get_model_config(self):
        """
        Get model config for Distilled Llama
        """
        configurable_dict = self._get_model_configurable_dict()
        if self._cfg.get("hf_model_name_or_path", None) is None:
            raise Exception(
                f"expected hf_model_name_or_path to point to a llama3 model from https://huggingface.co/deepseek-ai/ but is {self._cfg.get('hf_model_name_or_path', None)}"
            )
        model_config = get_hf_config_from_name_or_path(self._cfg)
        assert isinstance(
            model_config, LlamaConfig
        ), f"model_type is set to llama3 but hf_model_name_or_path is not the same model, getting {type(model_config)}"
        # Update the config based on user's input
        model_config.update(configurable_dict)

        return model_config


class SageMakerDeepSeekDistilledQwenModel(SageMakerQwenModel):
    """
    Lightning Model class for DeepSeek-r1 Distilled Qwen2
    """

    predefined_model = False

    def get_model_config(self):
        """
        Get model config for Distilled Qwen
        """
        configurable_dict = self._get_model_configurable_dict()
        if self._cfg.get("hf_model_name_or_path", None) is None:
            raise Exception(
                f"expected hf_model_name_or_path to point to a qwen2 model from https://huggingface.co/deepseek-ai/ but is {self._cfg.get('hf_model_name_or_path', None)}"
            )
        model_config = get_hf_config_from_name_or_path(self._cfg)
        assert isinstance(
            model_config, Qwen2Config
        ), f"model_type is set to qwen2 but hf_model_name_or_path is not the same model, getting {type(model_config)}"
        # Update the config based on user's input
        model_config.update(configurable_dict)

        return model_config
