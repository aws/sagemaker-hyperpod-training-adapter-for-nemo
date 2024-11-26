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

import transformers
from omegaconf import OmegaConf
from packaging import version as pversion
from transformers import LlamaConfig

from hyperpod_nemo_adapter.collections.model import SageMakerNLPBaseModel
from hyperpod_nemo_adapter.constants import CONFIG_MAPPING_HF_TO_RECIPE_ALIASES
from hyperpod_nemo_adapter.utils.config_utils import get_hf_config_from_name_or_path
from hyperpod_nemo_adapter.utils.general_utils import can_use_multimodal
from hyperpod_nemo_adapter.utils.log_utils import Logger

if can_use_multimodal():
    from transformers import MllamaConfig

_logger = Logger().get_logger()


class SageMakerLlamaModel(SageMakerNLPBaseModel):
    """
    Lightning Model class for Llama
    """

    predefined_model = True

    def set_config_mapping_hf_to_recipe_aliases(self):
        config_map = CONFIG_MAPPING_HF_TO_RECIPE_ALIASES
        if OmegaConf.select(self._cfg, "rope_scaling.rope_type") == "llama3":
            if pversion.parse(transformers.__version__) < pversion.parse("4.44.2"):
                _logger.warning(
                    f"Rope scaling type 'llama3' is only supported for transformers >= 4.44.2, the current version is {pversion.parse(transformers.__version__)}"
                )
            else:
                config_map["rope_scaling"] = ["rope_scaling"]
        self._config_mapping_hf_to_recipe_aliases = config_map

    def get_model_config(self):
        """
        Get model config for Llama
        """
        configurable_dict = self._get_model_configurable_dict()
        multi_modal_enabled = self._cfg.get("multi_modal", None)
        if self._cfg.get("hf_model_name_or_path", None) is not None:
            model_config = get_hf_config_from_name_or_path(self._cfg)
            assert isinstance(model_config, LlamaConfig) or (
                multi_modal_enabled and isinstance(model_config, MllamaConfig)
            ), f"model_type is set to llama but hf_model_name_or_path is not the same model, getting {type(model_config)}"
            # Update the config based on user's input
            model_config.update(configurable_dict)
        else:
            model_config = LlamaConfig(
                **configurable_dict,
                hidden_act="silu",
                use_cache=False,
            )
        return model_config
