import torch
from transformers.models.mllama import MllamaVisionModel

is_patched = False

orginal_dtype = MllamaVisionModel.dtype


def unapply_patch():
    global is_patched
    MllamaVisionModel.dtype = orginal_dtype
    is_patched = False


def apply_patch(dtype=torch.bfloat16):
    global is_patched
    MllamaVisionModel.dtype = dtype
    is_patched = True
