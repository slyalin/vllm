# ruff: noqa: SIM117
import copy
import glob
import os
from functools import partial
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Tuple, Type
from huggingface_hub import HfApi

import numpy as np
import huggingface_hub
import torch
from torch import nn

from vllm.config import (DeviceConfig, ModelConfig)
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import SamplerOutput
from vllm.utils import is_openvino_optimum_intel, STR_DTYPE_TO_TORCH_DTYPE
from vllm.attention.backends.abstract import AttentionMetadata
from vllm.attention.backends.torch_sdpa import TorchSDPAMetadata

import openvino as ov

logger = init_logger(__name__)


def _flattenize_inputs(inputs):
    """
    Helper function for making nested inputs flattens
    """
    flatten_inputs = []
    for input_data in inputs:
        if input_data is None:
            continue
        if isinstance(input_data, (list, tuple)):
            flatten_inputs.extend(_flattenize_inputs(input_data))
        elif isinstance(input_data, dict):
            flatten_inputs.extend(_flattenize_inputs(list(input_data.values())))
        else:
            flatten_inputs.append(input_data)
    return flatten_inputs


def _modify_cache_parameters(
        model: ov.Model,
        kv_cache_dtype: ov.Type,
        is_cpu: bool
):
    # Apply hardware dependent modifications to KV tensors
    for parameter in model.get_parameters():
        input = parameter.get_output_tensor(0)
        input_names = input.get_names()
        if len(input_names) != 1:
            continue
        input_name = next(iter(input_names))
        shape = parameter.get_partial_shape()
        x_size = 1  # use real block size if available, just a placeholder to provide the expected rank
        num_blocks = ov.Dimension()
        block_size = ov.Dimension()
        # TODO: Negotiate required layout with plugins (CPU is ~OK, GPU is TBD), pass more parameters to this function to set more static dimensions
        if input_name.startswith('key_cache.'):
            cpu_shape = [num_blocks, shape[1], block_size, shape[2]]
            gpu_shape = [num_blocks, shape[1], shape[2].get_length() // x_size if shape[2].is_static else ov.Dimension(), block_size, x_size]
        elif input_name.startswith('value_cache.'):
            cpu_shape = [num_blocks, shape[1], block_size, shape[2]]
            gpu_shape = [num_blocks, shape[1], shape[2], block_size]
        else:
            continue
        parameter.set_partial_shape(ov.PartialShape(cpu_shape if is_cpu else gpu_shape))
        parameter.set_element_type(kv_cache_dtype)
    model.validate_nodes_and_infer_types()


def _require_model_export(model_id, revision=None, subfolder=None):
    # Stored IR may not be suitable for vLLM purposes (too old, not stateful, not compressed etc.)
    # This is an option to override IR usage logic and alway do model conversion.
    if os.environ.get('VLLM_OPENVINO_OPTIMUM_FORCE_CONVERSION', '0') == '1':
        return True
    model_dir = Path(model_id)
    if subfolder is not None:
        model_dir = model_dir / subfolder
    if model_dir.is_dir():
        return not (model_dir / "openvino_model.xml").exists() or not (model_dir / "openvino_model.bin").exists()

    hf_api =  HfApi()
    try:
        model_info = hf_api.model_info(model_id, revision=revision or "main")
        normalized_subfolder = None if subfolder is None else Path(subfolder).as_posix()
        model_files = [file.rfilename for file in model_info.siblings if normalized_subfolder is None or file.rfilename.startswith(normalized_subfolder)]
        ov_model_path = "openvino_model.xml" if normalized_subfolder is None else f"{normalized_subfolder}/openvino_model.xml"
        return not ov_model_path in model_files or not ov_model_path.replace(".xml", ".bin") in model_files
    except Exception:
        return True


class OpenVINOCasualLM(nn.Module):

    def __init__(
        self,
        model_config: ModelConfig,
        device_config: DeviceConfig,
        kv_cache_dtype: ov.Type
    ) -> None:
        super().__init__()
        self.logits_processor = LogitsProcessor(model_config.hf_config.vocab_size,
                                                logits_as_input=True)
        self.sampler = Sampler()

        if not is_openvino_optimum_intel():
            raise ValueError("Optimum Intel is not installed. Please, install it via 'pip install optimum-intel[openvino]'")

        export = _require_model_export(model_config.model)
        if export:
            logger.warning(f'[ INFO ] Provided model id {model_config.model} does not contain OpenVINO IR, the model will be converted to IR with default options. '
                            'If you need to use specific options for model conversion, use optimum-cli export openvino with desired options.')
        else:
            logger.warning(f'[ INFO ] OpenVINO IR is avaialble for provided model id {model_config.model}. '
                            'This IR will be used for inference as-is, all possible options that may affect model conversion are ignored.')

        load_in_8bit = False if os.environ.get('VLLM_OPENVINO_ENABLE_QUANTIZED_WEIGHTS', '1') == '0' else None

        from optimum.intel import OVModelForCausalLM
        pt_model = OVModelForCausalLM.from_pretrained(
            model_config.model,
            export=export,
            compile=False,
            load_in_8bit=load_in_8bit,
            trust_remote_code=model_config.trust_remote_code
        )

        from openvino._offline_transformations import paged_attention_transformation
        paged_attention_transformation(pt_model.model)
        _modify_cache_parameters(pt_model.model, kv_cache_dtype, device_config.device.type == "cpu")

        # For deployment outside vLLM
        model_file_name = os.environ.get('VLLM_OPENVINO_EXPORTED_IR_NAME', '')
        if model_file_name:
            ov.save_model(pt_model.model, model_file_name)

        core = ov.Core()
        ov_compiled = core.compile_model(pt_model.model, "CPU")
        self.ov_request = ov_compiled.create_infer_request()


    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[Tuple[ov.Tensor, ov.Tensor]],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        print('OpenVINO Model inference')
        flatten_kv_cache = _flattenize_inputs(kv_caches)

        inputs = [
            input_ids,
            positions,
            *flatten_kv_cache,
            attn_metadata.is_prompt,
            attn_metadata.slot_mapping
        ]

        if attn_metadata.max_seq_len is not None:
            # available from the second iteration
            inputs.append(attn_metadata.max_seq_len)
            inputs.append(attn_metadata.seq_lens)
            inputs.append(attn_metadata.block_tables)
        else:
            print(input_ids.shape)
            inputs.append(np.array(input_ids.shape[1], dtype=np.int64))   # for optimum-based models this parameter can be used even on the first iteration

        self.ov_request.start_async(inputs, share_inputs=True)
        self.ov_request.wait()
        return torch.from_numpy(self.ov_request.get_tensor("logits").data)


    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(None, hidden_states, sampling_metadata)
        return logits


    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens


def get_model(model_config: ModelConfig,
              device_config: DeviceConfig,
              kv_cache_dtype: ov.Type,
              **kwargs) -> torch.nn.Module:
    lora_config = kwargs.get("lora_config", None)
    if lora_config:
        raise ValueError(
            f"OpenVINO modeling does not support LoRA, "
            "but LoRA is enabled. Support for this model may "
            "be added in the future. If this is important to you, "
            "please open an issue on github.")

    return OpenVINOCasualLM(model_config, device_config, kv_cache_dtype)
