# ruff: noqa: SIM117
import copy
import glob
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Tuple, Type

import huggingface_hub
import torch
from torch import nn

from vllm.config import (DeviceConfig, ModelConfig)
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor

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


def ov_wrapper(self, *args, **kwargs) -> torch.Tensor:
    attn_metadata = kwargs['attn_metadata']
    flatten_kv_cache = _flattenize_inputs(kwargs['kv_caches'])

    inputs = [
        kwargs['input_ids'],
        kwargs['positions'],
        *flatten_kv_cache,
        attn_metadata.is_prompt,
        attn_metadata.slot_mapping
    ]

    if attn_metadata.max_context_len is not None:
        # available from the second iteration
        inputs.append(attn_metadata.max_context_len)
        inputs.append(attn_metadata.context_lens)
        inputs.append(attn_metadata.block_tables)
    else:
        inputs.append(np.array(kwargs['input_ids'].shape[1], dtype=np.int64))   # for optimum-based models this parameter can be used even on the first iteration

    self._ov_request.start_async(inputs, share_inputs=True)
    self._ov_request.wait()
    return torch.from_numpy(self._ov_request.get_tensor("logits").data)


def arguments_as_outputs(arguments):
    outputs = []
    for argument in arguments:
        if issubclass(type(argument), ov.runtime.Output):
            outputs.append(argument)
        else:
            outputs.extend(argument.outputs())
    return outputs


def patch_stateful_model(
    model: ov.Model,
    kv_cache_dtype: Type):
    logger.warning('Transforming OPTIMUM-INTEL model to vLLM compatible form with PagedAttention')
    from openvino.runtime.passes import Manager, MatcherPass, WrapType, Matcher, AnyInput, Or
    from openvino.runtime import opset13
    from openvino.runtime.utils import replace_node

    max_context_len = opset13.parameter(shape=[], dtype=np.int64, name='max_context_len')  # max_context_len
    model_remaining_params = [
        opset13.parameter(shape=[], dtype=bool, name='is_prompt'),  # is_prompt
        opset13.parameter(shape=[-1, -1], dtype=np.int64, name='slot_mapping'),  # slot mapping
        max_context_len,
        opset13.parameter(shape=[-1], dtype=np.int64, name='context_lens'),  # context_lens
        opset13.parameter(shape=[-1, -1], dtype=np.int32, name='block_tables'),  # block_tables
    ]
    for parameter in model_remaining_params:
        parameter.get_output_tensor(0).set_names({parameter.get_friendly_name()})
    sliding_window = opset13.constant(np.array(0, np.int32))  # sliding_window

    current_seq_len = opset13.gather(opset13.shape_of(model.input('input_ids')), opset13.constant(1), opset13.constant(0))
    current_seq_len.set_friendly_name('my_current_seq_len')
    prev_max_seq_len = max_context_len - current_seq_len

    def has_parameter(model, name):
        return name in sum([list(t.get_names()) for t in model.inputs], [])

    kv_parameters = []
    assignes_to_remove = []  # not really used
    parameters_to_remove = []
    results_to_remove = []  # used, but cannot really track all Results in stateless model
    if not has_parameter(model, 'position_ids'):
        position_ids = opset13.parameter(shape=[-1, -1], dtype=np.int64, name="position_ids")
        position_ids.get_output_tensor(0).set_names({position_ids.get_friendly_name()})
        model.add_parameters([position_ids])
        logger.warning('CREATED A NEW position_ids PARAMETER')
    position_ids = model.input('position_ids')

    kv_transpose_order = opset13.constant([0, 2, 1, 3])

    class StateManagementPattern(MatcherPass):
        def __init__(self):
            MatcherPass.__init__(self)
            self.model_changed = False

            k_past_var = WrapType("opset13.ReadValue", AnyInput())
            k_past_par = WrapType("opset13.Parameter")
            k_past = Or([WrapType("opset13.Gather", [k_past_var, AnyInput(), AnyInput()]), k_past_par])
            k_past = Or([k_past, WrapType("opset13.Transpose", [k_past, AnyInput()])])  # Transpose is used when kv-cache is stored in a not usual layout, example: bloom
            k_current = AnyInput()
            k_current2 = AnyInput()
            k_current_reshaped = WrapType("opset13.Reshape", [k_current2, AnyInput()])
            k_concat = WrapType("opset13.Concat", [k_past, Or([k_current_reshaped, k_current])])

            def kv_shaping(kv_concat):
                interim = WrapType("opset13.StridedSlice", [kv_concat, *[AnyInput() for _ in range(3)]])
                interim = WrapType("opset13.StridedSlice", [interim, *[AnyInput() for _ in range(3)]])
                unsqueeze = WrapType("opset13.Unsqueeze", [Or([kv_concat, interim]), AnyInput()])
                interim = WrapType("opset13.StridedSlice", [unsqueeze, *[AnyInput() for _ in range(3)]])
                interim = WrapType("opset13.StridedSlice", [interim, *[AnyInput() for _ in range(3)]])
                interim = WrapType("opset13.Broadcast", [Or([unsqueeze, interim]), AnyInput()])
                interim = WrapType("opset13.Reshape", [interim, AnyInput()])
                return interim

            v_past_var = WrapType("opset13.ReadValue", AnyInput())
            v_past_par = WrapType("opset13.Parameter")
            v_past = Or([WrapType("opset13.Gather", [v_past_var, AnyInput(), AnyInput()]), v_past_par])
            v_past = Or([v_past, WrapType("opset13.Transpose", [v_past, AnyInput()])])
            v_current = AnyInput()
            v_current2 = AnyInput()
            v_current_reshaped = WrapType("opset13.Reshape", [v_current2, AnyInput()])
            v_concat = WrapType("opset13.Concat", [v_past, Or([v_current_reshaped, v_current])])

            k_shaped = kv_shaping(k_concat)
            v_shaped = kv_shaping(v_concat)

            k_simply_shaped = WrapType("opset13.Reshape", [k_concat, AnyInput()])
            v_simply_shaped = WrapType("opset13.Reshape", [v_concat, AnyInput()])

            k_order = AnyInput()
            v_order = AnyInput()

            # KV-path may already have Transposes that will be rewritten based on PA KV inputs required layout
            k_shaped_transposed = WrapType("opset13.Transpose", [Or([k_concat, k_shaped]), k_order])
            v_shaped_transposed = WrapType("opset13.Transpose", [Or([v_concat, v_shaped]), v_order])

            # Optional pattern to capture alibi slopes (based on pattern from bloom)
            alibi = AnyInput()
            sdpa_mask = WrapType("opset13.Multiply", [AnyInput(), alibi])  # apply input position_ids
            sdpa_mask = WrapType("opset13.Reshape", [sdpa_mask, AnyInput()])
            sdpa_mask = WrapType("opset13.Reshape", [sdpa_mask, AnyInput()])
            sdpa_mask = WrapType("opset13.Select", [AnyInput(), AnyInput(), sdpa_mask])

            q = AnyInput()
            sdpa = WrapType("opset13.ScaledDotProductAttention", [
                q,
                Or([k_concat, k_shaped, k_shaped_transposed, k_simply_shaped]),
                Or([v_concat, v_shaped, v_shaped_transposed, v_simply_shaped]),
                Or([sdpa_mask, AnyInput()])
            ])

            def callback(m: Matcher) -> bool:
                assert sdpa in m.get_pattern_value_map()
                mapping = m.get_pattern_value_map()
                assert sdpa in mapping
                real_q = mapping[q]

                # takes option that has 4D instead of fine-grained Reshape analysis
                # it avoids complication in the pattern, but we don't really have many options
                def take_4d(option1, option2, option3):
                    if option1 in mapping and mapping[option1].get_partial_shape().rank.get_length() == 4:
                        return mapping[option1]
                    elif mapping[option2].get_partial_shape().rank.get_length() == 4:
                        return mapping[option2]
                    else:
                        return mapping[option3]

                real_k = take_4d(k_current, k_current_reshaped, k_current2)
                real_v = take_4d(v_current, v_current_reshaped, v_current2)
                k_parameter = opset13.parameter(shape=[-1, -1, -1, -1], dtype=kv_cache_dtype)
                v_parameter = opset13.parameter(shape=[-1, -1, -1, -1], dtype=kv_cache_dtype)
                kv_parameters.append(k_parameter)
                kv_parameters.append(v_parameter)
                q_transpose = opset13.transpose(real_q, kv_transpose_order)
                q_reshape = opset13.reshape(q_transpose, opset13.constant([0, 0, -1]), True)

                k_tranpose_order = kv_transpose_order
                if k_order in mapping:  # reapply transpose found in the graph by manipulating of indices of our Transpose
                    k_tranpose_order = opset13.gather(mapping[k_order], kv_transpose_order, opset13.constant(0))
                k_transpose = opset13.transpose(real_k, k_tranpose_order)
                k_reshape = opset13.reshape(k_transpose, opset13.constant([0, 0, -1]), True)

                v_tranpose_order = kv_transpose_order
                if v_order in mapping:  # reapply transpose found in the graph by manipulating of indices of our Transpose
                    v_tranpose_order = opset13.gather(mapping[v_order], kv_transpose_order, opset13.constant(0))
                v_transpose = opset13.transpose(real_v, v_tranpose_order)
                v_reshape = opset13.reshape(v_transpose, opset13.constant([0, 0, -1]), True)

                # TODO: Detect whether SDPA in the model graph has `scale` argument set and use it instead of the computed scale below
                # Most likely `scale` will always be a constant in real inference, but dynamic dimension propagation may not always derive it as a constant
                # That's why a sub-graph computing `scale` is built instead of just a constant node.
                hidden_shape = opset13.shape_of(real_q)
                hidden_dim = opset13.gather(hidden_shape, opset13.constant(-1), opset13.constant(0))
                scale = opset13.constant(1.0, dtype=ov.Type.f32)/opset13.sqrt(opset13.convert(hidden_dim, destination_type=ov.Type.f32))

                if alibi in mapping:
                    logger.warning('alibi slopes are applied')
                    alibi_slopes = opset13.reshape(mapping[alibi], opset13.constant([-1]), special_zero=False)
                    if alibi_slopes.get_element_type() != ov.Type.f32:
                        alibi_slopes = opset13.convert(alibi_slopes, destination_type=ov.Type.f32)  #todo
                else:
                    alibi_slopes = opset13.constant(np.array([], np.float32))

                paged_attention = ov.runtime.op._PagedAttentionExtension(arguments_as_outputs([
                    q_reshape,
                    k_reshape,
                    v_reshape,
                    k_parameter,
                    v_parameter,
                    *model_remaining_params,
                    scale,
                    alibi_slopes,
                    sliding_window
                ]))
                pa_shape = opset13.concat([
                        opset13.constant([0]),
                        opset13.constant([0]),
                        opset13.constant([-1]),
                        opset13.unsqueeze(hidden_dim, opset13.constant(0))
                    ], axis=0)
                pa_reshape = opset13.reshape(paged_attention, pa_shape, True)
                pa_transpose = opset13.transpose(pa_reshape, kv_transpose_order)

                if v_past_par in mapping:
                    parameters_to_remove.append(mapping[v_past_par].get_node())

                if k_past_par in mapping:
                    parameters_to_remove.append(mapping[k_past_par].get_node())

                def add_assign_consumers(output):
                    for consumer in output.get_target_inputs():
                        consumer_node = consumer.get_node()
                        consumer_type = consumer_node.get_type_info().name
                        if consumer_type == 'Assign':  # stateful model
                            assignes_to_remove.append(consumer_node)
                        elif consumer_type == 'Result':  # stateless model
                            results_to_remove.append(consumer_node)

                add_assign_consumers(mapping[k_concat])
                add_assign_consumers(mapping[v_concat])

                replace_node(m.get_match_root(), pa_transpose)
                print('INSERTED PageAttentionExtension')
                return True

            self.register_matcher(Matcher(sdpa, "StateManagementPattern"), callback)

    class PrevSequenceLengthPattern(MatcherPass):
        def __init__(self):
            MatcherPass.__init__(self)
            self.model_changed = False

            kv_past = WrapType("opset13.ReadValue", AnyInput())
            kv_gather = WrapType("opset13.Gather", [kv_past, AnyInput(), AnyInput()])
            kv_shape = WrapType("opset13.ShapeOf", [kv_gather])
            seq = WrapType("opset13.Gather", [kv_shape, AnyInput(), AnyInput()])

            def callback(m: Matcher) -> bool:
                # TODO: Check that seq has axis that really takes sequence len but not any other dimension -- use symbolics or look at the constant input
                gather = m.get_match_root()
                target_type = gather.get_output_element_type(0)
                if prev_max_seq_len.get_output_element_type(0) != target_type:
                    print(f'Converting {prev_max_seq_len.get_output_element_type(0)} of max_context_len to {target_type}')
                    replacement = opset13.convert(prev_max_seq_len, target_type)
                else:
                    replacement = prev_max_seq_len
                replace_node(gather, replacement)
                print("DETECTED PATTERN PrevSequenceLengthPattern, CONNECTED TO A DEDICATED PARAMETER")
                return True

            self.register_matcher(Matcher(seq, "PrevSequenceLengthPattern"), callback)

    class TotalSequenceLengthPattern(MatcherPass):
        def __init__(self):
            MatcherPass.__init__(self)
            self.model_changed = False

            kv_past = WrapType("opset13.ReadValue", AnyInput())
            kv_gather = WrapType("opset13.Gather", [kv_past, AnyInput(), AnyInput()])
            kv_current = AnyInput()
            kv_concat = WrapType("opset13.Concat", [kv_gather, kv_current])
            kv_shape = WrapType("opset13.ShapeOf", [kv_concat])
            seq = WrapType("opset13.Gather", [kv_shape, AnyInput(), AnyInput()])

            def callback(m: Matcher) -> bool:
                # TODO: Check that seq has axis that really takes sequence len but not any other dimension -- use symbolic infra or look at the constant input
                gather = m.get_match_root()
                target_type = gather.get_output_element_type(0)
                if max_context_len.get_output_element_type(0) != target_type:
                    print(f'Converting {max_context_len.get_output_element_type(0)} of total_seq_len to {target_type}')
                    replacement = opset13.convert(max_context_len, target_type)
                else:
                    replacement = max_context_len
                replace_node(gather, replacement)
                print("DETECTED PATTERN TotalSequenceLengthPattern, CONNECTED TO A DEDICATED PARAMETER")
                return True

            self.register_matcher(Matcher(seq, "TotalSequenceLengthPattern"), callback)

    # TODO: Instead of using the following transformation that matches quite a specific place in a model graph in case when position_ids parameter is missing,
    #       consider replacing always existing attention_mask parameter with a sub-graph using a new slot_mapping parameter.
    class PositionIDsReplacer(MatcherPass):
        def __init__(self):
            MatcherPass.__init__(self)
            self.model_changed = False

            input_ids = AnyInput()
            input_embed = WrapType("opset13.Gather", [AnyInput(), input_ids, AnyInput()])

            position_ids_pattern = AnyInput()
            offset = WrapType('opset13.Constant')
            add_offset = WrapType('opset13.Add', [position_ids_pattern, offset])
            convert = WrapType('opset13.Convert', [add_offset])
            position_embed = WrapType("opset13.Gather", [AnyInput(), convert, AnyInput()])

            add = WrapType("opset13.Add", [input_embed, position_embed])

            def callback(m: Matcher) -> bool:
                mapping = m.get_pattern_value_map()
                replace_node(mapping[position_ids_pattern].get_node(), position_ids.get_node())
                print('APPLIED position_ids PARAMETER INSTEAD OF attention_mask-BASED SUB-GRAPH')
                return True

            self.register_matcher(Matcher(add, "PositionIDsReplacer"), callback)

    m = Manager()
    m.set_per_pass_validation(False)
    m.register_pass(StateManagementPattern())
    m.register_pass(PrevSequenceLengthPattern())
    m.register_pass(TotalSequenceLengthPattern())

    m.register_pass(PositionIDsReplacer())

    m.run_passes(model)

    if has_parameter(model, 'beam_idx'):
        model.remove_parameter(model.input('beam_idx').get_node())
    model.remove_parameter(model.input('attention_mask').get_node())
    for parameter in parameters_to_remove:
        model.remove_parameter(parameter)
    # Remove all Assigns aggressively, the path from the kv-cache concat to Assign can be complicated,
    # but there is no reason to track it and reject part of the Assigns, because the model will remain
    # in incorrect form anyway.
    sinks = model.get_sinks()
    for sink in sinks:
        model.remove_sink(sink)
    for result in results_to_remove:
        model.remove_result(result)
    model.add_parameters(kv_parameters)
    model.add_parameters(model_remaining_params)
    logger.warning('Parameters are reorganized, the state (if exists) is removed')


def require_model_export(model_id, revision=None, subfolder=None):
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


def ov_sample(
    self,
    hidden_states: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> Optional[SamplerOutput]:
    return self.sampler(hidden_states, sampling_metadata)


def ov_compute_logits(self, hidden_states: torch.Tensor,
                      sampling_metadata: SamplingMetadata) -> torch.Tensor:
    return self.logits_processor(None, hidden_states, sampling_metadata)


# class OpenVINOCasualLM(nn.Module):

#     def __init__(
#         self,
#         config: PretrainedConfig,
#     ) -> None:
#         super().__init__()
#         self.config = config
#         self.logits_processor = LogitsProcessor(config.vocab_size,
#                                                 logits_as_input=True)
#         self.sampler = Sampler()

#         # Lazy initialized
#         self.model: nn.Module

#     def forward(
#         self,
#         input_ids: torch.Tensor,
#         positions: torch.Tensor,
#         input_block_ids: torch.Tensor,
#     ) -> torch.Tensor:
#         logits = self.model(input_ids,
#                             cache_ids=positions,
#                             start_ids=input_block_ids)
#         return logits

#     def compute_logits(self, hidden_states: torch.Tensor,
#                        sampling_metadata: SamplingMetadata) -> torch.Tensor:
#         logits = self.logits_processor(None, hidden_states, sampling_metadata)
#         return logits

#     def sample(
#         self,
#         logits: torch.Tensor,
#         sampling_metadata: SamplingMetadata,
#     ) -> Optional[SamplerOutput]:
#         next_tokens = self.sampler(logits, sampling_metadata)
#         return next_tokens


def get_model(model_config: ModelConfig,
              device_config: DeviceConfig,
              kv_cache_dtype: Type,
              **kwargs) -> torch.nn.Module:
    lora_config = kwargs.get("lora_config", None)
    if lora_config:
        raise ValueError(
            f"OpenVINO modeling does not support LoRA, "
            "but LoRA is enabled. Support for this model may "
            "be added in the future. If this is important to you, "
            "please open an issue on github.")

    pt_model = None

    if not is_openvino_optimum_intel():
        raise ValueError("OpenVINO Intel is not installed.")

    import openvino as ov
    from optimum.intel import OVModelForCausalLM
    export = require_model_export(model_config.model)
    if export:
        logger.warning(f'[ INFO ] Provided model id {model_config.model} does not contain OpenVINO IR, the model will be converted to IR with default options. '
                        'If you need to use specific options for model conversion, use optimum-cli export openvino with desired options.')
    else:
        logger.warning(f'[ INFO ] OpenVINO IR is avaialble for provided model id {model_config.model}. '
                        'This IR will be used for inference as-is, all possible options that may affect model conversion are ignored.')

    load_in_8bit = False if os.environ.get('VLLM_OPENVINO_ENABLE_QUANTIZED_WEIGHTS', '1') == '0' else None
    pt_model = OVModelForCausalLM.from_pretrained(
        model_config.model,
        export=export,
        compile=False,
        load_in_8bit=load_in_8bit,
        trust_remote_code=model_config.trust_remote_code
    )
    patch_stateful_model(pt_model.model, kv_cache_dtype)

    # For deployment outside vLLM
    model_file_name = os.environ.get('VLLM_OPENVINO_EXPORTED_IR_NAME', '')
    if model_file_name:
        ov.save_model(pt_model.model, model_file_name)

    core = ov.Core()
    ov_compiled = core.compile_model(pt_model.model, "CPU")
    pt_model._ov_request = ov_compiled.create_infer_request()

    pt_model._openvino_patch_orig_forward = pt_model.forward
    pt_model.forward = partial(ov_wrapper, pt_model)

    from vllm.model_executor.layers.sampler import Sampler
    pt_model.sampler = Sampler(model_config.hf_config.vocab_size)
    pt_model.sample = partial(ov_sample, pt_model)

    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    pt_model.logits_processor = LogitsProcessor(config.vocab_size,
                                                logits_as_input=True)
    pt_model.compute_logits = partial(ov_compute_logits, pt_model)

    return pt_model
