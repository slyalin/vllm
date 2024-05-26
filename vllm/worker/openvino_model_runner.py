from typing import List, Optional, Tuple

import torch
from torch import nn

from vllm.attention import get_attn_backend
from vllm.attention.backends.openvino import OpenVINOAttentionMetadata
from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, LoRAConfig, ModelConfig,
                         ParallelConfig, SchedulerConfig, VisionLanguageConfig)
from vllm.logger import init_logger
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.model_loader.openvino import get_model
from vllm.sequence import SamplerOutput, SequenceGroupMetadata

logger = init_logger(__name__)

_PAD_SLOT_ID = -1


class OpenVINOModelRunner:

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        cache_config: CacheConfig,
        load_config: LoadConfig,
        lora_config: Optional[LoRAConfig],
        vision_language_config: Optional[VisionLanguageConfig],
        kv_cache_dtype: Optional[str] = "auto",
        is_driver_worker: bool = False,
        *args,
        **kwargs,
    ):
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        # Currently, CPU worker doesn't support chunked prefill.
        assert self.scheduler_config.chunked_prefill_enabled is False
        self.device_config = device_config
        self.cache_config = cache_config
        self.lora_config = lora_config
        self.vision_language_config = vision_language_config
        self.load_config = load_config
        self.is_driver_worker = is_driver_worker

        self.device = self.device_config.device

        self.kv_cache_dtype = kv_cache_dtype
        self.sliding_window = model_config.get_sliding_window()
        self.block_size = cache_config.block_size

        self.attn_backend = get_attn_backend(
            self.model_config.get_num_attention_heads(self.parallel_config),
            self.model_config.get_head_size(),
            self.model_config.get_num_kv_heads(self.parallel_config),
            self.model_config.get_sliding_window(),
            self.model_config.dtype,
            self.kv_cache_dtype,
            self.block_size)

        # Lazy initialization.
        self.model: nn.Module  # Set after init_Model
        self.block_size: int  # Set after initial profiling.

    def load_model(self) -> None:
        self.model = get_model(
            model_config=self.model_config,
            device_config=self.device_config,
            kv_cache_dtype=self.kv_cache_dtype)


    def _prepare_prompt(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, OpenVINOAttentionMetadata, List[int],
               Optional[torch.Tensor]]:
        assert len(seq_group_metadata_list) > 0
        input_tokens: List[int] = []
        input_positions: List[int] = []
        seq_lens: List[int] = []
        past_lens: List[int] = []
        query_lens: List[int] = []
        subsequence_begins: List[int] = []
        block_indices: List[int] = []
        block_indices_begins: List[int] = []
        multi_modal_input_list: List[torch.Tensor] = []

        # initialize beginning of prefix sums
        subsequence_begins.append(0)
        block_indices_begins.append(0)

        for seq_group_metadata in seq_group_metadata_list:
            assert seq_group_metadata.is_prompt
            seq_ids = list(seq_group_metadata.seq_data.keys())
            assert len(seq_ids) == 1
            seq_id = seq_ids[0]

            seq_data = seq_group_metadata.seq_data[seq_id]
            block_table = seq_group_metadata.block_tables[seq_id]
            prompt_tokens = seq_data.get_token_ids()
            computed_len = seq_data.get_num_computed_tokens()
            prompt_len = len(prompt_tokens)

            computed_block_nums = seq_group_metadata.computed_block_nums
            # Prefix cache was hit.
            # Prefix is not supported with sliding_window
            prefix_cache_hit = (computed_block_nums is not None
                                and len(computed_block_nums) > 0
                                and self.sliding_window is None)

            if prefix_cache_hit:
                computed_len = len(computed_block_nums) * self.block_size
                prompt_tokens = prompt_tokens[computed_len:]

            input_tokens.extend(prompt_tokens)
            seq_lens.append(prompt_len)

            # Token position ids
            # NOTE(woosuk): Here we assume that the first token in the prompt
            # is always the first token in the sequence.
            input_positions.extend(list(range(computed_len, prompt_len)))

            past_lens.append(computed_len)

            subsequence_len = prompt_len - computed_len
            query_lens.append(subsequence_len)
            subsequence_begins.append(subsequence_begins[-1] + subsequence_len)

            block_indices.extend(block_table)
            block_indices_begins.append(block_indices_begins[-1] + len(block_table))

            if seq_group_metadata.multi_modal_data:
                multi_modal_input_list.append(
                    seq_group_metadata.multi_modal_data.data)

        if multi_modal_input_list:
            assert self.vision_language_config, (
                "Multi-modal inputs are only supported by "
                "vision language models.")
            multi_modal_input = torch.cat(multi_modal_input_list,
                                          dim=0).to(self.device)
        else:
            multi_modal_input = None

        input_tokens = torch.tensor(input_tokens,
                                    dtype=torch.long,
                                    device=self.device)  # type: ignore
        input_positions = torch.tensor(input_positions,
                                       dtype=torch.long,
                                       device=self.device)  # type: ignore

        past_lens_tensor = torch.tensor(past_lens,
                                        dtype=torch.int32,
                                        device=self.device)  # type: ignore
        subsequence_begins_tensor = torch.tensor(subsequence_begins,
                                                 dtype=torch.int32,
                                                 device=self.device)  # type: ignore
        block_indices_tensor = torch.tensor(block_indices,
                                            dtype=torch.int32,
                                            device=self.device)  # type: ignore
        block_indices_begins_tensor = torch.tensor(block_indices_begins,
                                                   dtype=torch.int32,
                                                   device=self.device)  # type: ignore

        max_context_len = max(seq_lens)
        max_context_len_tensor = torch.tensor(max_context_len,
                                              dtype=torch.int32,
                                              device=self.device)  # type: ignore

        attn_metadata = self.attn_backend.make_metadata(
            past_lens = past_lens_tensor,
            subsequence_begins = subsequence_begins_tensor,
            block_indices = block_indices_tensor,
            block_indices_begins = block_indices_begins_tensor,
            max_context_len = max_context_len_tensor
        )
        return (input_tokens, input_positions, attn_metadata,
                seq_lens, query_lens, multi_modal_input)

    def _prepare_decode(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, OpenVINOAttentionMetadata]:
        assert len(seq_group_metadata_list) > 0
        input_tokens: List[int] = []
        input_positions: List[int] = []
        seq_lens: List[int] = []
        past_lens: List[int] = []
        subsequence_begins: List[int] = []
        block_indices: List[int] = []
        block_indices_begins: List[int] = []

        # initialize beginning of prefix sums
        subsequence_begins.append(0)
        block_indices_begins.append(0)

        for seq_group_metadata in seq_group_metadata_list:
            assert not seq_group_metadata.is_prompt
            assert seq_group_metadata.token_chunk_size == 1

            seq_ids = list(seq_group_metadata.seq_data.keys())

            for seq_id in seq_ids:
                seq_data = seq_group_metadata.seq_data[seq_id]
                generation_token = seq_data.get_last_token_id()
                input_tokens.append(generation_token)

                seq_len = seq_data.get_len()
                computed_len = position = seq_len - 1
                input_positions.append(position)

                seq_len = seq_len if self.sliding_window is None else min(
                    seq_len, self.sliding_window)
                seq_lens.append(seq_len)

                past_lens.append(computed_len)
                subsequence_begins.append(subsequence_begins[-1] + 1) # 1 is a number of scheduled tokens

                block_table = seq_group_metadata.block_tables[seq_id]
                if self.sliding_window is not None:
                    sliding_window_blocks = (self.sliding_window //
                                             self.block_size)
                    block_table = block_table[-sliding_window_blocks:]
                block_indices.extend(block_table)
                block_indices_begins.append(block_indices_begins[-1] + len(block_table))

        input_tokens = torch.tensor(input_tokens,
                                    dtype=torch.long,
                                    device=self.device)
        input_positions = torch.tensor(input_positions,
                                       dtype=torch.long,
                                       device=self.device)

        past_lens_tensor = torch.tensor(past_lens,
                                        dtype=torch.int32,
                                        device=self.device)  # type: ignore
        subsequence_begins_tensor = torch.tensor(subsequence_begins,
                                                 dtype=torch.int32,
                                                 device=self.device)  # type: ignore
        block_indices_tensor = torch.tensor(block_indices,
                                            dtype=torch.int32,
                                            device=self.device)  # type: ignore
        block_indices_begins_tensor = torch.tensor(block_indices_begins,
                                                   dtype=torch.int32,
                                                   device=self.device)  # type: ignore

        max_context_len = max(seq_lens)
        max_context_len_tensor = torch.tensor(max_context_len,
                                              dtype=torch.int32,
                                              device=self.device)  # type: ignore

        attn_metadata = self.attn_backend.make_metadata(
            past_lens = past_lens_tensor,
            subsequence_begins = subsequence_begins_tensor,
            block_indices = block_indices_tensor,
            block_indices_begins = block_indices_begins_tensor,
            max_context_len = max_context_len_tensor
        )
        return (
            input_tokens,
            input_positions,
            attn_metadata,
        )

    def prepare_input_tensors(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, OpenVINOAttentionMetadata, SamplingMetadata,
               Optional[torch.Tensor]]:
        multi_modal_input = None

        # NOTE: We assume that all sequences in the group are all prompts or
        # all decodes.
        is_prompt = seq_group_metadata_list[0].is_prompt
        # Prepare input tensors.
        if is_prompt:
            (input_tokens, input_positions, attn_metadata,
                seq_lens, query_lens, multi_modal_input
                ) = self._prepare_prompt(seq_group_metadata_list)
        else:
            (input_tokens, input_positions,
                attn_metadata) = self._prepare_decode(seq_group_metadata_list)
            seq_lens = []
            query_lens = []

        sampling_metadata = SamplingMetadata.prepare(
            seq_group_metadata_list,
            seq_lens,
            query_lens,
            self.device,
            pin_memory=False)

        return (input_tokens, input_positions, attn_metadata,
                sampling_metadata, multi_modal_input)

    @torch.inference_mode()
    def execute_model(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        kv_caches: List[Tuple["ov.Tensor", "ov.Tensor"]],
    ) -> Optional[SamplerOutput]:
        (input_tokens, input_positions, attn_metadata, sampling_metadata,
         multi_modal_input
         ) = self.prepare_input_tensors(seq_group_metadata_list)

        model_executable = self.model
        execute_model_kwargs = {
            "input_ids": input_tokens,
            "positions": input_positions,
            "kv_caches": kv_caches,
            "attn_metadata": attn_metadata,
        }
        if self.vision_language_config:
            execute_model_kwargs.update({"image_input": multi_modal_input})

        hidden_states = model_executable(**execute_model_kwargs, sampling_metadata=sampling_metadata)

        # Compute the logits.
        logits = self.model.compute_logits(hidden_states, sampling_metadata)

        # Only perform sampling in the driver worker.
        if not self.is_driver_worker:
            return None

        # Sample the next token.
        output = self.model.sample(
            logits=logits,
            sampling_metadata=sampling_metadata,
        )
        return output
