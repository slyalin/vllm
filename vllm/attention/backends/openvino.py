from dataclasses import dataclass
from typing import List, Tuple

import torch
from vllm.attention.backends.abstract import (AttentionBackend)


class OpenVINOAttentionBackend(AttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "openvino"

    @staticmethod
    def get_impl_cls():
        raise NotImplementedError

    @staticmethod
    def make_metadata(*args, **kwargs) -> "OpenVINOAttentionMetadata":
        return OpenVINOAttentionMetadata(*args, **kwargs)

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        raise NotImplementedError

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        raise NotImplementedError

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        raise NotImplementedError


@dataclass
class OpenVINOAttentionMetadata:
    """Metadata for OpenVINOAttentionBackend.
    """
    context_lens: torch.Tensor
    subsequence_begins: torch.Tensor
    block_indices: torch.Tensor
    block_indices_begins: torch.Tensor
    max_context_len: torch.Tensor
