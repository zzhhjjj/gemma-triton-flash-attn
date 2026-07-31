"""Context-parallel all-to-all primitives.

Self-contained copy of LlamaFactory v1's SeqAllToAll4D so the kernel package
has no LlamaFactory import. Original source:
LLaMA-Factory/src/llamafactory/v1/plugins/model_plugins/parallelization/seq_comm.py
"""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.distributed as dist
from torch import Tensor


def all_to_all_tensor(
    local_input: Tensor,
    scatter_dim: int,
    gather_dim: int,
    group: Optional[dist.ProcessGroup] = None,
) -> Tensor:
    seq_world_size = dist.get_world_size(group)
    input_list = [t.contiguous() for t in torch.tensor_split(local_input, seq_world_size, scatter_dim)]
    output_list = [torch.empty_like(input_list[0]) for _ in range(seq_world_size)]
    dist.all_to_all(output_list, input_list, group=group)
    return torch.cat(output_list, dim=gather_dim).contiguous()


class SeqAllToAll4D(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        local_input: Tensor,
        scatter_dim: int,
        gather_dim: int,
    ) -> Tensor:
        ctx.group = group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        return all_to_all_tensor(local_input, scatter_dim, gather_dim, group)

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> tuple[None, Tensor, None, None]:
        return (
            None,
            all_to_all_tensor(grad_output[0], ctx.gather_dim, ctx.scatter_dim, ctx.group),
            None,
            None,
        )
