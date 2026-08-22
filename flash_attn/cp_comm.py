# Copyright 2025 Bytedance Ltd. and/or its affiliates and the LlamaFactory team.
# Copyright 2026 gemma-triton-flash-attn contributors.
#
# This code is inspired by the Bytedance verl Ulysses implementation:
# https://github.com/verl-project/verl/blob/77476af84cc074edf5a6437f8d5ea418d7a54916/verl/utils/ulysses.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "all_to_all_tensor requires an initialized torch.distributed process group."
        )

    seq_world_size = dist.get_world_size(group)
    scatter_size = local_input.shape[scatter_dim]
    if scatter_size % seq_world_size != 0:
        raise ValueError(
            f"Tensor dimension {scatter_dim} (size={scatter_size}) must be divisible by "
            f"the process-group size ({seq_world_size})."
        )

    input_list = [
        t.contiguous()
        for t in torch.tensor_split(local_input, seq_world_size, scatter_dim)
    ]
    if dist.get_backend(group) == "gloo":
        # Gloo does not implement list-based all-to-all. Its CPU-only path is
        # useful for CI, so emulate the same exchange with an all-gather. The
        # production NCCL path below remains the memory-efficient collective.
        gathered_inputs = [torch.empty_like(local_input) for _ in range(seq_world_size)]
        dist.all_gather(gathered_inputs, local_input.contiguous(), group=group)
        group_rank = dist.get_rank(group)
        output_list = [
            torch.tensor_split(source, seq_world_size, scatter_dim)[
                group_rank
            ].contiguous()
            for source in gathered_inputs
        ]
    else:
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
            all_to_all_tensor(
                grad_output[0], ctx.gather_dim, ctx.scatter_dim, ctx.group
            ),
            None,
            None,
        )
