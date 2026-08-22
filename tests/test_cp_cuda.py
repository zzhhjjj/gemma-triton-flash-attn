"""NCCL/Triton correctness test for Gemma 4 Ulysses attention.

Run on a machine with at least two CUDA devices:

    torchrun --standalone --nproc-per-node=2 tests/test_cp_cuda.py

The test also supports four or eight ranks, which exercise KV replication for
Gemma 4 global layers with fewer KV heads than CP ranks.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from gemma_triton_flash_attn import (
    attention_gqa_ref,
    attention_swa_ref,
    register_triton_attention_ulysses,
)


def _cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(left.float().flatten(), right.float().flatten(), dim=0)


def _run_case(
    *, head_dim: int, num_query_heads: int, num_kv_heads: int, sliding_window: int
) -> None:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    # Keep the sequence at a representative attention tile size. Triton 3.2's
    # Hopper layout pass can abort (rather than raise) for the otherwise unused
    # D=256/BLOCK_Q=64 specialization produced by sequences shorter than 128.
    sequence_length = 512

    if num_query_heads % world_size != 0:
        raise AssertionError(
            f"Hq={num_query_heads} must be divisible by world_size={world_size}"
        )

    torch.manual_seed(2026 + head_dim + sliding_window)
    query_full = torch.randn(
        1,
        num_query_heads,
        sequence_length,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    key_full = torch.randn(
        1,
        num_kv_heads,
        sequence_length,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    value_full = torch.randn_like(key_full)
    grad_output_full = torch.randn(
        1,
        sequence_length,
        num_query_heads,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )

    query = (
        query_full.chunk(world_size, dim=2)[rank].detach().clone().requires_grad_(True)
    )
    key = key_full.chunk(world_size, dim=2)[rank].detach().clone().requires_grad_(True)
    value = (
        value_full.chunk(world_size, dim=2)[rank].detach().clone().requires_grad_(True)
    )
    grad_output = grad_output_full.chunk(world_size, dim=1)[rank].contiguous()

    implementation = f"triton_gqa_ulysses_test_d{head_dim}_w{sliding_window}"
    register_triton_attention_ulysses(dist.group.WORLD, name=implementation)
    attention_fn = ALL_ATTENTION_FUNCTIONS.get_interface(implementation, None)
    module = SimpleNamespace(head_dim=head_dim, is_causal=True)
    output, _ = attention_fn(
        module,
        query,
        key,
        value,
        attention_mask=None,
        dropout=0.0,
        scaling=None,
        softcap=None,
        sliding_window=sliding_window or None,
    )
    output.backward(grad_output)

    query_ref = query_full.detach().clone().requires_grad_(True)
    key_ref = key_full.detach().clone().requires_grad_(True)
    value_ref = value_full.detach().clone().requires_grad_(True)
    if sliding_window:
        output_ref = attention_swa_ref(query_ref, key_ref, value_ref, sliding_window)
    else:
        output_ref = attention_gqa_ref(query_ref, key_ref, value_ref, causal=True)

    output_ref = output_ref.transpose(1, 2).contiguous()
    output_ref.backward(grad_output_full)

    local_output_ref = output_ref.chunk(world_size, dim=1)[rank]
    similarities = torch.stack(
        [
            _cosine(output, local_output_ref),
            _cosine(query.grad, query_ref.grad.chunk(world_size, dim=2)[rank]),
            _cosine(key.grad, key_ref.grad.chunk(world_size, dim=2)[rank]),
            _cosine(value.grad, value_ref.grad.chunk(world_size, dim=2)[rank]),
        ]
    )
    dist.all_reduce(similarities, op=dist.ReduceOp.MIN)
    if torch.any(similarities < 0.999):
        raise AssertionError(
            f"D={head_dim}, window={sliding_window}: minimum output/dQ/dK/dV cosine values "
            f"were {similarities.tolist()}"
        )

    if rank == 0:
        print(
            f"PASS D={head_dim} Hq:Hkv={num_query_heads}:{num_kv_heads} "
            f"window={sliding_window} cos={similarities.tolist()}",
            flush=True,
        )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "This test requires CUDA and must be launched with torchrun."
        )

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    try:
        # Gemma 4 sliding attention: D=256 and a finite causal window.
        _run_case(head_dim=256, num_query_heads=8, num_kv_heads=4, sliding_window=128)
        dist.barrier()
        # Gemma 4 global attention: D=512 and Hkv < CP on four-rank runs.
        _run_case(head_dim=512, num_query_heads=8, num_kv_heads=1, sliding_window=0)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
