"""CPU distributed tests for the Ulysses adapter and its autograd path."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from gemma_triton_flash_attn.hf_integration import (
    _build_ulysses_attention,
    register_triton_attention,
    register_triton_attention_varlen,
    triton_gqa_attention,
    triton_gqa_varlen_attention,
)


def _torch_gqa_attention(
    module, query, key, value, attention_mask=None, scaling=None, **kwargs
):
    """Small differentiable attention reference matching the HF adapter contract."""
    del module, kwargs
    if attention_mask is not None:
        raise AssertionError("The Ulysses unit test expects kernel-owned masking.")

    groups = query.shape[1] // key.shape[1]
    key = key.repeat_interleave(groups, dim=1)
    value = value.repeat_interleave(groups, dim=1)
    scale = scaling if scaling is not None else query.shape[-1] ** -0.5
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    causal = torch.ones(
        scores.shape[-2:], dtype=torch.bool, device=scores.device
    ).tril()
    scores = scores.masked_fill(~causal, torch.finfo(scores.dtype).min)
    probabilities = torch.softmax(scores, dim=-1)
    output = torch.matmul(probabilities, value)
    return output.transpose(1, 2).contiguous(), None


def _run_case(rank: int, world_size: int, head_dim: int, num_kv_heads: int) -> None:
    torch.manual_seed(2026 + head_dim + num_kv_heads)
    batch_size, num_query_heads, sequence_length = 1, 8, 8
    query_full = (
        torch.randn(batch_size, num_query_heads, sequence_length, head_dim) * 0.1
    )
    key_full = torch.randn(batch_size, num_kv_heads, sequence_length, head_dim) * 0.1
    value_full = torch.randn(batch_size, num_kv_heads, sequence_length, head_dim) * 0.1

    query = (
        query_full.chunk(world_size, dim=2)[rank].detach().clone().requires_grad_(True)
    )
    key = key_full.chunk(world_size, dim=2)[rank].detach().clone().requires_grad_(True)
    value = (
        value_full.chunk(world_size, dim=2)[rank].detach().clone().requires_grad_(True)
    )

    module = SimpleNamespace(head_dim=head_dim, is_causal=True)
    ulysses_attention = _build_ulysses_attention(dist.group.WORLD, _torch_gqa_attention)
    output, _ = ulysses_attention(module, query, key, value, scaling=head_dim**-0.5)
    output.square().sum().backward()

    query_ref = query_full.detach().clone().requires_grad_(True)
    key_ref = key_full.detach().clone().requires_grad_(True)
    value_ref = value_full.detach().clone().requires_grad_(True)
    output_ref, _ = _torch_gqa_attention(
        module,
        query_ref,
        key_ref,
        value_ref,
        scaling=head_dim**-0.5,
    )
    output_ref.square().sum().backward()

    torch.testing.assert_close(
        output, output_ref.chunk(world_size, dim=1)[rank], rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        query.grad, query_ref.grad.chunk(world_size, dim=2)[rank], rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        key.grad, key_ref.grad.chunk(world_size, dim=2)[rank], rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        value.grad, value_ref.grad.chunk(world_size, dim=2)[rank], rtol=1e-5, atol=1e-6
    )


def _distributed_worker(rank: int, world_size: int, rendezvous_file: str) -> None:
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{rendezvous_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        # Gemma 4 sliding shape: KV heads divide CP.
        _run_case(rank, world_size, head_dim=256, num_kv_heads=4)
        dist.barrier()
        # Gemma 4 global shape: KV heads are fewer than CP and must be replicated.
        _run_case(rank, world_size, head_dim=512, num_kv_heads=1)
    finally:
        dist.destroy_process_group()


def test_attention_registration_uses_public_interface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def record_registration(name, attention_fn):
        calls.append((name, attention_fn))

    monkeypatch.setattr("transformers.AttentionInterface.register", record_registration)
    register_triton_attention("test_triton_gqa")
    register_triton_attention_varlen("test_triton_gqa_varlen")

    assert calls == [
        ("test_triton_gqa", triton_gqa_attention),
        ("test_triton_gqa_varlen", triton_gqa_varlen_attention),
    ]


def test_ulysses_forward_backward_matches_full_sequence_reference() -> None:
    world_size = 2
    with tempfile.TemporaryDirectory() as temp_dir:
        rendezvous_file = str(Path(temp_dir) / "distributed_init")
        mp.spawn(
            _distributed_worker,
            args=(world_size, rendezvous_file),
            nprocs=world_size,
            join=True,
        )


def test_ulysses_rejects_nondivisible_heads() -> None:
    if not dist.is_available():
        pytest.skip("torch.distributed is unavailable")

    with tempfile.TemporaryDirectory() as temp_dir:
        rendezvous_file = str(Path(temp_dir) / "single_rank_init")
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
        dist.init_process_group(
            "gloo", init_method=f"file://{rendezvous_file}", rank=0, world_size=1
        )
        try:
            with pytest.raises(ValueError, match="at least two ranks"):
                _build_ulysses_attention(dist.group.WORLD, _torch_gqa_attention)
        finally:
            dist.destroy_process_group()
