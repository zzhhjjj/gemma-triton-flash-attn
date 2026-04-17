# Allocation overhead 在 training bwd 里 ≠ 瓶颈：10-15μs，3-5% of bwd

## 场景
Triton flash-attention bwd 里每次 call 要分配 `delta, dq, dk, dv` 4 个 tensor，
直觉：caching allocator 多次调用是 Python 开销，可能是 short-N bwd 瓶颈之一。

## 量化

`benchmarks/alloc_overhead.py` 分离 "alloc INSIDE timed region" vs "pre-allocated OUTSIDE"：

| Config | bwd Total | Kernel only (pre-alloc) | **Alloc overhead** | Python autograd overhead |
|--------|-----------|-------------------------|--------------------|--------------------------|
| A N=1K | 374 μs | 332 μs | **12 μs** | 31 μs |
| B N=1K | 296 μs | 181 μs | **14 μs** | **101 μs (35%!)** |
| B N=2K | 354 μs | 258 μs | **9 μs** | 86 μs |
| B N=4K | 402 μs | 354 μs | **12 μs** | 36 μs |

**结论：alloc 只占 bwd 的 3-5%**，不是瓶颈。真正大头是
`torch.autograd.Function` **框架 overhead**。

## 为什么 alloc 这么小
PyTorch 的 CUDA caching allocator **命中 warm cache 很快**：
- 第一次调用：cudaMalloc 可能上百 μs（cold）
- 后续调用：仅从 free list 取块，~2-5μs
- cudaMemsetAsync (zeros_like)：~5μs（硬件 memset engine 异步）

训练 loop 里 bwd 反复用同形状 buffer → 永远 hot → 分配便宜。

## 为什么 autograd Python overhead 大 (111μs)

测量 (`triton_launch_overhead.py`)：
```
2 kernel launches back-to-back (no autograd):  174 μs  <-- GPU-bound
same wrapped in torch.autograd.Function:       285 μs  <-- +111 μs
```

111μs 拆分:
- 2× Triton kernel `fn[grid](args)` Python wrapper: ~40μs（**可被 GPU work 掩盖**）
- `torch.autograd.Function.backward` 框架 dispatch + ctx.saved_tensors: **~80μs (CPU-exclusive)**
- 其他（allocs that aren't in pre-allocated variant）: ~20μs

**关键观察：在短 N (GPU 快完成)，CPU work > GPU work → overhead 露出来**。
长 N（GPU work 多）时 Python 被完全 hide，bwd 是 GPU-bound。

## 因此不要
- 投入精力做 "合并 allocations"、"复用 buffer 跨 call" 等优化（省不了多少）
- 怀疑 `torch.empty_like` 慢（hot path 下 < 5μs）

## 要做（如果真关心 short-N Python overhead）
1. **CUDA Graph** 包裹 fwd+bwd：消掉全部 CPU launches（~100μs potential）但需 static shapes
2. **torch.compile**：dynamo trace 后可消 autograd 层的 Python 开销
3. **`torch.library.custom_op` + `register_autograd`**：C++-level autograd dispatch，比 `torch.autograd.Function` 快 ~30-50μs

这三个都是 **API-level 重构**，不是 kernel 级优化，ROI 不一定划算。

## 一个免费的小优化
合并 `dk + dv` alloc：
```python
# 前
dk = torch.zeros_like(k)
dv = torch.zeros_like(v)

# 后（省 1 allocator + 1 memset, ~3-5μs）
dkv = torch.empty((2,) + k.shape, dtype=k.dtype, device=k.device)
dkv.zero_()
dk, dv = dkv[0], dkv[1]
```

收益在 noise (±5μs) 内不显著，但语义零成本，值得保留。
