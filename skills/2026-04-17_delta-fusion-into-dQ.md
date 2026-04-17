# Delta 融入 dQ prologue 省 1 launch（短 N 专项）

## 场景
Flash-attn bwd 标准流水线是 4 个 kernel：**delta** → **dQ** → **dKV** → （GQA reduce）。
其中 `delta = rowsum(dO * O)` 只是一次 elementwise mul + row-reduce，在长 N 上可忽略，
但短 N (N≤2K) 上它的 **launch-bound 性质** 拖累严重。

## 为什么是 launch-bound
`_delta_kernel` grid = `(SEQ_LEN, B * H_Q)`：每个 program 只处理 1 行 × D 维向量，工作量
仅 `D` 次 mul + reduce。N=1K, B=1, H_Q=32 时 grid = 32768 programs，每个 program 的工作
量是 ~512 bytes HBM I/O + 几十条指令。**kernel launch 开销（~19μs）和 program 调度
固定成本** 占比远超 actual work。

## 关键洞察
**dQ 反正要加载 `dO`**（用于 `dp = dO @ V^T`）。多 load 一份 `O` 到 shared mem 做
`rowsum(dO*O)` 几乎零成本（同样 BQ×D tile，和 dO 同形状）。结果：

- 省掉 `_delta_kernel` 整个 launch（`_delta_kernel` 自身时间 ≈ 46μs @ N=1K）
- 新增 dQ 内部 O load ≈ 23μs
- **净收益 ≈ 23μs**（Config A N=1K = **3.8% fwd+bwd**）

## 实现要点
```python
@triton.jit
def _flash_attn_gqa_bwd_dq_kernel(
    Q, K, V, dO, O,  # ← 新增 O_ptr
    dQ, LSE, Delta,
    ..., stride_o*,  # ← 新增 O 的 4 个 stride
    ...,
    STORE_DELTA: tl.constexpr,  # ← 新增 constexpr
):
    ...
    do_block = tl.load(do_ptrs, ...)
    ...
    if STORE_DELTA:
        o_block = tl.load(o_ptrs, ...)
        delta = tl.sum(do_block.to(tl.float32) * o_block.to(tl.float32), axis=1)
        tl.store(delta_ptrs, delta, mask=q_mask)  # 给 dKV 用
    else:
        delta = tl.load(delta_ptrs, mask=q_mask, other=0.0)

    # 主循环使用 in-register `delta[:, None]`，与以前一致
```

## 执行顺序保证
dQ 写完 delta → dKV 读 delta。同 stream 的 Triton launch 是严格有序的，**无需
torch.cuda.synchronize**。dQ grid 完整覆盖 `(N × B × H_Q)`，所以 dKV 开跑时 delta 全部
就绪。

## 适用条件
- Bwd kernel 里有一个 launch-bound 的 "prepass" kernel（delta、rowsum、等）
- 下游大 kernel 反正要加载同一张 tensor（dQ 已经 load `dO`）
- 融进去的额外 tensor load (`O`) 和 prepass kernel 的 load 等价

## 不适用
- Prepass kernel 本身 compute-bound（融了之后下游 kernel 也会 compute-bound）
- Prepass 的输出被多个下游 kernel 共用，且每个下游 tile 粒度不同（难协调）

## 小坑
1. **所有 call site 的 kernel signature 都要更新**：改 `@triton.jit` 签名意味着 tests/
   benchmarks 全部要加 `O_ptr + 4 strides + STORE_DELTA=False`。当 `STORE_DELTA=False`
   时 Triton constexpr 会 DCE 掉 O 相关 load，传 `do` 当 dummy 也行。
2. **正确性校验 fp16 cos sim 会误报**，必须 fp32 accumulate。见
   `2026-04-17_fp16-cossim-pitfall.md`。

## 量化收益
| Config | N | Before | After | 相对提升 |
|--------|---|--------|-------|----------|
| H_Q=32, H_KV=16, D=256, slide=1024 | 1K | 0.604 ms | 0.581 ms | **-3.8%** |
| H_Q=32, H_KV=16, D=256, slide=1024 | 2K | 1.179 ms | 1.144 ms | -3.0% |
| H_Q=8, H_KV=1, D=256, slide=512 | 1K | 0.539 ms | 0.507 ms | **-5.9%** |
| H_Q=8, H_KV=1, D=256, slide=512 | 2K | 0.544 ms | 0.516 ms | -5.1% |
