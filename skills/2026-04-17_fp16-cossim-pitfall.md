# FP16 cosine similarity 会误报 → 校验 Triton kernel 必须先 `.float()`

## 场景
调 Triton bwd kernel 的时候用 `torch.nn.functional.cosine_similarity(a, b, dim=0)` 做正确性
对比，如果 `a`, `b` 还是 fp16，可能看到 **0.97-0.98 cos sim**，一眼判定 FAIL，但实际上
kernel 是对的。

## 量化
8M-元素 fp16 张量，两份数据相差约 `1e-4` 的噪声（fp16 的 epsilon 量级）：

```python
a = torch.randn(8_388_608, device="cuda", dtype=torch.float16)
b = a + torch.randn_like(a) * 1e-4  # tiny noise

cos_fp16 = F.cosine_similarity(a, b, dim=0).item()           # 0.983
cos_fp32 = F.cosine_similarity(a.float(), b.float(), dim=0)  # 0.9999999
```

**fp16 dot product 做 8M 项累加**，中间结果会饱和到 fp16 精度（~1e-4 相对误差），导致分
子/分母系统性偏差，cos sim 被拉下去 2-3%。fp32 累加没这问题。

## 怎么用
```python
def cs(a, b):
    return F.cosine_similarity(a.detach().float().flatten(),
                               b.detach().float().flatten(), dim=0).item()
```

## 踩坑成本
2026-04-17 delta-fusion 验证时，误以为 kernel 有 bug，差点回滚。加了 direct 对比发现
`(dq_f.float() - dq_t.float()).abs().max() = 2.4e-4`（fp16 epsilon 级），norm 完全相等
（fp32 307.39 vs 307.39），才意识到是校验侧的精度问题。

## 什么时候用 cos sim 都要 fp32 accumulate
- Triton kernel 输出是 fp16/bf16
- 张量 numel ≥ ~1M（累加次数足够多，fp16 精度损失才显著）
- 期望匹配到 1e-3 相对误差以内（"PASS" 阈值是 0.9999+ 而不是 0.99）

## 反向指引
cos sim < 0.9999 时，先换 fp32 重测，再怀疑 kernel。如果 fp32 还是低，
再去量化 max absolute diff 和 value norm 看相对误差具体多大。
