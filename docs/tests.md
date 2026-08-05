# 测试指南

## 当前 release gate

```bash
# 不使用 GPU；GPU 用例会明确跳过
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q

# 在目标 GPU 上运行完整数值门禁
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q --run-gpu
```

最近一次 e051 结果：CPU 100 passed、50 skipped；B200 GPU 50/50 passed。
其中新增 12 个 H200 registry selection 防回退 case。H100/H200 实机性能
仍需在对应硬件重新认证，不能把 B200 结果当作跨硬件认证。

GPU 覆盖：

- batched forward/backward；
- varlen output/dQ/dK/dV；
- causal、sliding、non-causal vision；
- image-group OR-mask；
- tile/window/packing 语义不变量。

required case 遇到编译失败、资源不足或数值错误必须失败，不得转成成功或静默跳过。

## Registry 与性能合同

```bash
python -m pytest -q \
  tests/test_registry.py \
  tests/test_performance.py \
  tests/test_regression.py \
  tests/test_telemetry.py
```

这些测试检查配置选择、硬件/版本约束、性能结果 schema、回归规则和 telemetry，不执行 GPU kernel。

## 单项 GPU 排查

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_batched_correctness.py --run-gpu
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_varlen_numerics.py --run-gpu
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_semantic_invariants.py --run-gpu
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_image_group_mask.py --run-gpu
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_noncausal_vision_shape.py --run-gpu
```

## 真实模型 integration

`tests/gemma4_integration/` 是独立环境，覆盖 adapter、真实 Gemma-4、memory 与多模态接线。它不在默认 pytest collection 中。

```bash
python tests/gemma4_integration/test_adapter.py
python tests/gemma4_integration/test_gemma4.py --seq-len 1024
python tests/gemma4_integration/test_memory.py
```

真实模型可能需要模型权重和访问凭证；未运行时必须明确写为“未验证”。

## 已保存的 integration 结论

以下是旧环境已完成的结果，保留用于回归参照，不代表本轮 B200 重跑：

- adapter：24 个参数化 case 全通过，cosine >0.999987；
- 覆盖 GQA 1:1/2:1/4:1/8:1、full causal、SWA、D256/D512、batch 1/2；
- Gemma-4-E2B N=1024：logits cosine 0.999758；
- 最后位置 top-1 匹配 100%，top-5 overlap 5/5；
- adapter 每次 forward 命中 35 层，其中 7 个 full、28 个 sliding。

对应历史入口仍保留：

```bash
python tests/gemma4_integration/test_adapter.py
python tests/gemma4_integration/test_gemma4.py --seq-len 1024
python benchmarks/run_final_benchmark.py
```

## H200 varlen 历史测试

H200/Triton 3.2 阶段的 `test_varlen_correctness.py` 同时检查
forward/backward 对 per-sample SDPA，以及 equal-length packed 对 batched
kernel 的等价性。原有命令和结果语义保存在
[`docs/varlen.md`](varlen.md)；旧脚本的 OOR→skip 与 upstream oracle
未实现问题也一并保留，避免把“exit 0”误解成完整认证。

## 历史/非门禁脚本

以下脚本被 `pyproject.toml` 明确排除，不代表当前 release gate：

- `test_packed_dkv.py`：production packed dKV 对旧 split dKV 的历史对比；
- `test_varlen_correctness.py`、`test_varlen_edge_cases.py`、`test_varlen_scaling.py`：H200/Triton 3.2 阶段脚本；
- `test_varlen_vs_flash_attn.py`：上游 oracle 比较尚未实现，当前即使找到模块也不做断言；
- `tests/legacy/`：grouped forward 与 fused backward 失败实验复现。

这些资产后续按 H100/H200/failed 归档；在迁移完成前保留，不再引用为“通过即发布”的门禁。
