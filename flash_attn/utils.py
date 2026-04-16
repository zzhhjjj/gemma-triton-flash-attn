import torch
import time
from typing import Callable, Dict, List, Tuple, Optional


def benchmark_fn(
    fn: Callable,
    *args,
    warmup: int = 10,
    rep: int = 100,
    **kwargs,
) -> float:
    """Benchmark a single function call. Returns median time in ms."""
    # Warmup
    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()

    # Timed runs using CUDA events for precise GPU timing
    times = []
    for _ in range(rep):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    return times[len(times) // 2]  # median
    # return sum(times) / len(times) # mean


def benchmark(
    implementations: Dict[str, Callable],
    input_shapes: List[Tuple],
    input_fn: Optional[Callable] = None,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
    warmup: int = 10,
    rep: int = 100,
    verify: bool = True,
    atol: float = 1e-2,
    rtol: float = 1e-2,
) -> List[Dict]:
    """
    Benchmark multiple implementations across different input shapes.

    Args:
        implementations: {"name": fn} dict. Each fn takes the same input tensor(s).
        input_shapes:    List of shapes to benchmark.
        input_fn:        Optional factory `input_fn(shape, dtype, device) -> args tuple`.
                         Defaults to creating a single `torch.randn` tensor.
        dtype:           Tensor dtype.
        device:          Device string.
        warmup:          Warmup iterations per benchmark.
        rep:             Timed iterations per benchmark.
        verify:          If True, check all implementations produce the same output
                         (compared against the first implementation).

    Returns:
        List of result dicts, one per (shape, impl) combination.
    """
    if input_fn is None:
        input_fn = lambda shape, dt, dev: (torch.randn(shape, dtype=dt, device=dev),)

    names = list(implementations.keys())
    results = []

    # Header
    name_width = max(len(n) for n in names)
    print(f"\n{'Shape':<30} {'Impl':<{name_width}}  {'Time (ms)':>10}  {'Speedup':>8}")
    print("-" * (30 + name_width + 24))

    for shape in input_shapes:
        args = input_fn(shape, dtype, device)

        # Run first impl as reference
        ref_output = None
        ref_time = None

        for name in names:
            fn = implementations[name]
            out = fn(*args)

            # Verify correctness
            if verify:
                if ref_output is None:
                    ref_output = out
                else:
                    if not torch.allclose(ref_output, out, atol=atol, rtol=rtol):
                        max_diff = (ref_output - out).abs().max().item()
                        print(f"  WARNING: {name} output differs from {names[0]} "
                              f"(max diff={max_diff:.2e})")

            # Benchmark
            t = benchmark_fn(fn, *args, warmup=warmup, rep=rep)

            if ref_time is None:
                ref_time = t

            speedup = ref_time / t
            speedup_str = f"{speedup:.2f}x"

            print(f"{str(shape):<30} {name:<{name_width}}  {t:>10.4f}  {speedup_str:>8}")

            results.append({
                "shape": shape,
                "impl": name,
                "time_ms": t,
                "speedup": speedup,
            })

        # Blank line between shapes
        ref_output = None
        print()

    return results
