"""Benchmark runner: measures and displays performance comparisons."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from tabulate import tabulate

from benchmark.implementations import numba_impl, numpy_impl, pure_python

def _load_config(config_path: Path) -> dict[str, int]:
    with config_path.open() as f:
        data: dict[str, Any] = yaml.safe_load(f)
    cfg = data["benchmark"]
    return {
        "array_size": int(cfg["array_size"]),
        "matrix_size": int(cfg["matrix_size"]),
        "repeat": int(cfg["repeat"]),
    }


def _measure(fn: Callable[..., Any], *args: Any, repeat: int = 5) -> float:
    """Return median wall-clock time in milliseconds over `repeat` runs."""
    times: list[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    return statistics.median(times) * 1000  # → ms


def _speedup(baseline_ms: float, ms: float) -> str:
    return f"{baseline_ms / ms:.1f}x"


def run(config_path: Path | None = None) -> None:
    if config_path is None:
        config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    cfg = _load_config(config_path)
    array_size = cfg["array_size"]
    matrix_size = cfg["matrix_size"]
    repeat = cfg["repeat"]

    rng = np.random.default_rng(42)
    arr_np = rng.random(array_size)
    mat_np = rng.random((matrix_size, matrix_size))

    arr_py: list[float] = arr_np.tolist()
    mat_py: list[list[float]] = mat_np.tolist()

    # ── Numba warmup (JIT compile) ─────────────────────────────────────────
    print(
        f"Config: array_size={array_size:,}  matrix_size={matrix_size}  repeat={repeat}"
    )
    print("Warming up Numba JIT (first-call compilation)...")
    numba_impl.warmup(arr_np, mat_np)

    # ── check for Rust extension ───────────────────────────────────────────
    rust_ext: Any = None
    try:
        import rust_ext as _rust_ext  # noqa: PLC0415

        if hasattr(_rust_ext, "array_sum"):
            rust_ext = _rust_ext
        else:
            raise ImportError("rust_ext exists but is not built yet")
    except ImportError:
        print("rust_ext not found — skipping PyO3/Rust. Build it with:")
        print("  cd rust_ext && uv run maturin develop --release && cd ..")

    print()

    # ── per-task benchmarks ────────────────────────────────────────────────
    def m(fn: Callable[..., Any], *args: Any) -> float:
        return _measure(fn, *args, repeat=repeat)

    tasks: list[tuple[str, list[tuple[str, float]]]] = []

    # 1. array_sum
    rows: list[tuple[str, float]] = []
    rows.append(("Pure Python", m(pure_python.array_sum, arr_py)))
    rows.append(("NumPy", m(numpy_impl.array_sum, arr_np)))
    rows.append(("Numba", m(numba_impl.array_sum, arr_np)))
    if rust_ext is not None:
        rows.append(("PyO3/Rust", m(rust_ext.array_sum, arr_np)))
    tasks.append((f"array_sum  (N={array_size:,})", rows))

    # 2. matrix_dot
    rows = []
    rows.append(("Pure Python", m(pure_python.matrix_dot, mat_py, mat_py)))
    rows.append(("NumPy", m(numpy_impl.matrix_dot, mat_np, mat_np)))
    rows.append(("Numba", m(numba_impl.matrix_dot, mat_np, mat_np)))
    if rust_ext is not None:
        rows.append(("PyO3/Rust", m(rust_ext.matrix_dot, mat_np, mat_np)))
    tasks.append((f"matrix_dot ({matrix_size}×{matrix_size})", rows))

    # 3. elementwise_sqrt
    rows = []
    rows.append(("Pure Python", m(pure_python.elementwise_sqrt, arr_py)))
    rows.append(("NumPy", m(numpy_impl.elementwise_sqrt, arr_np)))
    rows.append(("Numba", m(numba_impl.elementwise_sqrt, arr_np)))
    if rust_ext is not None:
        rows.append(("PyO3/Rust", m(rust_ext.elementwise_sqrt, arr_np)))
    tasks.append((f"elementwise_sqrt (N={array_size:,})", rows))

    # ── display ────────────────────────────────────────────────────────────
    for task_name, task_rows in tasks:
        baseline = task_rows[0][1]
        table = [
            (method, f"{ms:.2f}", _speedup(baseline, ms))
            for method, ms in task_rows
        ]
        print(f"Benchmark: {task_name}")
        print(
            tabulate(
                table,
                headers=["Method", "Time (ms)", "Speedup"],
                tablefmt="rounded_outline",
                colalign=("left", "right", "right"),
            )
        )
        print()
