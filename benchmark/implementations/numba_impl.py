"""Numba JIT-compiled implementations for benchmarking."""

import numba  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import NDArray


@numba.njit(cache=True)  # type: ignore[untyped-decorator]
def array_sum(arr: NDArray[np.float64]) -> float:
    total = 0.0
    for x in arr:
        total += x
    return total


@numba.njit(cache=True)  # type: ignore[untyped-decorator]
def matrix_dot(
    a: NDArray[np.float64], b: NDArray[np.float64]
) -> NDArray[np.float64]:
    n = a.shape[0]
    result = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for k in range(n):
            a_ik = a[i, k]
            for j in range(n):
                result[i, j] += a_ik * b[k, j]
    return result


@numba.njit(cache=True)  # type: ignore[untyped-decorator]
def elementwise_sqrt(arr: NDArray[np.float64]) -> NDArray[np.float64]:
    result = np.empty(len(arr), dtype=np.float64)
    for i in range(len(arr)):
        result[i] = arr[i] ** 0.5
    return result


def warmup(arr: NDArray[np.float64], mat: NDArray[np.float64]) -> None:
    """JIT warmup: run once to trigger compilation."""
    tiny = arr[:10].copy()
    tiny_mat = mat[:4, :4].copy()
    array_sum(tiny)
    matrix_dot(tiny_mat, tiny_mat)
    elementwise_sqrt(tiny)
