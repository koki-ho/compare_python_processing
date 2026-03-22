"""NumPy implementations for benchmarking."""

import numpy as np
from numpy.typing import NDArray


def array_sum(arr: NDArray[np.float64]) -> np.float64:
    return np.sum(arr)


def matrix_dot(
    a: NDArray[np.float64], b: NDArray[np.float64]
) -> NDArray[np.float64]:
    result: NDArray[np.float64] = np.dot(a, b)
    return result


def elementwise_sqrt(arr: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.sqrt(arr)
