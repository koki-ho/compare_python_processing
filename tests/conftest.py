"""共有フィクスチャ"""

import numpy as np
import pytest
from numpy.typing import NDArray

from benchmark.implementations import numba_impl


@pytest.fixture(scope="session")
def small_arr_np() -> NDArray[np.float64]:
    rng = np.random.default_rng(42)
    arr: NDArray[np.float64] = rng.random(100)
    return arr


@pytest.fixture(scope="session")
def small_mat_np() -> NDArray[np.float64]:
    rng = np.random.default_rng(42)
    mat: NDArray[np.float64] = rng.random((8, 8))
    return mat


@pytest.fixture(scope="session", autouse=True)
def warmup_numba(
    small_arr_np: NDArray[np.float64], small_mat_np: NDArray[np.float64]
) -> None:
    """Numba JIT を事前にコンパイルしておく"""
    numba_impl.warmup(small_arr_np, small_mat_np)
