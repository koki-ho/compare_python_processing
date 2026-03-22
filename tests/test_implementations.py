"""実装間の計算結果一致テスト

Pure Python をリファレンスとして、NumPy・Numba の結果が一致することを確認する。
実装をリファクタリングした際の安全網として機能する。
"""

import numpy as np
import pytest
from numpy.typing import NDArray

from benchmark.implementations import numba_impl, numpy_impl, pure_python


class TestArraySum:
    def test_numpy_matches_pure_python(
        self, small_arr_np: NDArray[np.float64]
    ) -> None:
        expected = pure_python.array_sum(small_arr_np.tolist())
        actual = float(numpy_impl.array_sum(small_arr_np))
        assert actual == pytest.approx(expected, rel=1e-9)

    def test_numba_matches_pure_python(
        self, small_arr_np: NDArray[np.float64]
    ) -> None:
        expected = pure_python.array_sum(small_arr_np.tolist())
        actual = numba_impl.array_sum(small_arr_np)
        assert actual == pytest.approx(expected, rel=1e-9)


class TestMatrixDot:
    def test_numpy_matches_pure_python(
        self, small_mat_np: NDArray[np.float64]
    ) -> None:
        mat_py = small_mat_np.tolist()
        expected = np.array(pure_python.matrix_dot(mat_py, mat_py))
        actual = numpy_impl.matrix_dot(small_mat_np, small_mat_np)
        np.testing.assert_allclose(actual, expected, rtol=1e-10)

    def test_numba_matches_pure_python(
        self, small_mat_np: NDArray[np.float64]
    ) -> None:
        mat_py = small_mat_np.tolist()
        expected = np.array(pure_python.matrix_dot(mat_py, mat_py))
        actual = numba_impl.matrix_dot(small_mat_np, small_mat_np)
        np.testing.assert_allclose(actual, expected, rtol=1e-10)


class TestElementwiseSqrt:
    def test_numpy_matches_pure_python(
        self, small_arr_np: NDArray[np.float64]
    ) -> None:
        expected = np.array(pure_python.elementwise_sqrt(small_arr_np.tolist()))
        actual = numpy_impl.elementwise_sqrt(small_arr_np)
        np.testing.assert_allclose(actual, expected, rtol=1e-14)

    def test_numba_matches_pure_python(
        self, small_arr_np: NDArray[np.float64]
    ) -> None:
        expected = np.array(pure_python.elementwise_sqrt(small_arr_np.tolist()))
        actual = numba_impl.elementwise_sqrt(small_arr_np)
        np.testing.assert_allclose(actual, expected, rtol=1e-14)
