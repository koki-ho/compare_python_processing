"""実装モジュールのインターフェース一貫性テスト

新しい実装を追加した際に、runner.py が暗黙的に期待する
3関数 (array_sum / matrix_dot / elementwise_sqrt) が揃っているか確認する。
"""

import types
from typing import Any

import pytest

from benchmark.implementations import numba_impl, numpy_impl, pure_python

REQUIRED_FUNCTIONS = ("array_sum", "matrix_dot", "elementwise_sqrt")

IMPLEMENTATIONS: list[types.ModuleType] = [pure_python, numpy_impl, numba_impl]
IMPLEMENTATION_IDS = ["pure_python", "numpy", "numba"]


@pytest.mark.parametrize("module", IMPLEMENTATIONS, ids=IMPLEMENTATION_IDS)
@pytest.mark.parametrize("fn_name", REQUIRED_FUNCTIONS)
def test_has_required_function(module: types.ModuleType, fn_name: str) -> None:
    assert hasattr(module, fn_name), f"{module.__name__!r} is missing '{fn_name}'"


@pytest.mark.parametrize("module", IMPLEMENTATIONS, ids=IMPLEMENTATION_IDS)
@pytest.mark.parametrize("fn_name", REQUIRED_FUNCTIONS)
def test_function_is_callable(module: types.ModuleType, fn_name: str) -> None:
    fn: Any = getattr(module, fn_name)
    assert callable(fn)
