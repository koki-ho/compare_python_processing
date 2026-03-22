"""runner.py のユニットテスト

_load_config / _measure / _speedup の各ロジックを独立して検証する。
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from benchmark.runner import _load_config, _measure, _speedup


class TestLoadConfig:
    def test_returns_all_keys(self, tmp_path: Path) -> None:
        config = tmp_path / "config.yaml"
        config.write_text(
            "benchmark:\n  array_size: 1000\n  matrix_size: 16\n  repeat: 3\n"
        )
        assert _load_config(config) == {
            "array_size": 1000,
            "matrix_size": 16,
            "repeat": 3,
        }

    def test_values_are_int(self, tmp_path: Path) -> None:
        config = tmp_path / "config.yaml"
        config.write_text(
            "benchmark:\n  array_size: 500\n  matrix_size: 8\n  repeat: 5\n"
        )
        cfg = _load_config(config)
        for key in ("array_size", "matrix_size", "repeat"):
            assert isinstance(cfg[key], int), f"{key} should be int"


class TestMeasure:
    def test_returns_float(self) -> None:
        result = _measure(lambda: None, repeat=3)
        assert isinstance(result, float)

    def test_calls_fn_repeat_times(self) -> None:
        call_count = 0

        def fn() -> None:
            nonlocal call_count
            call_count += 1

        _measure(fn, repeat=7)
        assert call_count == 7

    def test_result_is_milliseconds(self) -> None:
        # 3回分の (start, end): 1ms, 2ms, 3ms → 中央値 = 2ms
        counter_values = [0.0, 0.001, 0.001, 0.003, 0.003, 0.006]
        with patch("time.perf_counter", side_effect=counter_values):
            result = _measure(lambda: None, repeat=3)
        assert result == pytest.approx(2.0)


class TestSpeedup:
    @pytest.mark.parametrize(
        ("baseline_ms", "ms", "expected"),
        [
            (10.0, 2.0, "5.0x"),
            (1.0, 1.0, "1.0x"),
            (3.0, 2.0, "1.5x"),
        ],
    )
    def test_calculation(
        self, baseline_ms: float, ms: float, expected: str
    ) -> None:
        assert _speedup(baseline_ms, ms) == expected
