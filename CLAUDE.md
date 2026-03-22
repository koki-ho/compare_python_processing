# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## アーキテクチャ

### 実行フロー

`main.py` → `benchmark/runner.py::run()` → 各実装モジュール

`runner.py` が設定読み込み・データ生成・計測・結果表示をすべて担う。計測は `_measure()` でウォールクロック時間の中央値 (ms) を取得し、Pure Python をベースラインとしてスピードアップ倍率を表示する。

### 実装モジュール (`benchmark/implementations/`)

4 つの実装はすべて同じ 3 関数 (`array_sum`, `matrix_dot`, `elementwise_sqrt`) を提供する。

- `pure_python.py` — `list[float]` を受け取るループ実装 (ベースライン)
- `numpy_impl.py` — `np.ndarray` を受け取るベクトル演算実装
- `numba_impl.py` — `np.ndarray` を受け取る `@numba.njit(cache=True)` 実装。初回実行時に JIT コンパイルが走るため `warmup()` を先に呼ぶ必要がある
- `rust_ext` — PyO3 で生成される Rust 拡張モジュール。`rust_ext/` 以下を `maturin develop --release` でビルドして初めて使用可能。未ビルド時は runner がスキップする

### Rust 拡張 (`rust_ext/`)

独立した Cargo クレートかつ maturin プロジェクト。`pyproject.toml` と `Cargo.toml` を両方持つ。ビルドすると `.so` が生成されプロジェクトルートから `import rust_ext` できる。

### 設定 (`configs/config.yaml`)

`array_size` / `matrix_size` / `repeat` の 3 パラメータのみ。`runner.py` が YAML を直接読む。

## ツールチェーン

- パッケージ管理: `uv`
- Lint: `ruff` (E/F/I ルール、line-length=88)
- 型チェック: `mypy` (strict モード)
- Rust バインディング: `maturin` + `pyo3`
