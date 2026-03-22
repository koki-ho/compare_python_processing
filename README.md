# compare_python_processing

Python の数値計算実装を **Pure Python / NumPy / Numba / PyO3(Rust)** の 4 つのアプローチで比較するベンチマークプロジェクトです。

## 概要

以下の 3 種類の数値演算について、各実装の実行時間と Pure Python に対するスピードアップ倍率を計測・表示します。

| タスク | 内容 |
|---|---|
| `array_sum` | 大規模配列 (デフォルト: 1,000 万要素) の総和 |
| `matrix_dot` | 正方行列 (デフォルト: 128×128) の行列積 |
| `elementwise_sqrt` | 大規模配列の要素ごとの平方根 |

## ディレクトリ構成

```
compare_python_processing/
├── benchmark/
│   ├── runner.py                  # ベンチマーク実行・結果表示
│   └── implementations/
│       ├── pure_python.py         # 純粋 Python 実装
│       ├── numpy_impl.py          # NumPy 実装
│       └── numba_impl.py          # Numba JIT 実装
├── rust_ext/                      # PyO3 を使った Rust 拡張モジュール
│   ├── src/lib.rs
│   ├── Cargo.toml
│   └── pyproject.toml
├── configs/
│   └── config.yaml                # ベンチマーク設定 (配列サイズ・繰り返し回数 等)
├── tests/
│   └── test_sample.py
├── main.py                        # エントリーポイント
└── pyproject.toml
```

## セットアップ

### 前提条件

- Python 3.12+
- [uv](https://github.com/astral-sh/uv)
- Rust ツールチェーン (Rust 拡張を使う場合のみ)

### インストール

```bash
# 依存ライブラリのインストール
uv sync
```

### Rust 拡張のビルド (オプション)

PyO3/Rust 実装を含めてベンチマークしたい場合は、以下のコマンドで Rust 拡張をビルドしてください。

```bash
cd rust_ext
uv run maturin develop --release
cd ..
```

## 実行方法

```bash
uv run python main.py
```

設定ファイルを指定する場合:

```bash
uv run python main.py --config configs/config.yaml
```

### 出力例

```
Config: array_size=10,000,000  matrix_size=128  repeat=10
Warming up Numba JIT (first-call compilation)...

Benchmark: array_sum  (N=10,000,000)
╭─────────────┬───────────┬─────────╮
│ Method      │ Time (ms) │ Speedup │
├─────────────┼───────────┼─────────┤
│ Pure Python │    523.45 │    1.0x │
│ NumPy       │      5.12 │  102.2x │
│ Numba       │      4.87 │  107.5x │
│ PyO3/Rust   │      6.03 │   86.8x │
╰─────────────┴───────────┴─────────╯
```

## 設定

`configs/config.yaml` でベンチマークパラメータを変更できます。

```yaml
benchmark:
  array_size: 10000000  # 配列の要素数 (array_sum / elementwise_sqrt)
  matrix_size: 128      # 行列サイズ N (N×N の dot 積)
  repeat: 10            # 1タスクあたりの計測回数 (中央値を使用)
```

## テスト

```bash
uv run pytest
```

## 実装の比較

| 実装 | 特徴 |
|---|---|
| Pure Python | ループと標準ライブラリのみ。ベースライン |
| NumPy | C 実装のベクトル演算。大規模配列で高速 |
| Numba | `@njit` デコレータで Python コードを機械語にコンパイル |
| PyO3/Rust | Rust で実装し PyO3 で Python バインディングを生成 |
