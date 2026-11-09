#!/usr/bin/env bash
set -e

# スクリプトが置かれているディレクトリを取得
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# プロジェクトのルートへ移動
cd "$PROJECT_DIR"

# uv仮想環境内のNVIDIAライブラリを探索
NVIDIA_LIBS=$(find "$PROJECT_DIR/.venv/lib/python3.12/site-packages/nvidia" \
    -type d -name lib | paste -sd: -)

# CUDAライブラリの探索パスを設定
export LD_LIBRARY_PATH="$NVIDIA_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# uvでPythonプログラムを実行
exec uv run python "$@"