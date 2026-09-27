#!/usr/bin/env bash
# Make PyTorch use Homebrew's libomp so each process has ONE OpenMP runtime.
#
# Why: Homebrew OpenCV -> OpenBLAS -> Homebrew libomp, and pip torch bundles its own libomp.
# svo_env loads OpenCV, so `import torch, svo_env` gets two runtimes and aborts with "OMP: Error #15".
# torch loads libomp via @loader_path/libomp.dylib, so we replace that file with a symlink to Homebrew's copy
# (same OpenMP ABI 5.0, newer build). The original is kept as libomp.dylib.torch-orig.
#
# Re-run after every (re)install/upgrade of torch in .venv.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
TORCH_LIB="$("$PY" -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
BREW_OMP="$(brew --prefix)/opt/libomp/lib/libomp.dylib"
T="$TORCH_LIB/libomp.dylib"

if [ -L "$T" ] && [ "$(readlink "$T")" = "$BREW_OMP" ]; then
  echo "already linked: $T -> $BREW_OMP"; exit 0
fi
[ -e "$T.torch-orig" ] || cp -p "$T" "$T.torch-orig"
ln -sf "$BREW_OMP" "$T"
echo "linked: $T -> $BREW_OMP (backup: $T.torch-orig)"
