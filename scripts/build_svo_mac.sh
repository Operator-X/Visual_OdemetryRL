#!/usr/bin/env bash
# Build third_party/svo-lib (incl. the svo_env Python module) natively on Apple Silicon.
# Prereqs: brew install cmake eigen@3 opencv@4 boost yaml-cpp libomp suite-sparse glew ; .venv with pybind11
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/third_party/svo-lib"
BUILD="$SRC/build"
PY="$ROOT/.venv/bin/python"
BREW="$(brew --prefix)"
OMP="$BREW/opt/libomp"
YAML="$BREW/opt/yaml-cpp"

cmake -S "$SRC" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DWITH_GFLAGS=OFF \
  -DEigen3_DIR="$BREW/opt/eigen@3/share/eigen3/cmake" \
  -DOpenCV_DIR="$BREW/opt/opencv@4/lib/cmake/opencv4" \
  -DPython_EXECUTABLE="$PY" \
  -Dpybind11_DIR="$("$PY" -m pybind11 --cmakedir)" \
  -DOpenMP_C_FLAGS="-Xpreprocessor -fopenmp -I$OMP/include" -DOpenMP_C_LIB_NAMES=omp \
  -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I$OMP/include" -DOpenMP_CXX_LIB_NAMES=omp \
  -DOpenMP_omp_LIBRARY="$OMP/lib/libomp.dylib" \
  -DCMAKE_CXX_FLAGS="-isystem $YAML/include -isystem $BREW/opt/boost/include" \
  -DCMAKE_SHARED_LINKER_FLAGS="-L$YAML/lib" \
  -DCMAKE_MODULE_LINKER_FLAGS="-L$YAML/lib" \
  -DCMAKE_EXE_LINKER_FLAGS="-L$YAML/lib"

make -C "$BUILD" -k -j"$(sysctl -n hw.ncpu)" "$@"
