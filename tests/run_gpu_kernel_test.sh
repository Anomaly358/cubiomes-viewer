#!/bin/sh
# Compiles cubiomes + the kernel-as-C++ harness and runs it.
set -e
here=$(cd "$(dirname "$0")" && pwd)
out=${TMPDIR:-/tmp}/gk_build
mkdir -p "$out"
for f in "$here"/../cubiomes/*.c; do
  case "$f" in */tests.c|*/main.c) continue;; esac
  gcc -O2 -c "$f" -o "$out/$(basename "$f").o"
done
g++ -O2 -std=c++17 -o "$out/gk_test" "$here/gpu_kernel_test.cpp" "$out"/*.o -lm
"$out/gk_test"
