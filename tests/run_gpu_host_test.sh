#!/bin/sh
# Builds GpuFilter against Qt stand-ins and a mock OpenCL driver, then runs it.
set -e
here=$(cd "$(dirname "$0")" && pwd)
root="$here/.."
out=${TMPDIR:-/tmp}/gk_host_build
mkdir -p "$out"
for f in "$root"/cubiomes/*.c; do
  case "$f" in */tests.c|*/main.c) continue;; esac
  gcc -O2 -DSTRUCT_CONFIG_OVERRIDE=1 -fwrapv -c "$f" -o "$out/$(basename "$f").o"
done
g++ -O2 -std=c++17 -shared -fPIC -o "$out/libmockcl.so" "$here/mock_opencl.cpp"
g++ -O2 -std=c++17 -DSTRUCT_CONFIG_OVERRIDE=1 -I"$here/qtstub" -I"$root" -I"$root/src" \
    -o "$out/gk_host_test" "$here/gpu_host_test.cpp" "$root/src/gpusearch.cpp" "$out"/*.o -ldl -lm
GK_MOCK_CL="$out/libmockcl.so" GK_KERNEL_PATH="$root/src/gpukernel.cl" "$out/gk_host_test"
