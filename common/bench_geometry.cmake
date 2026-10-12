# Crop and resize sizes of every benchmark of this repository (common/bench_geometry.h derives the rest from them).
# Included by every CMake project; add BENCH_GEOMETRY_DEFINITIONS to the compile definitions of every target.
set(BENCH_CROP_W 32 CACHE STRING "Width of every crop in pixels (even)")
set(BENCH_CROP_H 16 CACHE STRING "Height of every crop in pixels (even)")
set(BENCH_RESIZE_W 64 CACHE STRING "Width every crop is resized to; the output is BENCH_RESIZE_W x BENCH_RESIZE_W")
set(BENCH_RESIZE_H 32 CACHE STRING "Height every crop is resized to, before the black rows on top and bottom")
set(BENCH_GEOMETRY_DEFINITIONS BENCH_CROP_W=${BENCH_CROP_W} BENCH_CROP_H=${BENCH_CROP_H}
                               BENCH_RESIZE_W=${BENCH_RESIZE_W} BENCH_RESIZE_H=${BENCH_RESIZE_H})
message(STATUS "Benchmark geometry: ${BENCH_CROP_W}x${BENCH_CROP_H} crops resized to ${BENCH_RESIZE_W}x${BENCH_RESIZE_H}")
