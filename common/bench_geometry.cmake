# Region sizes of every benchmark of this repository (common/bench_geometry.h has the rest of the workload). Included by
# every CMake project; add BENCH_GEOMETRY_DEFINITIONS to the compile definitions of every target.
#   mixed   20x10, 32x16 and 48x24 by turns (region i of iteration t has the size (i + t) mod 3), the paper's workload
#   20x10, 32x16 or 48x24: every region of that size (the NPPDx benchmarks, whose kernels are compiled for one size)
set(BENCH_REGIONS mixed CACHE STRING "Region sizes: mixed (20x10, 32x16 and 48x24 by turns), 20x10, 32x16 or 48x24")
set_property(CACHE BENCH_REGIONS PROPERTY STRINGS mixed 20x10 32x16 48x24)
set(_benchRegionIndex_mixed -1)
set(_benchRegionIndex_20x10 0)
set(_benchRegionIndex_32x16 1)
set(_benchRegionIndex_48x24 2)
if(NOT DEFINED _benchRegionIndex_${BENCH_REGIONS})
    message(FATAL_ERROR "BENCH_REGIONS must be mixed, 20x10, 32x16 or 48x24, not ${BENCH_REGIONS}")
endif()
set(BENCH_GEOMETRY_DEFINITIONS BENCH_REGIONS=${_benchRegionIndex_${BENCH_REGIONS}})
message(STATUS "Benchmark regions: ${BENCH_REGIONS}, resized to 64x32")
