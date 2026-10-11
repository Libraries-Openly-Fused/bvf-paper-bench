# Source this before building or running the ROCm benchmarks:  . rocm/scripts/env.sh
# ROCM_PATH: the ROCm installation. ROCM_EXTRA: a prefix with RPP, rocprofiler-sdk and amd-smi when they are not
# installed in ROCM_PATH (for example the ROCm 10.1 .deb packages amdrocm-rpp10.1, amdrocm-rpp-dev10.1,
# amdrocm-profiler-base10.1, amdrocm-profiler10.1 and amdrocm-amdsmi10.1 unpacked with dpkg -x, without root).
export ROCM_PATH=${ROCM_PATH:-/opt/rocm/core-10.1}
export HIP_PATH=$ROCM_PATH
export ROCM_EXTRA=${ROCM_EXTRA:-$ROCM_PATH}
# librpp needs libomp (ROCm's LLVM) and the ROCm sysdeps libraries
export LD_LIBRARY_PATH=$ROCM_EXTRA/lib:$ROCM_PATH/lib:$ROCM_PATH/lib/rocm_sysdeps/lib:$ROCM_PATH/lib/llvm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PATH=$ROCM_EXTRA/bin:$ROCM_PATH/bin:$PATH
export PYTHONPATH=$ROCM_EXTRA/lib/python3/site-packages${PYTHONPATH:+:$PYTHONPATH}
