# Regenerates the Fern code of the benchmark (fern/generated/) from the gern submodule, at its pinned commit, and checks
# that it is the checked-in copy, or replaces the copy with WRITE=ON:
#
#   cmake [-DCXX=<C++ compiler>] [-DWORK=<build folder>] [-DWRITE=ON] -P fkl-vs-fern-cpu/fern/regenerate.cmake
#
# It builds gern (CMake 3.30 or newer, a C++20 compiler; Linux or macOS: gern's runner, which the library includes,
# uses dlfcn.h) without its tests, then fern_pipeline_gen.cpp against it, and runs it. gern at d05bc66 calls std::find
# in src/codegen/codegen.cpp without including <algorithm>, which libstdc++ 15 no longer includes indirectly: the
# header is forced on every gern source (-include algorithm) instead of patching gern.
#
# The generated text is the same for every compiler but for the numbers in the names of Fern's temporaries
# (_gern_x_25_41, _query_frame_44), which come from counters of the generator's run: the comparison renames them in
# order of appearance first.
cmake_minimum_required(VERSION 3.30)

get_filename_component(HERE "${CMAKE_CURRENT_LIST_DIR}" ABSOLUTE)
get_filename_component(REPO "${HERE}/../.." ABSOLUTE)
set(GERN "${REPO}/gern")
if(NOT CXX)
    set(CXX c++)
endif()
if(NOT WORK)
    set(WORK "${REPO}/build-fern-regenerate")
endif()
if(NOT EXISTS "${GERN}/CMakeLists.txt")
    message(FATAL_ERROR "The gern submodule is not checked out: git submodule update --init gern")
endif()
execute_process(COMMAND git -C "${GERN}" rev-parse --short HEAD OUTPUT_VARIABLE gernCommit
                OUTPUT_STRIP_TRAILING_WHITESPACE)
message(STATUS "gern ${gernCommit}, compiler ${CXX}, build in ${WORK}")

function(run)
    execute_process(COMMAND ${ARGN} RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE output)
    if(NOT result STREQUAL "0")
        string(REPLACE ";" " " command "${ARGN}")
        message(FATAL_ERROR "${command} failed (${result}):\n${output}")
    endif()
endfunction()

run("${CMAKE_COMMAND}" -S "${GERN}" -B "${WORK}/gern" -DCMAKE_BUILD_TYPE=Release -DGern_BUILD_TESTING=OFF
    "-DCMAKE_CXX_COMPILER=${CXX}" "-DCMAKE_CXX_FLAGS=-include algorithm")
run("${CMAKE_COMMAND}" --build "${WORK}/gern" --parallel)
file(GLOB_RECURSE gernLibrary "${WORK}/gern/libGern.a")
if(NOT gernLibrary)
    message(FATAL_ERROR "No libGern.a in ${WORK}/gern")
endif()
list(GET gernLibrary 0 gernLibrary)
run("${CXX}" -std=c++20 -O2 "-I${GERN}/include/gern" "${HERE}/fern_pipeline_gen.cpp" "${gernLibrary}" -ldl
    -o "${WORK}/fern_pipeline_gen")
file(REMOVE_RECURSE "${WORK}/generated")
file(MAKE_DIRECTORY "${WORK}/generated")
run("${WORK}/fern_pipeline_gen" "${WORK}/generated")

# The text with Fern's temporaries renamed in order of appearance
function(normalized FILE RESULT)
    file(READ "${FILE}" text)
    string(REGEX MATCHALL "_(gern|query)_[A-Za-z0-9_]*[0-9]" names "${text}")
    list(REMOVE_DUPLICATES names)
    set(i 0)
    foreach(name IN LISTS names)
        string(REGEX REPLACE "${name}([^A-Za-z0-9_])" "@fern_temporary_${i}@\\1" text "${text}")
        math(EXPR i "${i} + 1")
    endforeach()
    set(${RESULT} "${text}" PARENT_SCOPE)
endfunction()

file(GLOB regenerated RELATIVE "${WORK}/generated" "${WORK}/generated/*.cpp")
file(GLOB checkedIn RELATIVE "${HERE}/generated" "${HERE}/generated/*.cpp")
if(WRITE)
    file(REMOVE ${checkedIn})
    foreach(name IN LISTS regenerated)
        file(COPY_FILE "${WORK}/generated/${name}" "${HERE}/generated/${name}")
    endforeach()
    message(STATUS "Wrote ${regenerated} to ${HERE}/generated")
    return()
endif()
set(differences "")
foreach(name IN LISTS regenerated checkedIn)
    if(NOT EXISTS "${HERE}/generated/${name}" OR NOT EXISTS "${WORK}/generated/${name}")
        list(APPEND differences "${name} (only in one of them)")
        continue()
    endif()
    normalized("${WORK}/generated/${name}" new)
    normalized("${HERE}/generated/${name}" old)
    if(NOT new STREQUAL old)
        list(APPEND differences "${name}")
    endif()
endforeach()
list(REMOVE_DUPLICATES differences)
if(differences)
    message(FATAL_ERROR "The regenerated Fern code differs from fern/generated/ in: ${differences} (regenerated in "
                        "${WORK}/generated)")
endif()
message(STATUS "The Fern code regenerated with gern ${gernCommit} is the checked-in one (${regenerated})")
