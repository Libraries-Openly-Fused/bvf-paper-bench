/* Copyright 2026 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

// Description of the system for the platform files of every benchmark of this repository, on Linux and Windows

#ifndef BVF_BENCH_PLATFORM_H
#define BVF_BENCH_PLATFORM_H

#include <cstdio>
#include <sstream>
#include <string>

namespace bench {

// Output of a shell command without its trailing new lines and spaces, or "" if it cannot run
inline std::string shell(const std::string& command) {
    std::string output;
#ifdef _WIN32
    FILE* pipe = _popen(command.c_str(), "r");
#else
    FILE* pipe = popen(command.c_str(), "r");
#endif
    if (pipe) {
        char buffer[512];
        while (std::fgets(buffer, sizeof(buffer), pipe)) {
            output += buffer;
        }
#ifdef _WIN32
        _pclose(pipe);
#else
        pclose(pipe);
#endif
    }
    while (!output.empty() && (output.back() == '\n' || output.back() == '\r' || output.back() == ' ')) {
        output.pop_back();
    }
    return output;
}

inline std::string cpuName() {
#ifdef _WIN32
    return shell("powershell -NoProfile -Command \"(Get-CimInstance Win32_Processor).Name\"");
#else
    // One line per model name (several on big.LITTLE CPUs)
    return shell("lscpu | sed -n 's/^Model name: *//p' | paste -sd ','");
#endif
}

inline std::string osName() {
#ifdef _WIN32
    return shell("powershell -NoProfile -Command \"(Get-CimInstance Win32_OperatingSystem).Caption\"");
#else
    return shell(". /etc/os-release && echo $PRETTY_NAME");
#endif
}

inline std::string kernelVersion() {
#ifdef _WIN32
    return shell("powershell -NoProfile -Command \"(Get-CimInstance Win32_OperatingSystem).Version\"");
#else
    return shell("uname -r");
#endif
}

// The compiler of the host code, from its predefined macros (nvcc passes on the ones of its host compiler)
inline std::string hostCompiler() {
#if defined(__clang__)
    return std::string("Clang ") + __clang_version__;
#elif defined(_MSC_VER)
    return "MSVC " + std::to_string(_MSC_FULL_VER);
#elif defined(__GNUC__)
    return std::string("GCC ") + __VERSION__;
#else
    return "unknown";
#endif
}

// The SIMD extensions the host code was compiled for
inline std::string compiledSimd() {
    std::string simd;
    [[maybe_unused]] const auto add = [&simd](const char* name) { simd += simd.empty() ? name : std::string(" ") + name; };
#ifdef __SSE4_2__
    add("SSE4.2");
#endif
#ifdef __AVX__
    add("AVX");
#endif
#ifdef __AVX2__
    add("AVX2");
#endif
#ifdef __FMA__
    add("FMA");
#endif
#ifdef __AVX512F__
    add("AVX-512F");
#endif
#ifdef __AVX512BW__
    add("AVX-512BW");
#endif
#ifdef __AVX512VL__
    add("AVX-512VL");
#endif
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    add("NEON");
#endif
#ifdef __ARM_FEATURE_SVE
    add("SVE");
#endif
#ifdef __ARM_FEATURE_SVE2
    add("SVE2");
#endif
    return simd.empty() ? "none beyond the baseline" : simd;
}

// The value of one line of OpenCV's cv::getBuildInformation(), for example "C++ Compiler:" or "NVIDIA GPU arch:"
inline std::string buildInfoValue(const std::string& buildInformation, const std::string& key) {
    std::istringstream lines(buildInformation);
    for (std::string line; std::getline(lines, line);) {
        const size_t start = line.find_first_not_of(' ');
        if (start != std::string::npos && line.compare(start, key.size(), key) == 0) {
            const size_t value = line.find_first_not_of(' ', start + key.size());
            return value == std::string::npos ? "" : line.substr(value);
        }
    }
    return "unknown";
}

} // namespace bench

#endif // BVF_BENCH_PLATFORM_H
