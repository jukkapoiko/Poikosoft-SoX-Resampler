# SoXR-Poikosoft | Enhanced SoX Resampler Library
# Poikosoft build integration (c) 2026 Jukka Poikolainen / Poikosoft
# https://www.poikosoft.com
# Module: Build configuration for scalar Ooura, SIMD PFFFT Fast and DD Extreme.
# Keep architecture-specific SIMD compilation flags local to backend files.
# The original source files and their license notices remain intact.

cmake_minimum_required(VERSION 3.18)
project(soxr LANGUAGES C)
option(BUILD_SHARED_LIBS "Shared library" ON)
set(SRC
  soxr.c data-io.c dbesi0.c filter.c cr.c cr32.c cr64.c cr64e.c cr64-fir-interp-pack.c cr64-fir-interp-sse2.c cr64-fir-interp-avx.c cr64-fir-interp-avx2.c soxr-dd-fft.c soxr-dd-sse2.c soxr-dd-avx.c soxr-dd-avx2.c
  fft4g32.c fft4g64.c pffft-runtime.c
  pffft-sse-f32.c pffft-sse2-f64.c pffft-avx-f32.c pffft-avx-f64.c
  pffft-avx2-f32.c pffft-avx2-f64.c)
add_library(soxr ${SRC})
target_compile_definitions(soxr PRIVATE SOXR_LIB=1)
target_include_directories(soxr PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(soxr PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED YES)
if(MSVC)
  if(CMAKE_SIZEOF_VOID_P EQUAL 4)
    set_source_files_properties(soxr-dd-sse2.c cr64-fir-interp-sse2.c pffft-sse2-f64.c PROPERTIES COMPILE_OPTIONS "/arch:SSE2")
  endif()
  target_compile_options(soxr PRIVATE /fp:strict)
else()
  if(CMAKE_SIZEOF_VOID_P EQUAL 4 AND CMAKE_SYSTEM_PROCESSOR MATCHES "i[3-6]86|x86")
    set_source_files_properties(soxr-dd-sse2.c cr64-fir-interp-sse2.c pffft-sse2-f64.c PROPERTIES COMPILE_OPTIONS "-msse2")
  endif()
  target_compile_options(soxr PRIVATE -fno-fast-math -ffp-contract=off)
  target_link_libraries(soxr PRIVATE m)
endif()
# GCC's intrinsics require per-file ISA flags (Clang/oneAPI use source targets).
if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
  set_source_files_properties(pffft-avx-f32.c pffft-avx-f64.c PROPERTIES COMPILE_OPTIONS "-mavx")
  set_source_files_properties(pffft-avx2-f32.c pffft-avx2-f64.c PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
endif()

# MSVC requires an explicit ISA option for each PFFFT translation unit.
# Keep AVX/AVX2 instructions out of the baseline/scalar source files.
if(MSVC)
  set_source_files_properties(pffft-avx-f32.c pffft-avx-f64.c PROPERTIES COMPILE_OPTIONS "/arch:AVX")
  set_source_files_properties(pffft-avx2-f32.c pffft-avx2-f64.c PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
endif()

# This dedicated translation unit must never make AVX2 a baseline requirement.
if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
  set_source_files_properties(soxr-dd-avx2.c PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
elseif(CMAKE_C_COMPILER_ID MATCHES "Clang" AND NOT MSVC)
  set_source_files_properties(soxr-dd-avx2.c PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
endif()

# SIMD interpreter in a separate file, with per-engine runtime selection.
if(MSVC)
  set_source_files_properties(cr64-fir-interp-avx2.c PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
elseif(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  set_source_files_properties(cr64-fir-interp-avx2.c PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
endif()

# The AVX-only interpolating FIR never requires AVX2 or FMA3.
if(MSVC)
  set_source_files_properties(cr64-fir-interp-avx.c PROPERTIES COMPILE_OPTIONS "/arch:AVX")
elseif(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  set_source_files_properties(cr64-fir-interp-avx.c PROPERTIES COMPILE_OPTIONS "-mavx;-mno-fma")
endif()

# EXTREME AVX-only backend: never compile FMA3 or AVX2 into this source.
if(MSVC)
  set_source_files_properties(soxr-dd-avx.c PROPERTIES COMPILE_OPTIONS "/arch:AVX")
elseif(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  set_source_files_properties(soxr-dd-avx.c PROPERTIES COMPILE_OPTIONS "-mavx;-mno-avx2;-mno-fma")
endif()
