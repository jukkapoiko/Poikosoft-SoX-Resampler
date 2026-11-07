/* Per-instance CPU/FFT dispatcher: no global state, no atomics. */

/*
 * SoXR-Poikosoft | Enhanced SoX Resampler Library
 *
 * Poikosoft integration and enhancements Copyright (c) 2026
 * Jukka Poikolainen / Poikosoft | https://www.poikosoft.com
 *
 * Module: FFT dispatch API
 * Features:
 *  - Internal selector API for per-engine FFT callback tables.
 *
 * Original third-party copyrights and license terms are preserved.
 */

#ifndef SOXR_PFFFT_RUNTIME_H
#define SOXR_PFFFT_RUNTIME_H
#include "rdft_t.h"
/* Returns 0 on success, 1 if forced USE_SIMD is unsupported. */
int soxr_fft_pick_backend(int is_double, soxr_rdft_callbacks_t const **result);
int soxr_cpu_has_avx2_fma3(void);
int soxr_cpu_has_avx(void);
int soxr_cpu_has_sse2(void);
#endif
