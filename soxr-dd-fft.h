/*
 * SoXR-Poikosoft | Enhanced SoX Resampler Library
 *
 * Poikosoft integration and enhancements Copyright (c) 2026
 * Jukka Poikolainen / Poikosoft | https://www.poikosoft.com
 *
 * Module: Extreme FFT declarations
 * Features:
 *  - Double-double complex type and internal FFT interface.
 *
 * Original third-party copyrights and license terms are preserved.
 */

#ifndef SOXR_DD_FFT_H
#define SOXR_DD_FFT_H
#include "soxr-config.h"
#include <stddef.h>
typedef struct { soxr_dd_t re, im; } soxr_dd_complex;
/* In-place radix-2 FFT, inverse != 0 means normalized inverse FFT.
 * Data remain hi/lo throughout. No double intermediate audio buffers.
 * Returns 0 on success, -1 on invalid n. */
int soxr_dd_fft(soxr_dd_complex * a, size_t n, int inverse);
typedef struct soxr_dd_scalar_plan soxr_dd_scalar_plan;
soxr_dd_scalar_plan * soxr_dd_scalar_plan_create(size_t n);
size_t soxr_dd_scalar_plan_length(soxr_dd_scalar_plan const * plan);
soxr_dd_complex const * soxr_dd_scalar_plan_stage(soxr_dd_scalar_plan const * plan,
                                                  size_t stage, int inverse);
void soxr_dd_scalar_plan_destroy(soxr_dd_scalar_plan * plan);
int soxr_dd_fft_planned(soxr_dd_complex * a, size_t n, int inverse,
                         soxr_dd_scalar_plan const * plan);
int soxr_dd_fft_avx_planned(soxr_dd_complex * a, size_t n, int inverse,
                             soxr_dd_scalar_plan const * plan);
int soxr_dd_fft_sse2_planned(soxr_dd_complex * a, size_t n, int inverse,
                              soxr_dd_scalar_plan const * plan);
/* Shared only within one resampler; all plans are immutable after creation. */

/* AVX-only FFT butterflies live in soxr-dd-avx.c, AVX2/FMA3 kernels
 * live in soxr-dd-avx2.c; keep ISA flags isolated to those source files. */
int soxr_dd_fft_avx2(soxr_dd_complex * a, size_t n, int inverse);
void soxr_dd_convolve_avx2(soxr_dd_complex * x, soxr_dd_complex const * h, size_t n);
typedef struct soxr_dd_avx2_plan soxr_dd_avx2_plan;
struct stage;
void soxr_dd_poly0_pack_create(struct stage * p);
soxr_dd_avx2_plan * soxr_dd_avx2_plan_create(size_t n);
void soxr_dd_avx2_plan_destroy(soxr_dd_avx2_plan * p);
int soxr_dd_fft_avx2_planned(soxr_dd_complex * a, size_t n, int inverse,
                             soxr_dd_avx2_plan const * plan);
typedef int (*soxr_dd_fft_fn)(soxr_dd_complex *, size_t, int);
typedef void (*soxr_dd_convolve_fn)(soxr_dd_complex *, const soxr_dd_complex *, size_t);
#endif
