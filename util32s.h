/* Private backend helpers for 32-bit PFFFT. */

/*
 * SoXR-Poikosoft | Enhanced SoX Resampler Library
 *
 * Poikosoft integration and enhancements Copyright (c) 2026
 * Jukka Poikolainen / Poikosoft | https://www.poikosoft.com
 *
 * Module: Float32 PFFFT utilities
 * Features:
 *  - Aligned-memory helpers and partial convolution bindings.
 *
 * Original third-party copyrights and license terms are preserved.
 */


#include "pffft-backend-util.h"
#define SIMD_ALIGNED_MALLOC soxr_pffft_aligned_malloc
#define SIMD_ALIGNED_CALLOC soxr_pffft_aligned_calloc
#define SIMD_ALIGNED_FREE soxr_pffft_aligned_free
#define ORDERED_PARTIAL_CONVOLVE_SIMD lsx_ordered_partial_convolve_f
