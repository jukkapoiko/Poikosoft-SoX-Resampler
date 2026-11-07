/* pffft-sse2-f64: four-lane FFT using paired SSE2 doubles. */

/*
 * SoXR-Poikosoft | Enhanced SoX Resampler Library
 *
 * Poikosoft integration and enhancements Copyright (c) 2026
 * Jukka Poikolainen / Poikosoft | https://www.poikosoft.com
 *
 * Module: Float64 SSE2 PFFFT FFT backend
 * Features:
 *  - Four-double PFFFT layout with two SSE2 vectors; no AVX requirement.
 *
 * Original third-party copyrights and license terms are preserved.
 */


#define PFFFT_DOUBLE 1
#define SOXR_PFFFT_DOUBLE_SSE2 1
#define SOXR_PFFFT_USE_FMA 0
#include "filter.h"
#include "rdft_t.h"
#include "pffft-wrap.c"
static void * setup(int len) {
  return pffft_new_setup(len, PFFFT_REAL);
}
static void forward(int n, void * ctx, void * h, void * tmp) {
  (void) n;
  pffft_transform(ctx, h, h, tmp, PFFFT_FORWARD);
}
static void oforward(int n, void * ctx, void * h, void * tmp) {
  (void) n;
  pffft_transform_ordered(ctx, h, h, tmp, PFFFT_FORWARD);
}
static void backward(int n, void * ctx, void * h, void * tmp) {
  (void) n;
  pffft_transform(ctx, h, h, tmp, PFFFT_BACKWARD);
}
static void obackward(int n, void * ctx, void * h, void * tmp) {
  (void) n;
  pffft_transform_ordered(ctx, h, h, tmp, PFFFT_BACKWARD);
}
static void convolve(int n, void * ctx, void * h, const void * coef) {
  (void) n;
  pffft_zconvolve(ctx, h, coef, h);
}
static int multiplier(void) {
  return 1;
}
static int fft_no_oom(void) { return 0; }
static int flags(void) {
  return RDFT_NEEDS_SCRATCH | RDFT_IS_SIMD;
}
/* Match the generic FFT callback ABI exactly (void * argument). */
static void destroy_setup(void * p) { pffft_destroy_setup((PFFFT_Setup *)p); }
static void typed_partial(int n, void * h, void const * coefs)
{
  lsx_ordered_partial_convolve(n, (double *) h, (double const *) coefs);
}
static void typed_reorder(int n, void * setup, void * h, void * work)
{
  pffft_reorder_back(n, setup, (double *) h, (double *) work);
}
soxr_rdft_callbacks_t const _soxr_pffft_sse2_f64_cb = {
  setup, setup, destroy_setup, forward, oforward, backward, obackward,
  convolve, typed_partial, multiplier, typed_reorder,
  soxr_pffft_aligned_malloc, soxr_pffft_aligned_calloc,
  soxr_pffft_aligned_free, flags, fft_no_oom,
  NULL
};

