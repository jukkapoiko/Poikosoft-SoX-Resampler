/* SoX Resampler Library      Copyright (c) 2007-16 robs@users.sourceforge.net
 * Licence for this file: LGPL v2.1                  See LICENCE for details. */

/*
 * SoXR-Poikosoft | Enhanced SoX Resampler Library
 *
 * Poikosoft integration and enhancements Copyright (c) 2026
 * Jukka Poikolainen / Poikosoft | https://www.poikosoft.com
 *
 * Module: PFFFT to SoXR FFT adapter
 * Features:
 *  - PFFFT transform and spectral-convolution wrappers.
 *  - Maps the PFFFT spectrum layout onto SoXR FFT callbacks.
 *
 * Original third-party copyrights and license terms are preserved.
 */



#if !defined PFFT_MACROS_ONLY

#include "math-wrap.h"

#if PFFFT_DOUBLE
  #include "util64s.h"
#else
  #include "util32s.h"
  #define sin(x) sinf(x)
  #define cos(x) cosf(x)
#endif

#define pffft_aligned_free    SIMD_ALIGNED_FREE
#define pffft_aligned_malloc  SIMD_ALIGNED_MALLOC
#define pffft_aligned_calloc  SIMD_ALIGNED_CALLOC

#undef inline
#define inline __inline

#endif



#define SOXR_PFFFT_INCLUDED_BY_WRAPPER 1
#include "pffft.c"
#undef SOXR_PFFFT_INCLUDED_BY_WRAPPER



#if !defined PFFT_MACROS_ONLY

#if !defined PFFFT_SIMD_DISABLE

/*
 * Float32 wide-spectrum multiply for AVX/AVX2 backends.
 * PFFFT's radix FFT still uses v4sf (__m128) and its established four-lane
 * layout. Adjacent 4-float blocks are packed into a single 8-float __m256
 * only for spectral convolution. No change to transform, coefficient order,
 * FFT normalization, or the SSE backend. The two real-valued endpoint bins
 * are restored separately, matching the original implementation.
 * AVX2 builds also require FMA3 at runtime through pffft-runtime.c.
 */
#if defined(SOXR_PFFFT_WIDE_CONVOLVE) && SOXR_PFFFT_WIDE_CONVOLVE && !PFFFT_DOUBLE
#include <immintrin.h>
static void pffft_zconvolve(PFFFT_Setup * s, const float * a, const float * b, float * ab) {
  const int nc = s->Ncvec;
  const float dc_a = a[0], ny_a = a[4];
  const float dc_b = b[0], ny_b = b[4];
  int i;
  assert((nc & 1) == 0);
  assert(VALIGNED(a) && VALIGNED(b) && VALIGNED(ab));
  for(i = 0; i < nc; i += 2) {
    /* Each pair has [real(4),imag(4),real(4),imag(4)]. */
    __m256 a0 = _mm256_load_ps(a + 8 * i), a1 = _mm256_load_ps(a + 8 * i + 8);
    __m256 b0 = _mm256_load_ps(b + 8 * i), b1 = _mm256_load_ps(b + 8 * i + 8);
    __m256 ar = _mm256_permute2f128_ps(a0, a1, 0x20);
    __m256 ai = _mm256_permute2f128_ps(a0, a1, 0x31);
    __m256 br = _mm256_permute2f128_ps(b0, b1, 0x20);
    __m256 bi = _mm256_permute2f128_ps(b0, b1, 0x31);
    __m256 real, imag;
#if defined(SOXR_PFFFT_USE_FMA) && SOXR_PFFFT_USE_FMA
    real = _mm256_fmsub_ps(ar, br, _mm256_mul_ps(ai, bi));
    imag = _mm256_fmadd_ps(ai, br, _mm256_mul_ps(ar, bi));
#else
    real = _mm256_sub_ps(_mm256_mul_ps(ar, br), _mm256_mul_ps(ai, bi));
    imag = _mm256_add_ps(_mm256_mul_ps(ai, br), _mm256_mul_ps(ar, bi));
#endif
    _mm256_store_ps(ab + 8 * i, _mm256_permute2f128_ps(real, imag, 0x20));
    _mm256_store_ps(ab + 8 * i + 8, _mm256_permute2f128_ps(real, imag, 0x31));
  }
  if(s->transform == PFFFT_REAL) {
    ab[0] = dc_a * dc_b;
    ab[4] = ny_a * ny_b;
  }
}
#else
static void pffft_zconvolve(PFFFT_Setup * s, const float * a, const float * b, float * ab) {
  int i, Ncvec = s->Ncvec;
  const v4sf * /*RESTRICT*/ va = (const v4sf*)a;
  const v4sf * RESTRICT vb = (const v4sf *) b;
  v4sf * /*RESTRICT*/ vab = (v4sf*)ab;

  float ar, ai, br, bi;

#ifdef __arm__
  __builtin_prefetch(va);
  __builtin_prefetch(vb);
  __builtin_prefetch(va + 2);
  __builtin_prefetch(vb + 2);
  __builtin_prefetch(va + 4);
  __builtin_prefetch(vb + 4);
  __builtin_prefetch(va + 6);
  __builtin_prefetch(vb + 6);
#endif

  assert(VALIGNED(a) && VALIGNED(b) && VALIGNED(ab));
  ar = ((v4sf_union *) va)[0].f[0];
  ai = ((v4sf_union *) va)[1].f[0];
  br = ((v4sf_union *) vb)[0].f[0];
  bi = ((v4sf_union *) vb)[1].f[0];

  for(i = 0; i < Ncvec; i += 2) {
    v4sf ar, ai, br, bi;
    ar = va[2 * i + 0]; ai = va[2 * i + 1];
    br = vb[2 * i + 0]; bi = vb[2 * i + 1];
    VCPLXMUL(ar, ai, br, bi);
    vab[2 * i + 0] = ar;
    vab[2 * i + 1] = ai;
    ar = va[2 * i + 2]; ai = va[2 * i + 3];
    br = vb[2 * i + 2]; bi = vb[2 * i + 3];
    VCPLXMUL(ar, ai, br, bi);
    vab[2 * i + 2] = ar;
    vab[2 * i + 3] = ai;
  }
  if(s->transform == PFFFT_REAL) {
    ((v4sf_union *) vab)[0].f[0] = ar * br;
    ((v4sf_union *) vab)[1].f[0] = ai * bi;
  }
}

#endif /* SOXR_PFFFT_WIDE_CONVOLVE */

#else

static void pffft_zconvolve(PFFFT_Setup * s, const float * a, const float * b, float * ab) {
  int i, Ncvec = s->Ncvec;

  if(s->transform == PFFFT_REAL) {
    /* take care of the fftpack ordering */
    ab[0] = a[0] * b[0];
    ab[2 * Ncvec - 1] = a[2 * Ncvec - 1] * b[2 * Ncvec - 1];
    ++ab; ++a; ++b; --Ncvec;
  }
  for(i = 0; i < Ncvec; ++i) {
    float ar, ai, br, bi;
    ar = a[2 * i + 0]; ai = a[2 * i + 1];
    br = b[2 * i + 0]; bi = b[2 * i + 1];
    VCPLXMUL(ar, ai, br, bi);
    ab[2 * i + 0] = ar;
    ab[2 * i + 1] = ai;
  }
}

#endif

#include <string.h>

static void pffft_reorder_back(int length, void * setup, float * data, float * work)
{
  memcpy(work, data, (size_t) length * sizeof(*work));
  pffft_zreorder(setup, work, data, PFFFT_BACKWARD);
}

#endif
