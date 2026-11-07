/* Poikosoft EXTREME SSE2 fallback. Full DD precision; no FMA3 or AVX.
 * The DD complex multiply and division are unchanged from scalar.
 * SSE2 operates on both real and imaginary DD add/sub results together.
 * Immutable twiddles are owned by the caller's resampler instance.
 */
#include <stdint.h>
#include <stddef.h>
#include <emmintrin.h>
#include "soxr-config.h"
#include "soxr-dd-fft.h"

static soxr_dd_complex soxr_dd_complex_mul_scalar(soxr_dd_complex a,
                                                  soxr_dd_complex b)
{
  soxr_dd_complex r;
  r.re = soxr_dd_sub(soxr_dd_mul(a.re, b.re), soxr_dd_mul(a.im, b.im));
  r.im = soxr_dd_add(soxr_dd_mul(a.re, b.im), soxr_dd_mul(a.im, b.re));
  return r;
}

/* Same evaluation sequence as soxr_dd_add() for two independent numbers. */
static void dd_complex_butterfly_sse2(soxr_dd_complex * up,
                                     soxr_dd_complex * down,
                                     soxr_dd_complex v)
{
  soxr_dd_complex const u = *up;
  __m128d uh = _mm_set_pd(u.im.hi, u.re.hi);
  __m128d ul = _mm_set_pd(u.im.lo, u.re.lo);
  __m128d vh = _mm_set_pd(v.im.hi, v.re.hi);
  __m128d vl = _mm_set_pd(v.im.lo, v.re.lo);
  __m128d s = _mm_add_pd(uh, vh);
  __m128d w = _mm_sub_pd(s, uh);
  __m128d e = _mm_add_pd(_mm_sub_pd(uh, _mm_sub_pd(s, w)), _mm_sub_pd(vh, w));
  __m128d t = _mm_add_pd(_mm_add_pd(ul, vl), e);
  __m128d h = _mm_add_pd(s, t);
  __m128d l = _mm_sub_pd(t, _mm_sub_pd(h, s));
  _mm_storel_pd(&up->re.hi, h);
  _mm_storeh_pd(&up->im.hi, h);
  _mm_storel_pd(&up->re.lo, l);
  _mm_storeh_pd(&up->im.lo, l);
  /* Difference uses soxr_dd_sub() -> negate second input, then add. */
  vh = _mm_xor_pd(vh, _mm_set1_pd(-0.0));
  vl = _mm_xor_pd(vl, _mm_set1_pd(-0.0));
  s = _mm_add_pd(uh, vh);
  w = _mm_sub_pd(s, uh);
  e = _mm_add_pd(_mm_sub_pd(uh, _mm_sub_pd(s, w)), _mm_sub_pd(vh, w));
  t = _mm_add_pd(_mm_add_pd(ul, vl), e);
  h = _mm_add_pd(s, t);
  l = _mm_sub_pd(t, _mm_sub_pd(h, s));
  _mm_storel_pd(&down->re.hi, h);
  _mm_storeh_pd(&down->im.hi, h);
  _mm_storel_pd(&down->re.lo, l);
  _mm_storeh_pd(&down->im.lo, l);
}

int soxr_dd_fft_sse2_planned(soxr_dd_complex * a, size_t n, int inverse,
                             soxr_dd_scalar_plan const * plan)
{
  size_t len, stage_index = 0;
  if(!a || !plan || n < 2 || (n & (n - 1)) ||
     soxr_dd_scalar_plan_length(plan) != n) return -1;
  for(size_t i = 1, j = 0; i < n; ++i) {
    size_t bit = n >> 1;
    for(; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if(i < j) { soxr_dd_complex t = a[i]; a[i] = a[j]; a[j] = t; }
  }
  for(len = 2; len <= n; len <<= 1) {
    soxr_dd_complex const * twiddles = soxr_dd_scalar_plan_stage(plan, stage_index++, inverse);
    for(size_t i = 0; i < n; i += len) {
      for(size_t j = 0; j < len / 2; ++j) {
        soxr_dd_complex v = soxr_dd_complex_mul_scalar(a[i + j + len / 2], twiddles[j]);
        dd_complex_butterfly_sse2(&a[i + j], &a[i + j + len / 2], v);
      }
    }
    if(len == n) break;
  }
  if(inverse) {
    for(size_t i = 0; i < n; ++i) {
      a[i].re = soxr_dd_div(a[i].re, soxr_dd_from((double) n));
      a[i].im = soxr_dd_div(a[i].im, soxr_dd_from((double) n));
    }
  }
  return 0;
}
