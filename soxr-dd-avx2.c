/*
 * SoXR-Poikosoft | Enhanced SoX Resampler Library
 * Poikosoft AVX2/FMA3 Extreme double-double kernels (2026).
 * Copyright (c) 2026 Jukka Poikolainen / Poikosoft.
 * LGPL v2.1, see LICENSE. Derived from the adjacent scalar DD code.
 *
 * Four independent DD complex values per AVX2 vector. Low DD components
 * remain double precision; FMA is used for exact TwoProd residuals.
 * Build this translation unit with AVX2 and FMA3 only.
 */
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "soxr-config.h"
#include "internal.h"
#include "cr.h"
#include "soxr-dd-fft.h"
#include <immintrin.h>
#include <stdint.h>
#include <stddef.h>
#include <limits.h>

#if defined(__clang__)
  #pragma clang attribute push(__attribute__((target("avx2,fma"))), apply_to=function)
#endif

typedef struct { __m256d hi, lo; } dd4;
typedef struct { dd4 re, im; } z4;

/* Four adjacent DD outputs use two contiguous AVX stores. */
static inline void dd4_store(soxr_dd_t * p, size_t step, dd4 v) {
  if(step == 1) {
    __m256d a = _mm256_unpacklo_pd(v.hi, v.lo);
    __m256d b = _mm256_unpackhi_pd(v.hi, v.lo);
    __m256d c = _mm256_permute2f128_pd(a, b, 0x20);
    __m256d d = _mm256_permute2f128_pd(a, b, 0x31);
    memcpy(p, &c, sizeof(c));
    memcpy(p + 2, &d, sizeof(d));
    return;
  }
  double hi[4], lo[4];
  _mm256_storeu_pd(hi, v.hi);
  _mm256_storeu_pd(lo, v.lo);
  for(int j = 0; j < 4; ++j) {
    p[(size_t)j * step].hi = hi[j];
    p[(size_t)j * step].lo = lo[j];
  }
}
/* Never treat complex struct members as a flat array; pack from the
 * enclosing objects to avoid undefined pointer arithmetic across members. */
/* Convert four AoS DD complex values to four SIMD lanes with four
 * contiguous loads and a 4x4 register transpose (no scalar gathering).
 * memcpy copies whole objects, avoiding a pointer stepping across struct
 * members (a potential strict-aliasing / pointer-arithmetic violation). */
/* C89/C99-compatible compile-time layout validation; supported by MSVC C. */
typedef char soxr_dd_complex_size_check[
    sizeof(soxr_dd_complex) == sizeof(double) * 4 ? 1 : -1];
typedef char soxr_dd_complex_offset_check[
    offsetof(soxr_dd_complex, im) == 2 * sizeof(double) ? 1 : -1];
typedef char soxr_dd_low_component_offset_check[
    offsetof(soxr_dd_t, lo) == sizeof(double) ? 1 : -1];
static inline z4 z4_load(soxr_dd_complex const * p) {
  __m256d a;
  memcpy(&a, &p[0], sizeof(a));
  __m256d b;
  memcpy(&b, &p[1], sizeof(b));
  __m256d c;
  memcpy(&c, &p[2], sizeof(c));
  __m256d d;
  memcpy(&d, &p[3], sizeof(d));
  __m256d t0 = _mm256_unpacklo_pd(a, b);
  __m256d t1 = _mm256_unpackhi_pd(a, b);
  __m256d t2 = _mm256_unpacklo_pd(c, d);
  __m256d t3 = _mm256_unpackhi_pd(c, d);
  z4 z;
  z.re.hi = _mm256_permute2f128_pd(t0, t2, 0x20);
  z.re.lo = _mm256_permute2f128_pd(t1, t3, 0x20);
  z.im.hi = _mm256_permute2f128_pd(t0, t2, 0x31);
  z.im.lo = _mm256_permute2f128_pd(t1, t3, 0x31);
  return z;
}
static inline void z4_store(soxr_dd_complex * p, z4 z) {
  __m256d t0 = _mm256_permute2f128_pd(z.re.hi, z.im.hi, 0x20);
  __m256d t1 = _mm256_permute2f128_pd(z.re.lo, z.im.lo, 0x20);
  __m256d t2 = _mm256_permute2f128_pd(z.re.hi, z.im.hi, 0x31);
  __m256d t3 = _mm256_permute2f128_pd(z.re.lo, z.im.lo, 0x31);
  __m256d out0 = _mm256_unpacklo_pd(t0, t1);
  memcpy(&p[0], &out0, sizeof(out0));
  __m256d out1 = _mm256_unpackhi_pd(t0, t1);
  memcpy(&p[1], &out1, sizeof(out1));
  __m256d out2 = _mm256_unpacklo_pd(t2, t3);
  memcpy(&p[2], &out2, sizeof(out2));
  __m256d out3 = _mm256_unpackhi_pd(t2, t3);
  memcpy(&p[3], &out3, sizeof(out3));
}
/* In-place AoSoA blocks of 4 DD complex values. */
static inline z4 z4_soa_load(soxr_dd_complex const *p) {
  z4 z; memcpy(&z, p, sizeof(z)); return z;
}
static inline void z4_soa_store(soxr_dd_complex *p, z4 z) {
  memcpy(p, &z, sizeof(z));
}

static void fft_aosoa_to_aos(soxr_dd_complex *a, size_t n) {
  for(size_t i = 0; i < n; i += 4)
    z4_store(a + i, z4_soa_load(a + i));
}
static inline dd4 dd4_add(dd4 a, dd4 b) {
  dd4 r;
  __m256d s = _mm256_add_pd(a.hi, b.hi);
  __m256d v = _mm256_sub_pd(s, a.hi);
  __m256d e = _mm256_add_pd(
      _mm256_sub_pd(a.hi, _mm256_sub_pd(s, v)),
      _mm256_sub_pd(b.hi, v));
  __m256d t = _mm256_add_pd(_mm256_add_pd(a.lo, b.lo), e);
  r.hi = _mm256_add_pd(s, t);
  r.lo = _mm256_sub_pd(t, _mm256_sub_pd(r.hi, s));
  return r;
}
static inline dd4 dd4_neg(dd4 a) {
  dd4 r;
  __m256d const sign = _mm256_set1_pd(-0.0);
  r.hi = _mm256_xor_pd(a.hi, sign);
  r.lo = _mm256_xor_pd(a.lo, sign);
  return r;
}
static inline dd4 dd4_sub(dd4 a, dd4 b) { return dd4_add(a, dd4_neg(b)); }
static inline dd4 dd4_product(__m256d a, __m256d b) {
  dd4 p;
  p.hi = _mm256_mul_pd(a, b);
  p.lo = _mm256_fmsub_pd(a, b, p.hi);
  return p;
}
static inline dd4 dd4_mul(dd4 a, dd4 b) {
  dd4 p = dd4_product(a.hi, b.hi);
  p = dd4_add(p, dd4_product(a.hi, b.lo));
  p = dd4_add(p, dd4_product(a.lo, b.hi));
  return dd4_add(p, dd4_product(a.lo, b.lo));
}
/* Same DD quotient correction sequence as soxr_dd_div(), lane-by-lane.
 * No reciprocal approximation, no changed rounding sequence. */
static inline dd4 dd4_div_d(dd4 a, __m256d divisor) {
  dd4 b = {divisor, _mm256_setzero_pd()};
  dd4 v = {_mm256_div_pd(a.hi, divisor), _mm256_setzero_pd()};
  dd4 residual = dd4_sub(a, dd4_mul(b, v));
  dd4 first = {_mm256_div_pd(residual.hi, divisor), _mm256_setzero_pd()};
  v = dd4_add(v, first);
  residual = dd4_sub(a, dd4_mul(b, v));
  dd4 second = {_mm256_div_pd(residual.hi, divisor), _mm256_setzero_pd()};
  return dd4_add(v, second);
}
static inline z4 z4_div_d(z4 z, __m256d divisor) {
  z.re = dd4_div_d(z.re, divisor);
  z.im = dd4_div_d(z.im, divisor);
  return z;
}
static inline z4 z4_mul(z4 a, z4 b) {
  /* Expose all four independent complex DD products to the scheduler;
   * do not reassociate operations within any DD multiplication. */
  dd4 rr = dd4_mul(a.re,b.re);
  dd4 ii = dd4_mul(a.im,b.im);
  dd4 ri = dd4_mul(a.re,b.im);
  dd4 ir = dd4_mul(a.im,b.re);
  z4 r = {dd4_sub(rr,ii),dd4_add(ri,ir)};
  return r;
}
static inline z4 z4_add(z4 a, z4 b) {
  z4 r = {dd4_add(a.re, b.re), dd4_add(a.im, b.im)};
  return r;
}
static inline z4 z4_sub(z4 a, z4 b) {
  z4 r = {dd4_sub(a.re, b.re), dd4_sub(a.im, b.im)};
  return r;
}

/* First two radix-2 FFT stages are dominated by tiny scalar butterflies.
 * Fuse independent groups in four SIMD lanes without changing the operation
 * order *within* any butterfly, including the original complex DD multiply. */
static inline dd4 dd4_evengroups(dd4 a, dd4 b) {
  __m256d a02 = _mm256_permute4x64_pd(a.hi, 0x88);
  __m256d b02 = _mm256_permute4x64_pd(b.hi, 0x88);
  __m256d al02 = _mm256_permute4x64_pd(a.lo, 0x88);
  __m256d bl02 = _mm256_permute4x64_pd(b.lo, 0x88);
  dd4 z = {_mm256_permute2f128_pd(a02,b02,0x20),
           _mm256_permute2f128_pd(al02,bl02,0x20)};
  return z;
}
static inline dd4 dd4_oddgroups(dd4 a, dd4 b) {
  __m256d a13 = _mm256_permute4x64_pd(a.hi, 0xdd);
  __m256d b13 = _mm256_permute4x64_pd(b.hi, 0xdd);
  __m256d al13 = _mm256_permute4x64_pd(a.lo, 0xdd);
  __m256d bl13 = _mm256_permute4x64_pd(b.lo, 0xdd);
  dd4 z = {_mm256_permute2f128_pd(a13,b13,0x20),
           _mm256_permute2f128_pd(al13,bl13,0x20)};
  return z;
}
static inline dd4 dd4_zip_low(dd4 s, dd4 d) {
  __m256d sh = _mm256_unpacklo_pd(s.hi, d.hi);
  __m256d dh = _mm256_unpackhi_pd(s.hi, d.hi);
  __m256d sl = _mm256_unpacklo_pd(s.lo, d.lo);
  __m256d dl = _mm256_unpackhi_pd(s.lo, d.lo);
  dd4 z = {_mm256_permute2f128_pd(sh,dh,0x20),
           _mm256_permute2f128_pd(sl,dl,0x20)};
  return z;
}
static inline dd4 dd4_zip_high(dd4 s, dd4 d) {
  __m256d sh = _mm256_unpacklo_pd(s.hi, d.hi);
  __m256d dh = _mm256_unpackhi_pd(s.hi, d.hi);
  __m256d sl = _mm256_unpacklo_pd(s.lo, d.lo);
  __m256d dl = _mm256_unpackhi_pd(s.lo, d.lo);
  dd4 z = {_mm256_permute2f128_pd(sh,dh,0x31),
           _mm256_permute2f128_pd(sl,dl,0x31)};
  return z;
}
static inline z4 z4_evengroups(z4 a,z4 b) {
  z4 z={dd4_evengroups(a.re,b.re),dd4_evengroups(a.im,b.im)};
  return z;
}
static inline z4 z4_oddgroups(z4 a,z4 b) {
  z4 z={dd4_oddgroups(a.re,b.re),dd4_oddgroups(a.im,b.im)};
  return z;
}
static inline z4 z4_zip_low(z4 a,z4 b) {
  z4 z={dd4_zip_low(a.re,b.re),dd4_zip_low(a.im,b.im)};
  return z;
}
static inline z4 z4_zip_high(z4 a,z4 b) {
  z4 z={dd4_zip_high(a.re,b.re),dd4_zip_high(a.im,b.im)};
  return z;
}
static inline z4 z4_halves_lo(z4 a,z4 b) {
  z4 z = {
    {_mm256_permute2f128_pd(a.re.hi,b.re.hi,0x20),
     _mm256_permute2f128_pd(a.re.lo,b.re.lo,0x20)},
    {_mm256_permute2f128_pd(a.im.hi,b.im.hi,0x20),
     _mm256_permute2f128_pd(a.im.lo,b.im.lo,0x20)}
  };
  return z;
}
static inline z4 z4_halves_hi(z4 a,z4 b) {
  z4 z = {
    {_mm256_permute2f128_pd(a.re.hi,b.re.hi,0x31),
     _mm256_permute2f128_pd(a.re.lo,b.re.lo,0x31)},
    {_mm256_permute2f128_pd(a.im.hi,b.im.hi,0x31),
     _mm256_permute2f128_pd(a.im.lo,b.im.lo,0x31)}
  };
  return z;
}
static void fft_small_stage_2(soxr_dd_complex * a, size_t n) {
  z4 w;
  w.re.hi=_mm256_set1_pd(1.);
  w.re.lo=_mm256_setzero_pd();
  w.im.hi=_mm256_setzero_pd();
  w.im.lo=_mm256_setzero_pd();
  for(size_t i=0;i<n;i+=8) {
    z4 p=z4_load(a+i), q=z4_load(a+i+4);
    z4 u=z4_evengroups(p,q),v=z4_mul(z4_oddgroups(p,q),w);
    z4 s=z4_add(u,v),d=z4_sub(u,v);
    z4_store(a+i,z4_zip_low(s,d));
    z4_store(a+i+4,z4_zip_high(s,d));
  }
}
static void fft_small_stage_4(soxr_dd_complex * a,size_t n,
                               soxr_dd_complex const * wt, bool output_aosoa) {
  soxr_dd_complex tw[4]={wt[0],wt[1],wt[0],wt[1]};
  z4 w=z4_load(tw);
  for(size_t i=0;i<n;i+=8) {
    z4 p=z4_load(a+i),q=z4_load(a+i+4);
    z4 u=z4_halves_lo(p,q),v=z4_mul(z4_halves_hi(p,q),w);
    z4 s=z4_add(u,v),d=z4_sub(u,v);
    z4 low=z4_halves_lo(s,d),high=z4_halves_hi(s,d);
    if(output_aosoa) {
      z4_soa_store(a+i,low); z4_soa_store(a+i+4,high);
    } else {
      z4_store(a+i,low); z4_store(a+i+4,high);
    }
  }
}

static soxr_dd_complex cmul_scalar(soxr_dd_complex a, soxr_dd_complex b) {
  soxr_dd_complex r;
  r.re = soxr_dd_sub(soxr_dd_mul(a.re, b.re), soxr_dd_mul(a.im, b.im));
  r.im = soxr_dd_add(soxr_dd_mul(a.re, b.im), soxr_dd_mul(a.im, b.re));
  return r;
}

static soxr_dd_t dd_pi(void) {
  return soxr_dd_make(3.141592653589793116, 1.2246467991473532072e-16);
}
static void dd_sincos(soxr_dd_t x, soxr_dd_t * s, soxr_dd_t * c) {
  soxr_dd_t xx = soxr_dd_mul(x, x);
  soxr_dd_t st = x, ct = soxr_dd_from(1.);
  *s = st; *c = ct;
  for(int k = 1; k <= 45; ++k) {
    st = soxr_dd_mul(st, soxr_dd_div(soxr_dd_neg(xx),
         soxr_dd_from((double)(2 * k) * (2 * k + 1))));
    ct = soxr_dd_mul(ct, soxr_dd_div(soxr_dd_neg(xx),
         soxr_dd_from((double)(2 * k - 1) * (2 * k))));
    *s = soxr_dd_add(*s, st);
    *c = soxr_dd_add(*c, ct);
    if(st.hi == 0. && ct.hi == 0.) break;
  }
}

/* An immutable FFT twiddle plan belongs to one Extreme stage, not to
 * global state. Twiddles use the unchanged DD sin/cos and recurrence order.
 * Planning once avoids transcendental and DD recurrence work on every block. */
struct soxr_dd_avx2_plan {
  size_t n;
  size_t *swap_pairs; /* pairs of indices for immutable bit reversal */
  size_t swap_count;
  soxr_dd_complex *tw[2]; /* forward and inverse, stage offset = half - 1 */
  z4 *tw_simd[2]; /* contiguous packed 4-lane twiddles; offset = half/4 - 1 */
};

soxr_dd_avx2_plan * soxr_dd_avx2_plan_create(size_t n) {
  if(n < 2 || (n & (n - 1)) || n - 1 > SIZE_MAX / sizeof(soxr_dd_complex))
    return NULL;
  soxr_dd_avx2_plan * p = (soxr_dd_avx2_plan *)calloc(1, sizeof(*p));
  if(!p) return NULL;
  p->n = n;
  /* Stage bit reversal is independent of signal data. Precompute the
   * exact swap sequence; optional on allocation failure. */
  if(n <= SIZE_MAX / sizeof(size_t)) {
    p->swap_pairs = (size_t *)malloc(n * sizeof(size_t));
    if(p->swap_pairs) {
      for(size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1;
        for(; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if(i < j) {
          p->swap_pairs[2 * p->swap_count] = i;
          p->swap_pairs[2 * p->swap_count + 1] = j;
          ++p->swap_count;
        }
      }
    }
  }
  for(int inverse = 0; inverse < 2; ++inverse) {
    p->tw[inverse] = (soxr_dd_complex *)malloc((n - 1) * sizeof(soxr_dd_complex));
    if(!p->tw[inverse]) { soxr_dd_avx2_plan_destroy(p); return NULL; }
    for(size_t len = 2; len <= n; len <<= 1) {
      size_t const half = len / 2;
      soxr_dd_t theta = soxr_dd_div(soxr_dd_mul_d(dd_pi(), inverse ? 2. : -2.),
                                    soxr_dd_from((double)len));
      soxr_dd_complex wlen;
      dd_sincos(theta, &wlen.im, &wlen.re);
      soxr_dd_complex w = {{1., 0.}, {0., 0.}};
      for(size_t j = 0; j < half; ++j) {
        p->tw[inverse][half - 1 + j] = w;
        w = cmul_scalar(w, wlen);
      }
      if(len == n) break;
    }
    /* Pack each group of four immutable DD twiddles once. A failed
     * optional allocation is safe: the AoS plan remains usable. */
    if(n >= 8 && n / 4 - 1 <= SIZE_MAX / sizeof(z4)) {
      size_t count = n / 4 - 1;
      p->tw_simd[inverse] = (z4 *)_mm_malloc(count * sizeof(z4), 32);
      if(p->tw_simd[inverse]) {
        for(size_t half = 4; half <= n / 2; half <<= 1) {
          for(size_t j = 0; j < half; j += 4)
            p->tw_simd[inverse][half / 4 - 1 + j / 4] =
                z4_load(p->tw[inverse] + half - 1 + j);
          if(half == n / 2) break;
        }
      }
    }
  }
  return p;
}

void soxr_dd_avx2_plan_destroy(soxr_dd_avx2_plan * p) {
  if(!p) return;
  free(p->swap_pairs);
  free(p->tw[0]);
  free(p->tw[1]);
  _mm_free(p->tw_simd[0]);
  _mm_free(p->tw_simd[1]);
  free(p);
}

/* FFT is in-place. Only butterflies with independent j positions use SIMD;
 * the scalar stage twiddles are generated in the exact original order. */
int soxr_dd_fft_avx2_planned(soxr_dd_complex * a, size_t n, int inverse,
                             soxr_dd_avx2_plan const * plan) {
  if(!a || n == 0 || (n & (n - 1))) return -1;
  if(!plan || plan->n != n || !plan->tw[0] || !plan->tw[1]) plan = NULL;
  if(plan && plan->swap_pairs) {
    for(size_t k = 0; k < plan->swap_count; ++k) {
      size_t i = plan->swap_pairs[2 * k];
      size_t j = plan->swap_pairs[2 * k + 1];
      soxr_dd_complex t = a[i]; a[i] = a[j]; a[j] = t;
    }
  } else {
    for(size_t i = 1, j = 0; i < n; ++i) {
      size_t bit = n >> 1;
      for(; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if(i < j) { soxr_dd_complex t = a[i]; a[i] = a[j]; a[j] = t; }
    }
  }
  /* The same twiddles occur in every FFT block of one radix-2 stage.
   * Generate them once in precisely the scalar recurrence order, so no
   * double-double result is changed by sharing the immutable values. */
  soxr_dd_complex * twiddle = (!plan && n >= 16 && n / 4 <= SIZE_MAX / sizeof(soxr_dd_complex)) ?
      (soxr_dd_complex *)malloc((n / 4) * sizeof(*twiddle)) : NULL;
  bool const packed_fft = plan && n >= 32 && plan->tw_simd[!!inverse];
  for(size_t len = 2; len <= n; len <<= 1) {
    if(plan && n >= 8 && len == 2) {
      fft_small_stage_2(a,n);
      continue;
    }
    if(plan && n >= 8 && len == 4) {
      fft_small_stage_4(a,n,plan->tw[!!inverse] + 1, packed_fft);
      continue;
    }
    soxr_dd_complex wlen = {{1., 0.}, {0., 0.}};
    size_t half = len / 2;
    bool const use_plan = plan != NULL;
    if(!use_plan) {
      soxr_dd_t theta = soxr_dd_div(soxr_dd_mul_d(dd_pi(), inverse ? 2. : -2.),
                                    soxr_dd_from((double)len));
      dd_sincos(theta, &wlen.im, &wlen.re);
    }
    if(packed_fft && len >= 8) {
      /* stage 4 already stored the four-complex AoSoA layout. */
      z4 const *const tw4 = plan->tw_simd[!!inverse] + half / 4 - 1;
      for(size_t i = 0; i < n; i += len)
        for(size_t j = 0; j < half; j += 4) {
          z4 u = z4_soa_load(a + i + j);
          z4 v = z4_mul(z4_soa_load(a + i + j + half), tw4[j / 4]);
          z4_soa_store(a + i + j, z4_add(u,v));
          z4_soa_store(a + i + j + half, z4_sub(u,v));
        }
      if(len == n) break;
      continue;
    }
    bool const use_cache = use_plan || (twiddle != NULL && len < n && half >= 4);
    if(!use_plan && use_cache) {
      soxr_dd_complex w = {{1., 0.}, {0., 0.}};
      for(size_t j = 0; j < half; ++j) {
        twiddle[j] = w;
        w = cmul_scalar(w, wlen);
      }
    }
    soxr_dd_complex const *const wt = use_plan ? plan->tw[!!inverse] + half - 1 : twiddle;
    for(size_t i = 0; i < n; i += len) {
      size_t j = 0;
      soxr_dd_complex w = {{1., 0.}, {0., 0.}};
      for(; j + 4 <= half; j += 4) {
        z4 tw_v;
        if(use_plan && plan->tw_simd[!!inverse]) {
          tw_v = plan->tw_simd[!!inverse][half / 4 - 1 + j / 4];
        } else {
          soxr_dd_complex tw[4];
          if(use_cache) {
            memcpy(tw, wt + j, sizeof(tw));
          } else {
            for(int k = 0; k < 4; ++k) {
              tw[k] = w;
              w = cmul_scalar(w, wlen);
            }
          }
          tw_v = z4_load(tw);
        }
        z4 u = z4_load(a + i + j);
        z4 v = z4_mul(z4_load(a + i + j + half), tw_v);
        z4_store(a + i + j, z4_add(u, v));
        z4_store(a + i + j + half, z4_sub(u, v));
      }
      for(; j < half; ++j) {
        soxr_dd_complex use_w = use_cache ? wt[j] : w;
        soxr_dd_complex u = a[i + j];
        soxr_dd_complex v = cmul_scalar(a[i + j + half], use_w);
        a[i + j].re = soxr_dd_add(u.re, v.re);
        a[i + j].im = soxr_dd_add(u.im, v.im);
        a[i + j + half].re = soxr_dd_sub(u.re, v.re);
        a[i + j + half].im = soxr_dd_sub(u.im, v.im);
        if(!use_cache) w = cmul_scalar(w, wlen);
      }
    }
    if(len == n) break;
  }
  if(packed_fft && inverse) {
    /* Normalize and convert to the external AoS layout in a single pass.
     * Exactly the same DD divisions, only moved before the stores. */
    __m256d divisor = _mm256_set1_pd((double)n);
    for(size_t i = 0; i < n; i += 4)
      z4_store(a + i, z4_div_d(z4_soa_load(a + i), divisor));
  } else if(packed_fft) fft_aosoa_to_aos(a, n);
  free(twiddle);
  if(inverse && !packed_fft) {
    size_t i = 0;
    __m256d divisor = _mm256_set1_pd((double)n);
    for(; i + 4 <= n; i += 4)
      z4_store(a + i, z4_div_d(z4_load(a + i), divisor));
    for(; i < n; ++i) {
      a[i].re = soxr_dd_div(a[i].re, soxr_dd_from((double)n));
      a[i].im = soxr_dd_div(a[i].im, soxr_dd_from((double)n));
    }
  }
  return 0;
}

int soxr_dd_fft_avx2(soxr_dd_complex * a, size_t n, int inverse) {
  return soxr_dd_fft_avx2_planned(a, n, inverse, NULL);
}

void soxr_dd_convolve_avx2(soxr_dd_complex * x, soxr_dd_complex const * h, size_t n) {
  size_t i = 0;
  for(; i + 4 <= n; i += 4) {
    z4_store(x + i, z4_mul(z4_load(x + i), z4_load(h + i)));
  }
  for(; i < n; ++i) x[i] = cmul_scalar(x[i], h[i]);
}

/* Vectorize four independent Extreme FIR OUTPUTS rather than four taps of
 * one output. Consequently the summation order of every output is unchanged.
 * There is no horizontal floating-point reduction and no loss of DD precision.
 */
static inline dd4 dd4_pack(soxr_dd_t a, soxr_dd_t b, soxr_dd_t c, soxr_dd_t d) {
  dd4 v;
  v.hi = _mm256_set_pd(d.hi, c.hi, b.hi, a.hi);
  v.lo = _mm256_set_pd(d.lo, c.lo, b.lo, a.lo);
  return v;
}
static inline dd4 dd4_scalar(double x) {
  dd4 v;
  v.hi = _mm256_set1_pd(x);
  v.lo = _mm256_setzero_pd();
  return v;
}
static inline dd4 dd4_zero(void) { return dd4_scalar(0.); }
static inline dd4 dd4_values(soxr_dd_t const *const p[4], int j) {
  return dd4_pack(p[0][j],p[1][j],p[2][j],p[3][j]);
}
static inline void dd4_output(soxr_dd_t * p, dd4 x) { dd4_store(p, 1, x); }

void soxr_dd_half_avx2(struct stage * p, fifo_t * outfifo) {
  int n = min(stage_occupancy(p), p->input_size);
  int amount = (n + 1) >> 1;
  soxr_dd_t const * input = (soxr_dd_t const *) fifo_read_ptr(&p->fifo) + p->pre;
  soxr_dd_t * output = fifo_reserve(outfifo, amount);
  if(!output) { soxr_stage_oom(p); return; }
  double const * coefs = (double const *) p->coefs;
  int i = 0;
  for(; i + 4 <= amount; i += 4) {
    soxr_dd_t const * base = input + 2*i;
    dd4 sum = dd4_mul(dd4_pack(base[0],base[2],base[4],base[6]), dd4_scalar(.5));
    for(int j = 0; j < p->n; ++j) {
      int m = 2*j + 1;
      dd4 l = dd4_pack(base[-m],base[2-m],base[4-m],base[6-m]);
      dd4 r = dd4_pack(base[m],base[2+m],base[4+m],base[6+m]);
      sum = dd4_add(sum, dd4_mul(dd4_add(l,r), dd4_scalar(coefs[j])));
    }
    dd4_output(output + i, sum);
  }
  for(; i < amount; ++i) {
    soxr_dd_t const * in = input + 2*i;
    soxr_dd_t sum = soxr_dd_mul_d(in[0], .5);
    for(int j = 0; j < p->n; ++j) {
      soxr_dd_t pair = soxr_dd_add(in[-(2*j+1)],in[2*j+1]);
      sum = soxr_dd_add(sum, soxr_dd_mul_d(pair, coefs[j]));
    }
    output[i] = sum;
  }
  fifo_read(&p->fifo, 2*amount, NULL);
}

/* Optional per-stage packed coefficient cache. All DD arithmetic remains
 * exactly unchanged; prepacking moves the four-lane pack out of the tap loop.
 * _mm_malloc returns the required 32-byte alignment for dd4 vector loads. */
void soxr_dd_poly0_pack_destroy(void *ptr) { _mm_free(ptr); }
void soxr_dd_poly0_pack_create(struct stage *p) {
  if(p->extreme_poly0_pack_attempted) return;
  p->extreme_poly0_pack_attempted = true;
  if(!p->shared || !p->shared->poly_dd || p->L <= 0 || p->n <= 0) return;
  size_t const phases = (size_t)p->L, taps = (size_t)p->n;
  if(phases > 1048576 / sizeof(dd4) / taps) return;
  dd4 *plan = (dd4 *)_mm_malloc(phases*taps*sizeof(dd4),32);
  if(!plan) return;
  for(size_t phase=0; phase<phases; ++phase) {
    size_t r[4];
    for(int lane=0;lane<4;++lane)
      r[lane]= (phase + (size_t)lane*(size_t)p->step.integer)%phases;
    for(size_t j=0;j<taps;++j) {
      soxr_dd_t const *coef=p->shared->poly_dd;
      plan[phase*taps+j]=dd4_pack(coef[r[0]*taps+j],coef[r[1]*taps+j],
                                  coef[r[2]*taps+j],coef[r[3]*taps+j]);
    }
  }
  p->extreme_poly0_packed_coefs = plan;
}

void soxr_dd_poly0_avx2(struct stage * p, fifo_t * outfifo) {
  int n = min(stage_occupancy(p), p->input_size);
  if(!n) return;
  soxr_dd_t const * input = (soxr_dd_t const *) fifo_read_ptr(&p->fifo) + p->pre;
  int at = p->at.integer, step = p->step.integer;
  int amount = (n*p->L - at + step-1) / step;
  dd4 const * const packed=(dd4 const *)p->extreme_poly0_packed_coefs;
  soxr_dd_t * output = fifo_reserve(outfifo, amount);
  if(!output) { soxr_stage_oom(p); return; }
  int i = 0;
  for(; i + 4 <= amount; i += 4, at += 4*step) {
    soxr_dd_t const * in[4], *co[4];
    for(int lane = 0; lane < 4; ++lane) {
      int pos = at + lane*step;
      in[lane] = input + pos/p->L;
      co[lane] = p->shared->poly_dd + (size_t)p->n*(pos%p->L);
    }
    dd4 sum = dd4_zero();
    if(packed) {
      dd4 const *coeff=packed+(size_t)(at%p->L)*(size_t)p->n;
      for(int j = 0; j < p->n; ++j)
        sum = dd4_add(sum, dd4_mul(dd4_values(in,j),coeff[j]));
    } else {
      for(int j = 0; j < p->n; ++j)
        sum = dd4_add(sum,dd4_mul(dd4_values(in,j),dd4_values(co,j)));
    }
    dd4_output(output+i,sum);
  }
  for(; at < n * p->L; ++i, at += step) {
    int d = at/p->L, rem = at%p->L;
    soxr_dd_t const * in = input + d;
    soxr_dd_t const * co = p->shared->poly_dd + (size_t)p->n*rem;
    soxr_dd_t sum = soxr_dd_from(0);
    for(int j = 0; j < p->n; ++j)
      sum = soxr_dd_add(sum,soxr_dd_mul(in[j],co[j]));
    output[i] = sum;
  }
  fifo_read(&p->fifo, at/p->L, NULL);
  p->at.integer = at%p->L;
}

static inline void advance_clock(step_t *at, stage_t const * p) {
  if(p->use_hi_prec_clock) {
    uint64_t old = at->fix.ls.all;
    at->fix.ls.all += p->step.fix.ls.all;
    at->whole += p->step.whole + (at->fix.ls.all < old);
  } else at->whole += p->step.whole;
}
void soxr_dd_poly_interp_avx2(struct stage * p, fifo_t * outfifo, int order) {
  int n = min(stage_occupancy(p),p->input_size);
  int max_out = 1 + (int)(n*p->out_in_ratio);
  soxr_dd_t const * input = (soxr_dd_t const *)fifo_read_ptr(&p->fifo)+p->pre;
  soxr_dd_t * output = fifo_reserve(outfifo,max_out);
  if(!output) { soxr_stage_oom(p); return; }
  step_t at = p->at;
  int i = 0;
  while(at.integer < n) {
    step_t positions[4];
    int count = 0;
    while(count < 4 && at.integer < n) {
      positions[count++] = at;
      advance_clock(&at,p);
    }
    if(count == 4) {
      soxr_dd_t const * ins[4];
      int phases[4];
      soxr_dd_t xs[4];
      for(int lane = 0; lane < 4; ++lane) {
        uint32_t frac = positions[lane].fraction;
        phases[lane] = (int)(frac >> (32-p->phase_bits));
        uint32_t residue = frac << p->phase_bits;
        xs[lane] = soxr_dd_mul_d(soxr_dd_from((double)residue),1./4294967296.);
        ins[lane] = input + positions[lane].integer;
      }
      dd4 x = dd4_pack(xs[0],xs[1],xs[2],xs[3]);
      dd4 sum = dd4_zero();
      for(int j = 0; j < p->n; ++j) {
        soxr_dd_t co[4];
        for(int lane=0;lane<4;++lane)
          co[lane] = coef(p->shared->poly_dd,order,p->n,phases[lane],order,j);
        dd4 v = dd4_pack(co[0],co[1],co[2],co[3]);
        for(int k = order-1; k >= 0; --k) {
          for(int lane=0;lane<4;++lane)
            co[lane] = coef(p->shared->poly_dd,order,p->n,phases[lane],k,j);
          v = dd4_add(dd4_mul(v,x),dd4_pack(co[0],co[1],co[2],co[3]));
        }
        v = dd4_mul(v,dd4_values(ins,j));
        sum = dd4_add(sum,v);
      }
      dd4_output(output+i,sum);
      i += 4;
    } else {
      for(int lane=0;lane<count;++lane) {
        uint32_t frac=positions[lane].fraction;
        int phase=(int)(frac>>(32-p->phase_bits));
        uint32_t residue=frac<<p->phase_bits;
        soxr_dd_t x=soxr_dd_mul_d(soxr_dd_from((double)residue),1./4294967296.);
        soxr_dd_t sum=soxr_dd_from(0);
        soxr_dd_t const * in=input+positions[lane].integer;
        for(int j=0;j<p->n;++j) {
          soxr_dd_t v=coef(p->shared->poly_dd,order,p->n,phase,order,j);
          for(int k=order-1;k>=0;--k)
            v=soxr_dd_add(soxr_dd_mul(v,x),coef(p->shared->poly_dd,order,p->n,phase,k,j));
          sum=soxr_dd_add(sum,soxr_dd_mul(v,in[j]));
        }
        output[i++]=sum;
      }
    }
  }
  fifo_trim_by(outfifo,max_out-i);
  fifo_read(&p->fifo,at.integer,NULL);
  p->at.whole=at.fraction;
  if(p->use_hi_prec_clock)p->at.fix.ls=at.fix.ls;
}

#if defined(__clang__) && (defined(__i386__) || defined(__x86_64__) || defined(_M_X64))
  #pragma clang attribute pop
#endif
