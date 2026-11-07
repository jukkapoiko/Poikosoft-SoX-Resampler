/* Poikosoft EXTREME AVX1 FFT (no FMA3 or AVX2 instructions).
 * Two complex DD butterflies operate in parallel using four AVX lanes.
 * Products use Dekker error-free splitting to preserve all DD terms.
 * Requires only OS-supported AVX1; no FMA3 feature detection.
 */
#include <stddef.h>
#include <string.h>
#include <immintrin.h>
#include "soxr-dd-fft.h"
#include "internal.h"
#include "cr.h"

/* Keep ISA specialization local to this backend under Clang/Intel oneAPI.
 * Plain-C, SSE2 and dispatch files can still be compiled for baseline x64.
 * GCC users still need their normal per-file -m flags. */
#if defined(__clang__)
#pragma clang attribute push (__attribute__((target("avx"))), apply_to=function)
#endif

typedef struct { __m256d hi, lo; } dd4;

typedef char avx_complex_layout_check[
    sizeof(soxr_dd_complex) == 4 * sizeof(double) &&
    offsetof(soxr_dd_complex, im) == 2 * sizeof(double) &&
    offsetof(soxr_dd_t, lo) == sizeof(double) ? 1 : -1];

/* Lane order: [first.real, second.real, first.imag, second.imag].
 * Pure AVX1 cross-lane shuffles; no gather or AVX2 instructions. */
static inline dd4 dd4_load(soxr_dd_complex const * a,
                            soxr_dd_complex const * b)
{
  __m256d first, second;
  dd4 r;
  memcpy(&first, a, sizeof(first));
  memcpy(&second, b, sizeof(second));
  r.hi = _mm256_unpacklo_pd(first, second);
  r.lo = _mm256_unpackhi_pd(first, second);
  return r;
}

static inline void dd4_store(soxr_dd_complex * a,
                             soxr_dd_complex * b, dd4 x)
{
  __m256d first = _mm256_unpacklo_pd(x.hi, x.lo);
  __m256d second = _mm256_unpackhi_pd(x.hi, x.lo);
  memcpy(a, &first, sizeof(first));
  memcpy(b, &second, sizeof(second));
}

static inline dd4 dd4_add(dd4 a, dd4 b)
{
  dd4 r;
  __m256d s = _mm256_add_pd(a.hi, b.hi);
  __m256d v = _mm256_sub_pd(s, a.hi);
  __m256d e = _mm256_add_pd(_mm256_sub_pd(a.hi, _mm256_sub_pd(s, v)),
                             _mm256_sub_pd(b.hi, v));
  __m256d t = _mm256_add_pd(_mm256_add_pd(a.lo, b.lo), e);
  r.hi = _mm256_add_pd(s, t);
  r.lo = _mm256_sub_pd(t, _mm256_sub_pd(r.hi, s));
  return r;
}

static inline dd4 dd4_sub(dd4 a, dd4 b)
{
  dd4 neg;
  __m256d sign = _mm256_set1_pd(-0.0);
  neg.hi = _mm256_xor_pd(b.hi, sign);
  neg.lo = _mm256_xor_pd(b.lo, sign);
  return dd4_add(a, neg);
}

static inline dd4 dd4_product(__m256d a, __m256d b)
{
  /* Dekker TwoProd: no FMA3. Exact error for finite normal inputs
   * without exponent extremes; all four AVX lanes independent. */
  __m256d splitter = _mm256_set1_pd(134217729.0);
  __m256d ta = _mm256_mul_pd(a, splitter);
  __m256d tb = _mm256_mul_pd(b, splitter);
  __m256d ahi = _mm256_sub_pd(ta, _mm256_sub_pd(ta, a));
  __m256d bhi = _mm256_sub_pd(tb, _mm256_sub_pd(tb, b));
  __m256d alo = _mm256_sub_pd(a, ahi);
  __m256d blo = _mm256_sub_pd(b, bhi);
  dd4 p;
  p.hi = _mm256_mul_pd(a, b);
  __m256d e = _mm256_sub_pd(_mm256_mul_pd(ahi, bhi), p.hi);
  e = _mm256_add_pd(e, _mm256_mul_pd(ahi, blo));
  e = _mm256_add_pd(e, _mm256_mul_pd(alo, bhi));
  p.lo = _mm256_add_pd(e, _mm256_mul_pd(alo, blo));
  return p;
}

static inline dd4 dd4_mul(dd4 a, dd4 b)
{
  dd4 p = dd4_product(a.hi, b.hi);
  p = dd4_add(p, dd4_product(a.hi, b.lo));
  p = dd4_add(p, dd4_product(a.lo, b.hi));
  return dd4_add(p, dd4_product(a.lo, b.lo));
}

/* Complex multiplication exact DD order: rr - ii, ri + ir.
 * Real and imaginary for two complex values share the four SIMD lanes. */
static inline dd4 dd4_complex_mul(dd4 a, dd4 b)
{
  dd4 bre = {
    _mm256_permute2f128_pd(b.hi, b.hi, 0x00),
    _mm256_permute2f128_pd(b.lo, b.lo, 0x00)
  };
  dd4 bim = {
    _mm256_permute2f128_pd(b.hi, b.hi, 0x11),
    _mm256_permute2f128_pd(b.lo, b.lo, 0x11)
  };
  dd4 aswap = {
    _mm256_permute2f128_pd(a.hi, a.hi, 0x01),
    _mm256_permute2f128_pd(a.lo, a.lo, 0x01)
  };
  dd4 rr_ir = dd4_mul(a, bre);
  dd4 ii_ri = dd4_mul(aswap, bim);
  dd4 real = dd4_sub(rr_ir, ii_ri);
  dd4 imag = dd4_add(ii_ri, rr_ir);
  dd4 result = {
    _mm256_permute2f128_pd(real.hi, imag.hi, 0x30),
    _mm256_permute2f128_pd(real.lo, imag.lo, 0x30)
  };
  return result;
}

static inline soxr_dd_complex dd_complex_mul_scalar(soxr_dd_complex a,
                                                     soxr_dd_complex b)
{
  soxr_dd_complex result;
  result.re = soxr_dd_sub(soxr_dd_mul(a.re, b.re), soxr_dd_mul(a.im, b.im));
  result.im = soxr_dd_add(soxr_dd_mul(a.re, b.im), soxr_dd_mul(a.im, b.re));
  return result;
}

/* Dekker splitting needs normal, comfortably scaled inputs. Preserve the
 * original scalar double-double multiplication for exceptional magnitudes
 * (including subnormals, non-finite values and extreme exponent ranges).
 * This leaves ordinary audio on the accelerated AVX1-only path.
 */
static inline int dd4_needs_scalar(dd4 a, dd4 b)
{
  const __m256d sign = _mm256_set1_pd(-0.0);
  const __m256d lo = _mm256_set1_pd(1.e-100);
  const __m256d hi = _mm256_set1_pd(1.e100);
  const __m256d zero = _mm256_setzero_pd();
  __m256d values[4] = {a.hi, a.lo, b.hi, b.lo};
  for (int i = 0; i < 4; ++i) {
    __m256d x = _mm256_andnot_pd(sign, values[i]);
    __m256d too_small = _mm256_and_pd(_mm256_cmp_pd(x, zero, _CMP_NEQ_OQ),
                                       _mm256_cmp_pd(x, lo, _CMP_LT_OQ));
    __m256d too_large = _mm256_cmp_pd(x, hi, _CMP_NLE_UQ);
    if (_mm256_movemask_pd(_mm256_or_pd(too_small, too_large))) return 1;
  }
  return 0;
}

static inline void butterfly2(soxr_dd_complex * up0,
                              soxr_dd_complex * up1,
                              soxr_dd_complex * down0,
                              soxr_dd_complex * down1,
                              dd4 twiddles)
{
  dd4 u = dd4_load(up0, up1);
  dd4 lower = dd4_load(down0, down1);
  if (dd4_needs_scalar(lower, twiddles)) {
    /* Two independent scalar DD butterflies; original v14 arithmetic. */
    soxr_dd_complex w[2];
    dd4_store(&w[0], &w[1], twiddles);
    soxr_dd_complex v0 = dd_complex_mul_scalar(*down0, w[0]);
    soxr_dd_complex v1 = dd_complex_mul_scalar(*down1, w[1]);
    soxr_dd_complex s0 = {soxr_dd_add(up0->re, v0.re), soxr_dd_add(up0->im, v0.im)};
    soxr_dd_complex s1 = {soxr_dd_add(up1->re, v1.re), soxr_dd_add(up1->im, v1.im)};
    soxr_dd_complex d0 = {soxr_dd_sub(up0->re, v0.re), soxr_dd_sub(up0->im, v0.im)};
    soxr_dd_complex d1 = {soxr_dd_sub(up1->re, v1.re), soxr_dd_sub(up1->im, v1.im)};
    *up0 = s0; *up1 = s1; *down0 = d0; *down1 = d1;
    return;
  }
  dd4 v = dd4_complex_mul(lower, twiddles);
  dd4 sum = dd4_add(u, v);
  dd4 difference = dd4_sub(u, v);
  dd4_store(up0, up1, sum);
  dd4_store(down0, down1, difference);
}

static inline dd4 dd4_div_d(dd4 a, __m256d d)
{
  dd4 divisor = {d, _mm256_setzero_pd()};
  dd4 v = {_mm256_div_pd(a.hi, d), _mm256_setzero_pd()};
  dd4 residual = dd4_sub(a, dd4_mul(divisor, v));
  dd4 first = {_mm256_div_pd(residual.hi, d), _mm256_setzero_pd()};
  v = dd4_add(v, first);
  residual = dd4_sub(a, dd4_mul(divisor, v));
  dd4 second = {_mm256_div_pd(residual.hi, d), _mm256_setzero_pd()};
  return dd4_add(v, second);
}

int soxr_dd_fft_avx_planned(soxr_dd_complex * a, size_t n,
                                  int inverse, soxr_dd_scalar_plan const * plan)
{
  size_t stage_index = 0;
  if(!a || !plan || n < 2 || (n & (n - 1)) ||
     soxr_dd_scalar_plan_length(plan) != n) return -1;

  /* n=2 is not a real-world DFT stage; avoid a special SIMD tail. */
  if(n == 2) return soxr_dd_fft_planned(a, n, inverse, plan);

  for(size_t i = 1, j = 0; i < n; ++i) {
    size_t bit = n >> 1;
    for(; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if(i < j) { soxr_dd_complex t = a[i]; a[i] = a[j]; a[j] = t; }
  }

  for(size_t len = 2; len <= n; len <<= 1) {
    size_t half = len >> 1;
    soxr_dd_complex const * tw = soxr_dd_scalar_plan_stage(plan, stage_index++, inverse);
    if(len == 2) {
      dd4 unity = dd4_load(&tw[0], &tw[0]);
      for(size_t i = 0; i < n; i += 4)
        butterfly2(a + i, a + i + 2, a + i + 1, a + i + 3, unity);
    } else {
      for(size_t i = 0; i < n; i += len) {
        for(size_t j = 0; j < half; j += 2) {
          dd4 w = dd4_load(tw + j, tw + j + 1);
          butterfly2(a + i + j, a + i + j + 1,
                     a + i + j + half, a + i + j + half + 1, w);
        }
      }
    }
    if(len == n) break;
  }

  if(inverse) {
    __m256d d = _mm256_set1_pd((double) n);
    for(size_t i = 0; i < n; i += 2)
      dd4_store(a + i, a + i + 1, dd4_div_d(dd4_load(a + i, a + i + 1), d));
  }
  return 0;
}

/* AVX1-only four-output DD polyphase FIR: never uses FMA3 or AVX2.
 * Preserve each output's original tap accumulation order. */
static inline dd4 dd4_poly_pack(soxr_dd_t a, soxr_dd_t b,
                                soxr_dd_t c, soxr_dd_t d)
{
  dd4 v;
  v.hi = _mm256_set_pd(d.hi, c.hi, b.hi, a.hi);
  v.lo = _mm256_set_pd(d.lo, c.lo, b.lo, a.lo);
  return v;
}
static inline dd4 dd4_poly_get(soxr_dd_t const * const values[4], int j)
{
  return dd4_poly_pack(values[0][j], values[1][j],
                       values[2][j], values[3][j]);
}
static inline void dd4_poly_store(soxr_dd_t * output, dd4 x)
{
  __m256d a = _mm256_unpacklo_pd(x.hi, x.lo);
  __m256d b = _mm256_unpackhi_pd(x.hi, x.lo);
  __m256d c = _mm256_permute2f128_pd(a, b, 0x20);
  __m256d d = _mm256_permute2f128_pd(a, b, 0x31);
  memcpy(output, &c, sizeof(c));
  memcpy(output + 2, &d, sizeof(d));
}
void soxr_dd_poly0_avx(struct stage * p, fifo_t * outfifo)
{
  int n = min(stage_occupancy(p), p->input_size);
  if (!n) return;
  soxr_dd_t const * input = (soxr_dd_t const *)fifo_read_ptr(&p->fifo) + p->pre;
  int at = p->at.integer, step = p->step.integer;
  int amount = (n * p->L - at + step - 1) / step;
  soxr_dd_t * output = fifo_reserve(outfifo, amount);
  if (!output) { soxr_stage_oom(p); return; }
  int i = 0;
  for (; i + 4 <= amount; i += 4, at += 4 * step) {
    soxr_dd_t const * in[4], * co[4];
    for (int lane = 0; lane < 4; ++lane) {
      int pos = at + lane * step;
      in[lane] = input + pos / p->L;
      co[lane] = p->shared->poly_dd + (size_t)p->n * (pos % p->L);
    }
    dd4 sum = {_mm256_setzero_pd(), _mm256_setzero_pd()};
    int fallback = 0;
    for (int j = 0; j < p->n; ++j) {
      dd4 x = dd4_poly_get(in, j);
      dd4 c = dd4_poly_get(co, j);
      if (dd4_needs_scalar(x, c)) {fallback = 1; break;}
      sum = dd4_add(sum, dd4_mul(x, c));
    }
    if (fallback) {
      for (int lane = 0; lane < 4; ++lane) {
        soxr_dd_t scalar = soxr_dd_from(0.);
        for (int j = 0; j < p->n; ++j)
          scalar = soxr_dd_add(scalar, soxr_dd_mul(in[lane][j], co[lane][j]));
        output[i + lane] = scalar;
      }
    } else dd4_poly_store(output + i, sum);
  }
  for (; at < n * p->L; ++i, at += step) {
    int d = at / p->L, rem = at % p->L;
    soxr_dd_t const * in = input + d;
    soxr_dd_t const * co = p->shared->poly_dd + (size_t)p->n * rem;
    soxr_dd_t sum = soxr_dd_from(0.);
    for (int j = 0; j < p->n; ++j)
      sum = soxr_dd_add(sum, soxr_dd_mul(in[j], co[j]));
    output[i] = sum;
  }
  fifo_read(&p->fifo, at / p->L, NULL);
  p->at.integer = at % p->L;
}

#if defined(__clang__)
#pragma clang attribute pop
#endif
