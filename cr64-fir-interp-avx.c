/* SoXR-Poikosoft: AVX interpolated polyphase FIR.
 * Original SoXR FIR source/implementation LGPL-2.1 (robs@users.sourceforge.net).
 * Separate translation unit: AVX instructions never reach scalar fallback.
 */
#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <immintrin.h>
#include "soxr-config.h"
#include "internal.h"
typedef double sample_t;
#include "cr.h"
#include "pffft-backend-util.h"
/* Keep ISA specialization local to this backend under Clang/Intel oneAPI.
 * Plain-C, SSE2 and dispatch files can still be compiled for baseline x64.
 * GCC users still need their normal per-file -m flags. */
#if defined(__clang__)
#pragma clang attribute push (__attribute__((target("avx"))), apply_to=function)
#endif

/* Double samples stay in the original scalar coefficient layout; only this
 * optional copy uses the original cr64s coef4 (4 lanes per coefficient). */
/* The vectorized convolution and the original SoXR high-precision clock
 * are shared across all three polynomial interpolation orders. No FMA
 * contraction: match vMac's multiply-then-add numerical behavior. */
typedef __m256d v4_t;
#define vZero()      _mm256_setzero_pd()
#define vMul(a,b)    _mm256_mul_pd(a,b)
#define vAdd(a,b)    _mm256_add_pd(a,b)
#define vMac(a,b,c)  vAdd(vMul(a,b),c)
#define vLds(a)      _mm256_set1_pd(a)
#define vLdu(a)      _mm256_loadu_pd(a)
static inline void vStorSum(double * a, v4_t b)
{
  b = _mm256_hadd_pd(b, _mm256_permute2f128_pd(b, b, 1));
  _mm_store_sd(a, _mm256_castpd256_pd128(_mm256_hadd_pd(b, b)));
}
/* Tail coefficient order in the scalar buffer is [d,c,b,a] for degree 3.
 * This matches scalar Horner evaluation without FMA contraction. */
static inline double soxr_fast64_interp_tail(double const * coe, double x, int degree)
{
  double value = coe[0];
  int i;
  for (i = 1; i <= degree; ++i)
    value = value * x + coe[i];
  return value;
}
#define SIMD_AVX 1
#define SIMD_SSE 0
#define SIMD_NEON 0
#define COEFS (sample_t * __restrict) p->shared->poly_fir_simd_coefs
#define VAR_LENGTH p->n
#define VAR_CONVOLVE(n) while (j < (n)) _
#define VAR_POLY_PHASE_BITS p->phase_bits

#define FUNCTION soxr_fast64_vpoly1_avx_private
#define COEF_INTERP 1
#define PHASE_BITS VAR_POLY_PHASE_BITS
#define FIR_LENGTH VAR_LENGTH
#define CONVOLVE(n) VAR_CONVOLVE(n)
#include "poly-fir-avx2-tail.h"

#define FUNCTION soxr_fast64_vpoly2_avx_private
#define COEF_INTERP 2
#define PHASE_BITS VAR_POLY_PHASE_BITS
#define FIR_LENGTH VAR_LENGTH
#define CONVOLVE(n) VAR_CONVOLVE(n)
#include "poly-fir-avx2-tail.h"

#define FUNCTION soxr_fast64_vpoly3_avx_private
#define COEF_INTERP 3
#define PHASE_BITS VAR_POLY_PHASE_BITS
#define FIR_LENGTH VAR_LENGTH
#define CONVOLVE(n) VAR_CONVOLVE(n)
#include "poly-fir-avx2-tail.h"

void soxr_fast64_vpoly1_avx(stage_t * p, fifo_t * out) { soxr_fast64_vpoly1_avx_private(p, out); }
void soxr_fast64_vpoly2_avx(stage_t * p, fifo_t * out) { soxr_fast64_vpoly2_avx_private(p, out); }
void soxr_fast64_vpoly3_avx(stage_t * p, fifo_t * out) { soxr_fast64_vpoly3_avx_private(p, out); }

/* Rational polyphase FIR: the same four-lane accumulation order as the
 * existing GCC/Clang Fast64 kernel, now also available to MSVC /arch:AVX.
 * No changes to coefficients, source taps or rounding order.
 */
void soxr_fast64_vpoly0_avx(stage_t * p, fifo_t * output_fifo)
{
  int num_in = min(stage_occupancy(p), p->input_size);
  if(num_in) {
    sample_t const * input = stage_read_p(p);
    int at = p->at.integer, step = p->step.integer;
    int div = at / p->L, rem = at % p->L;
    int const sd = step / p->L, sr = step % p->L;
    int i, nout = (num_in * p->L - at + step - 1) / step;
    int const taps = p->n, veclen = taps &~3;
    sample_t * output = fifo_reserve(output_fifo, nout);
    if(!output) { soxr_stage_oom(p); return; }
    sample_t const * coefs = (sample_t const *) p->shared->poly_fir_coefs;
    for(i = 0; at < num_in * p->L; ++i, at += step) {
      int j; __m256d vs = _mm256_setzero_pd(); double sum;
      sample_t const * x = input + div, *h = coefs + taps * rem;
      for(j = 0; j < veclen; j += 4)
        vs = _mm256_add_pd(vs, _mm256_mul_pd(_mm256_loadu_pd(h + j), _mm256_loadu_pd(x + j)));
      { __m128d vl = _mm256_castpd256_pd128(vs), vh = _mm256_extractf128_pd(vs, 1);
        vl = _mm_add_pd(vl, vh); vh = _mm_unpackhi_pd(vl, vl);
        vl = _mm_add_sd(vl, vh); sum = _mm_cvtsd_f64(vl); }
      for(; j < taps; ++j) sum += h[j] * x[j];
      output[i] = sum;
      div += sd; rem += sr; if(rem >= p->L) { rem -= p->L; ++div; }
    }
    assert(i == nout); fifo_read(&p->fifo, div, NULL); p->at.integer = rem;
  }
}
#if defined(__clang__)
#pragma clang attribute pop
#endif
