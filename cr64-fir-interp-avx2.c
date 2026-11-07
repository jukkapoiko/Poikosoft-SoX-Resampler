/* SoXR-Poikosoft: AVX2 interpolated polyphase FIR.
 * Original SoXR FIR source/implementation LGPL-2.1 (robs@users.sourceforge.net).
 * Separate translation unit: AVX2 instructions never reach scalar fallback.
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
#pragma clang attribute push (__attribute__((target("avx2,fma"))), apply_to=function)
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

#define FUNCTION soxr_fast64_vpoly1_avx2_private
#define COEF_INTERP 1
#define PHASE_BITS VAR_POLY_PHASE_BITS
#define FIR_LENGTH VAR_LENGTH
#define CONVOLVE(n) VAR_CONVOLVE(n)
#include "poly-fir-avx2-tail.h"

#define FUNCTION soxr_fast64_vpoly2_avx2_private
#define COEF_INTERP 2
#define PHASE_BITS VAR_POLY_PHASE_BITS
#define FIR_LENGTH VAR_LENGTH
#define CONVOLVE(n) VAR_CONVOLVE(n)
#include "poly-fir-avx2-tail.h"

#define FUNCTION soxr_fast64_vpoly3_avx2_private
#define COEF_INTERP 3
#define PHASE_BITS VAR_POLY_PHASE_BITS
#define FIR_LENGTH VAR_LENGTH
#define CONVOLVE(n) VAR_CONVOLVE(n)
#include "poly-fir-avx2-tail.h"

void soxr_fast64_vpoly1_avx2(stage_t * p, fifo_t * out) { soxr_fast64_vpoly1_avx2_private(p, out); }
void soxr_fast64_vpoly2_avx2(stage_t * p, fifo_t * out) { soxr_fast64_vpoly2_avx2_private(p, out); }
void soxr_fast64_vpoly3_avx2(stage_t * p, fifo_t * out) { soxr_fast64_vpoly3_avx2_private(p, out); }

#if defined(__clang__)
#pragma clang attribute pop
#endif
