/* Poikosoft Fast64 SSE2: two-double vector operations, no AVX/FMA. */
#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <emmintrin.h>
#include "soxr-config.h"
#include "internal.h"
typedef double sample_t;
#include "cr.h"
#include "pffft-backend-util.h"

/* Preserve 4-tap packed interpolation layout; pair of 128-bit registers. */
typedef struct { __m128d lo, hi; } v4_t;
static inline v4_t sse2_zero(void) { v4_t x = {_mm_setzero_pd(), _mm_setzero_pd()}; return x; }
static inline v4_t sse2_load(double const * p) { v4_t x = {_mm_loadu_pd(p), _mm_loadu_pd(p + 2)}; return x; }
static inline v4_t sse2_broadcast(double x) { v4_t v = {_mm_set1_pd(x), _mm_set1_pd(x)}; return v; }
static inline v4_t sse2_add(v4_t a, v4_t b) { v4_t r = {_mm_add_pd(a.lo, b.lo), _mm_add_pd(a.hi, b.hi)}; return r; }
static inline v4_t sse2_mul(v4_t a, v4_t b) { v4_t r = {_mm_mul_pd(a.lo, b.lo), _mm_mul_pd(a.hi, b.hi)}; return r; }
static inline void sse2_sum(double * out, v4_t v) {
  __m128d a = _mm_add_pd(v.lo, v.hi);
  __m128d b = _mm_unpackhi_pd(a, a);
  *out = _mm_cvtsd_f64(_mm_add_sd(a, b));
}
static inline double soxr_fast64_interp_tail(double const * coe, double x, int degree) {
  double v = coe[0];
  for(int k = 1; k <= degree; ++k) v = v * x + coe[k];
  return v;
}
#define vZero() sse2_zero()
#define vMul(a,b) sse2_mul((a),(b))
#define vAdd(a,b) sse2_add((a),(b))
#define vMac(a,b,c) vAdd(vMul((a),(b)),(c))
#define vLds(a) sse2_broadcast(a)
#define vLdu(a) sse2_load(a)
#define vStorSum(a,b) sse2_sum((a),(b))
#define SIMD_AVX 0
#define SIMD_SSE 1
#define SIMD_NEON 0
#define COEFS (sample_t * __restrict) p->shared->poly_fir_simd_coefs
#define VAR_LENGTH p->n
#define VAR_CONVOLVE(n) while (j < (n)) _
#define VAR_POLY_PHASE_BITS p->phase_bits
#define FUNCTION soxr_fast64_vpoly1_sse2_private
#define COEF_INTERP 1
#define PHASE_BITS VAR_POLY_PHASE_BITS
#define FIR_LENGTH VAR_LENGTH
#define CONVOLVE(n) VAR_CONVOLVE(n)
#include "poly-fir-avx2-tail.h"

#define FUNCTION soxr_fast64_vpoly2_sse2_private
#define COEF_INTERP 2
#define PHASE_BITS VAR_POLY_PHASE_BITS
#define FIR_LENGTH VAR_LENGTH
#define CONVOLVE(n) VAR_CONVOLVE(n)
#include "poly-fir-avx2-tail.h"

#define FUNCTION soxr_fast64_vpoly3_sse2_private
#define COEF_INTERP 3
#define PHASE_BITS VAR_POLY_PHASE_BITS
#define FIR_LENGTH VAR_LENGTH
#define CONVOLVE(n) VAR_CONVOLVE(n)
#include "poly-fir-avx2-tail.h"

void soxr_fast64_vpoly1_sse2(stage_t * p, fifo_t * out) { soxr_fast64_vpoly1_sse2_private(p, out); }

void soxr_fast64_vpoly2_sse2(stage_t * p, fifo_t * out) { soxr_fast64_vpoly2_sse2_private(p, out); }

void soxr_fast64_vpoly3_sse2(stage_t * p, fifo_t * out) { soxr_fast64_vpoly3_sse2_private(p, out); }

/* Rational polyphase FIR: the same four-lane accumulation order as the
 * existing GCC/Clang Fast64 kernel, now also available to MSVC /arch:AVX.
 * No changes to coefficients, source taps or rounding order.
 */
void soxr_fast64_vpoly0_sse2(stage_t * p, fifo_t * output_fifo)
{
  int num_in = min(stage_occupancy(p), p->input_size);
  if(num_in) {
    sample_t const * input = stage_read_p(p);
    int at = p->at.integer, step = p->step.integer;
    int div = at / p->L, rem = at % p->L;
    int const sd = step / p->L, sr = step % p->L;
    int i, nout = (num_in * p->L - at + step - 1) / step;
    int const taps = p->n, veclen = taps &~1;
    sample_t * output = fifo_reserve(output_fifo, nout);
    if(!output) { soxr_stage_oom(p); return; }
    sample_t const * coefs = (sample_t const *) p->shared->poly_fir_coefs;
    for(i = 0; at < num_in * p->L; ++i, at += step) {
      int j; __m128d vs = _mm_setzero_pd(); double sum;
      sample_t const * x = input + div, *h = coefs + taps * rem;
      for(j = 0; j < veclen; j += 2)
        vs = _mm_add_pd(vs, _mm_mul_pd(_mm_loadu_pd(h + j), _mm_loadu_pd(x + j)));
      { __m128d vh = _mm_unpackhi_pd(vs, vs);
        vs = _mm_add_sd(vs, vh); sum = _mm_cvtsd_f64(vs); }
      for(; j < taps; ++j) sum += h[j] * x[j];
      output[i] = sum;
      div += sd; rem += sr; if(rem >= p->L) { rem -= p->L; ++div; }
    }
    assert(i == nout); fifo_read(&p->fifo, div, NULL); p->at.integer = rem;
  }
}
