/* Four-lane PFFFT Float64 operations implemented with paired SSE2 registers.
 * Each logical PFFFT vector contains four doubles, preserving PFFFT's
 * existing radix-4 data layout without requiring AVX instructions.
 * No AVX instructions are needed; the FFT uses the established PFFFT layout. */
#ifndef SOXR_PFFFT_SSE2_F64_H
#define SOXR_PFFFT_SSE2_F64_H
#include <emmintrin.h>
#include <stdint.h>

typedef struct { __m128d lo, hi; } v4sf;

static inline v4sf soxr_v4d_make(__m128d lo, __m128d hi)
{
  v4sf v;
  v.lo = lo;
  v.hi = hi;
  return v;
}
static inline v4sf soxr_v4d_zero(void)
{
  return soxr_v4d_make(_mm_setzero_pd(), _mm_setzero_pd());
}
static inline v4sf soxr_v4d_splat(double x)
{
  return soxr_v4d_make(_mm_set1_pd(x), _mm_set1_pd(x));
}
static inline v4sf soxr_v4d_add(v4sf a, v4sf b)
{
  return soxr_v4d_make(_mm_add_pd(a.lo, b.lo), _mm_add_pd(a.hi, b.hi));
}
static inline v4sf soxr_v4d_sub(v4sf a, v4sf b)
{
  return soxr_v4d_make(_mm_sub_pd(a.lo, b.lo), _mm_sub_pd(a.hi, b.hi));
}
static inline v4sf soxr_v4d_mul(v4sf a, v4sf b)
{
  return soxr_v4d_make(_mm_mul_pd(a.lo, b.lo), _mm_mul_pd(a.hi, b.hi));
}

#define VZERO() soxr_v4d_zero()
#define VMUL(a,b) soxr_v4d_mul((a),(b))
#define VADD(a,b) soxr_v4d_add((a),(b))
#define VMADD(a,b,c) VADD(VMUL((a),(b)),(c))
#define VSUB(a,b) soxr_v4d_sub((a),(b))
#define LD_PS1(p) soxr_v4d_splat(p)

/* Input: a0 a1 a2 a3 / b0 b1 b2 b3.
 * Interleave: a0 b0 a1 b1 / a2 b2 a3 b3. */
#define INTERLEAVE2(in1,in2,out1,out2) do { \
  v4sf a_ = (in1), b_ = (in2); \
  (out1) = soxr_v4d_make(_mm_unpacklo_pd(a_.lo,b_.lo), _mm_unpackhi_pd(a_.lo,b_.lo)); \
  (out2) = soxr_v4d_make(_mm_unpacklo_pd(a_.hi,b_.hi), _mm_unpackhi_pd(a_.hi,b_.hi)); \
} while (0)
/* Input a0 a1 a2 a3 / b0 b1 b2 b3.
 * Uninterleave: a0 a2 b0 b2 / a1 a3 b1 b3. */
#define UNINTERLEAVE2(in1,in2,out1,out2) do { \
  v4sf a_ = (in1), b_ = (in2); \
  (out1) = soxr_v4d_make(_mm_unpacklo_pd(a_.lo,a_.hi), _mm_unpacklo_pd(b_.lo,b_.hi)); \
  (out2) = soxr_v4d_make(_mm_unpackhi_pd(a_.lo,a_.hi), _mm_unpackhi_pd(b_.lo,b_.hi)); \
} while (0)
/* Transpose a four-by-four matrix of doubles using SSE2 unpack instructions. */
#define VTRANSPOSE4(x0,x1,x2,x3) do { \
  v4sf a_=(x0), b_=(x1), c_=(x2), d_=(x3); \
  (x0)=soxr_v4d_make(_mm_unpacklo_pd(a_.lo,b_.lo), _mm_unpacklo_pd(c_.lo,d_.lo)); \
  (x1)=soxr_v4d_make(_mm_unpackhi_pd(a_.lo,b_.lo), _mm_unpackhi_pd(c_.lo,d_.lo)); \
  (x2)=soxr_v4d_make(_mm_unpacklo_pd(a_.hi,b_.hi), _mm_unpacklo_pd(c_.hi,d_.hi)); \
  (x3)=soxr_v4d_make(_mm_unpackhi_pd(a_.hi,b_.hi), _mm_unpackhi_pd(c_.hi,d_.hi)); \
} while (0)
#define VSWAPHL(a,b) soxr_v4d_make((b).lo,(a).hi)
#define VALIGNED(ptr) ((((uintptr_t)(ptr)) & 0x1F) == 0)
#endif
