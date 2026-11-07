/* Scalar Ooura / SIMD PFFFT FFT selector. No mutable globals or synchronization.
 * Only called during each rate engine's initialization, never for FFT samples. */

/*
 * SoXR-Poikosoft | Enhanced SoX Resampler Library
 *
 * Poikosoft integration and enhancements Copyright (c) 2026
 * Jukka Poikolainen / Poikosoft | https://www.poikosoft.com
 *
 * Module: Per-engine Fast FFT selection
 * Features:
 *  - CPUID/XGETBV capability detection during engine setup.
 *  - Ooura for scalar only; PFFFT for SSE2/AVX/AVX2+FMA3.
 *  - No mutable global FFT dispatch state.
 *
 * Original third-party copyrights and license terms are preserved.
 */

#include "soxr-config.h"
#include "pffft-runtime.h"
#include <stdint.h>
#if defined(_MSC_VER)
 #include <intrin.h>
#elif defined(__i386__) || defined(__x86_64__)
 #include <cpuid.h>
#endif
extern soxr_rdft_callbacks_t const _soxr_rdft32_cb;
extern soxr_rdft_callbacks_t const _soxr_rdft64_cb;
extern soxr_rdft_callbacks_t const _soxr_pffft_sse2_f64_cb;
extern soxr_rdft_callbacks_t const _soxr_pffft_sse_f32_cb;
extern soxr_rdft_callbacks_t const _soxr_pffft_avx_f32_cb;
extern soxr_rdft_callbacks_t const _soxr_pffft_avx_f64_cb;
extern soxr_rdft_callbacks_t const _soxr_pffft_avx2_f32_cb;
extern soxr_rdft_callbacks_t const _soxr_pffft_avx2_f64_cb;

static void cpuinfo(unsigned leaf, unsigned sub, unsigned * a, unsigned * b, unsigned * c, unsigned * d)
{
#if defined(_MSC_VER)
  int r[4]; __cpuidex(r, (int) leaf, (int) sub);
  *a = (unsigned) r[0]; *b = (unsigned) r[1]; *c = (unsigned) r[2]; *d = (unsigned) r[3];
#elif defined(__i386__) || defined(__x86_64__)
  __cpuid_count(leaf, sub, *a, *b, *c, *d);
#else
  (void) leaf; (void) sub; *a =*b =*c =*d = 0;
#endif
}
static uint64_t xgetbv0(void)
{
#if defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
  return _xgetbv(0);
#elif defined(__i386__) || defined(__x86_64__)
  unsigned a, d; __asm__ volatile("xgetbv": "=a"(a), "=d"(d): "c"(0));
  return (uint64_t) a | ((uint64_t) d << 32);
#else
  return 0;
#endif
}
static void cpu_caps(int * sse, int * avx, int * avx2_fma)
{
  unsigned a, b, c, d, max_leaf = 0;
  *sse =*avx =*avx2_fma = 0;
  cpuinfo(0, 0, &max_leaf, &b, &c, &d);
  if(max_leaf < 1) return;
  cpuinfo(1, 0, &a, &b, &c, &d);
  *sse =!!(d & (1u << 26)); /* SSE2: x64 baseline; F32 keeps original SSE FFT. */
  if(!((c & (1u << 27)) && (c & (1u << 28)))) return; /* OSXSAVE and AVX */
  if((xgetbv0() & 6) != 6) return; /* XMM+YMM enabled by operating system */
  *avx = 1;
  if(!(c & (1u << 12)) || max_leaf < 7) return; /* FMA3 */
  cpuinfo(7, 0, &a, &b, &c, &d);
  *avx2_fma =!!(b & (1u << 5));
}
int soxr_fft_pick_backend(int is_double, soxr_rdft_callbacks_t const **result)
{
  int sse, avx, avx2_fma, selected = USE_SIMD;
  soxr_rdft_callbacks_t const * chosen = is_double?&_soxr_rdft64_cb : &_soxr_rdft32_cb;
  if(selected == SIMD_DISABLE) { *result = chosen; return 0; }
  cpu_caps(&sse, &avx, &avx2_fma);
  if(selected == SIMD_AUTO)
    selected = avx2_fma?SIMD_AVX2 : avx?SIMD_AVX : sse?SIMD_SSE2 : SIMD_DISABLE;
  else if((selected == SIMD_AVX2 && !avx2_fma) ||
          (selected == SIMD_AVX && !avx) ||
          (selected == SIMD_SSE2 && !sse))
    return 1;
  switch(selected) {
    case SIMD_AVX2:
      chosen = is_double?&_soxr_pffft_avx2_f64_cb : &_soxr_pffft_avx2_f32_cb;
      break;
    case SIMD_AVX:
      chosen = is_double?&_soxr_pffft_avx_f64_cb : &_soxr_pffft_avx_f32_cb;
      break;
    case SIMD_SSE2:
      /* F32 uses the original SSE PFFFT; F64 uses paired-SSE2 PFFFT. */
      chosen = is_double ? &_soxr_pffft_sse2_f64_cb : &_soxr_pffft_sse_f32_cb;
      break;
    case SIMD_DISABLE:
    default:
      break;
  }
  *result = chosen;
  return 0;
}

/* Extreme performs its own ISA selection, independent of Fast USE_SIMD. */
int soxr_cpu_has_sse2(void) {
  unsigned a, b, c, d;
  cpuinfo(1, 0, &a, &b, &c, &d);
  return !!(d & (1u << 26));
}
int soxr_cpu_has_avx(void) {
  int sse, avx, avx2_fma;
  cpu_caps(&sse, &avx, &avx2_fma);
  return avx;
}
int soxr_cpu_has_avx2_fma3(void) {
  int sse, avx, avx2_fma;
  cpu_caps(&sse, &avx, &avx2_fma);
  return avx2_fma;
}
