/* SoX Resampler Library      Copyright (c) 2007-18 robs@users.sourceforge.net
 * Licence for this file: LGPL v2.1                  See LICENCE for details.
 *
 * Constant-rate resampling engine-specific code. */

/*
 * SoXR-Poikosoft | Enhanced SoX Resampler Library
 *
 * Poikosoft integration and enhancements Copyright (c) 2026
 * Jukka Poikolainen / Poikosoft | https://www.poikosoft.com
 *
 * Module: Fast constant-rate processing engine
 * Features:
 *  - Shared Float32/Float64 stage graph.
 *  - Selected FFT callbacks and SIMD-capable FIR kernels.
 *
 * Original third-party copyrights and license terms are preserved.
 */



#include <math.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

#include "filter.h"
#include "soxr-config.h"
#if defined SOXR_LIB
  #include "internal.h"
  #include "cr.h"
  #include "rate-callbacks.h"
  #include "pffft-runtime.h"
#if CORE_TYPE & CORE_DBL
    typedef double sample_t;
    #define RDFT_CB _soxr_rdft64_cb /* Ooura fallback */
#else
    typedef float sample_t;
    #define RDFT_CB _soxr_rdft32_cb /* Ooura fallback */
#endif

#if CORE_TYPE & (CORE_SIMD_POLY|CORE_SIMD_HALF|CORE_SIMD_DFT)
#if CORE_TYPE & CORE_DBL
      #include "util64s.h"
      #include "dev64s.h"
#else
      #include "util32s.h"
      #include "dev32s.h"
#endif
#endif

  extern soxr_rdft_callbacks_t const RDFT_CB;
#else
  #define RDFT_CB 0
#endif



static void cubic_stage_fn(stage_t * p, fifo_t * output_fifo)
{
  sample_t const * input = stage_read_p(p);
  int num_in = min(stage_occupancy(p), p->input_size);
  int i, max_num_out = 1 + (int)(num_in * p->out_in_ratio);
  sample_t * output = fifo_reserve(output_fifo, max_num_out);
  if(!output) { soxr_stage_oom(p); return; }

  for(i = 0; p->at.integer < num_in; ++i, p->at.whole += p->step.whole) {
    sample_t const * s = input + p->at.integer;
    double x = p->at.fraction * (1 / MULT32);
    double b = .5 * (s[1] + s[-1]) -*s, a = (1 / 6.) * (s[2] - s[1] + s[-1] -*s - 4 * b);
    double c = s[1] -*s - a - b;
    output[i] = (sample_t)(p->mult * (((a * x + b) * x + c) * x + *s));
  }
  assert(max_num_out - i >= 0);
  fifo_trim_by(output_fifo, max_num_out - i);
  fifo_read(&p->fifo, p->at.integer, NULL);
  p->at.integer = 0;
}



#if defined __AVX__
  #define DEFINED_AVX 1
#else
  #define DEFINED_AVX 0
#endif

#if defined __x86_64__ || defined _M_X64 || defined i386 || defined _M_IX86
  #define DEFINED_X86 1
#else
  #define DEFINED_X86 0
#endif

#if defined __arm__
  #define DEFINED_ARM 1
#else
  #define DEFINED_ARM 0
#endif



#if CORE_TYPE & CORE_DBL
  #define SIMD_AVX ((CORE_TYPE & CORE_SIMD_HALF) && DEFINED_AVX)
  #define SIMD_SSE 0
#else
  #define SIMD_SSE ((CORE_TYPE & CORE_SIMD_HALF) && DEFINED_X86)
  #define SIMD_AVX 0
#endif

#define SIMD_NEON ((CORE_TYPE & CORE_SIMD_HALF) && DEFINED_ARM)



#include "half-coefs.h"

#if !(CORE_TYPE & CORE_SIMD_HALF)
#define FUNCTION_H h7
#define CONVOLVE ____ __ _
#include "half-fir.h"
#endif

#define FUNCTION_H h8
#define CONVOLVE ____ ____
#include "half-fir.h"

#define FUNCTION_H h9
#define CONVOLVE ____ ____ _
#include "half-fir.h"

#if CORE_TYPE & CORE_DBL
  #define FUNCTION_H h10
  #define CONVOLVE ____ ____ __
  #include "half-fir.h"

  #define FUNCTION_H h11
  #define CONVOLVE ____ ____ __ _
  #include "half-fir.h"

  #define FUNCTION_H h12
  #define CONVOLVE ____ ____ ____
  #include "half-fir.h"

  #define FUNCTION_H h13
  #define CONVOLVE ____ ____ ____ _
  #include "half-fir.h"
#endif

static half_fir_info_t const half_firs[] = {
#if !(CORE_TYPE & CORE_SIMD_HALF)
  { 7, half_fir_coefs_7, h7, 0, 120.65f },
#endif
  { 8, half_fir_coefs_8, h8, 0, 136.51f },
  { 9, half_fir_coefs_9, h9, 0, 152.32f },
#if CORE_TYPE & CORE_DBL
  { 10, half_fir_coefs_10, h10, 0, 168.08f },
  { 11, half_fir_coefs_11, h11, 0, 183.79f },
  { 12, half_fir_coefs_12, h12, 0, 199.46f },
  { 13, half_fir_coefs_13, h13, 0, 215.12f },
#endif
};

#undef SIMD_AVX
#undef SIMD_NEON
#undef SIMD_SSE



#if CORE_TYPE & CORE_DBL
  #define SIMD_AVX ((CORE_TYPE & CORE_SIMD_POLY) && DEFINED_AVX)
  #define SIMD_SSE 0
#else
  #define SIMD_SSE ((CORE_TYPE & CORE_SIMD_POLY) && DEFINED_X86)
  #define SIMD_AVX 0
#endif

#define SIMD_NEON ((CORE_TYPE & CORE_SIMD_POLY) && DEFINED_ARM)



#define COEFS (sample_t * __restrict)p->shared->poly_fir_coefs
#define VAR_LENGTH p->n
#define VAR_CONVOLVE(n) while (j < (n)) _
#define VAR_POLY_PHASE_BITS p->phase_bits



#define FUNCTION vpoly0
#define FIR_LENGTH VAR_LENGTH
#define CONVOLVE(n) VAR_CONVOLVE(n)
#include "poly-fir0.h"

#define FUNCTION vpoly1
#define COEF_INTERP 1
#define PHASE_BITS VAR_POLY_PHASE_BITS
#define FIR_LENGTH VAR_LENGTH
#define CONVOLVE(n) VAR_CONVOLVE(n)
#include "poly-fir.h"

#define FUNCTION vpoly2
#define COEF_INTERP 2
#define PHASE_BITS VAR_POLY_PHASE_BITS
#define FIR_LENGTH VAR_LENGTH
#define CONVOLVE(n) VAR_CONVOLVE(n)
#include "poly-fir.h"

#define FUNCTION vpoly3
#define COEF_INTERP 3
#define PHASE_BITS VAR_POLY_PHASE_BITS
#define FIR_LENGTH VAR_LENGTH
#define CONVOLVE(n) VAR_CONVOLVE(n)
#include "poly-fir.h"



#if !(CORE_TYPE & CORE_SIMD_POLY)

#define poly_fir_convolve_U100 _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
#define FUNCTION U100_0
#define FIR_LENGTH U100_l
#define CONVOLVE(n) poly_fir_convolve_U100
#include "poly-fir0.h"

#define u100_l 11
#define poly_fir_convolve_u100 _ _ _ _ _ _ _ _ _ _ _
#define FUNCTION u100_0
#define FIR_LENGTH u100_l
#define CONVOLVE(n) poly_fir_convolve_u100
#include "poly-fir0.h"

#define FUNCTION u100_1
#define COEF_INTERP 1
#define PHASE_BITS 8
#define FIR_LENGTH u100_l
#define CONVOLVE(n) poly_fir_convolve_u100
#include "poly-fir.h"

#define FUNCTION u100_2
#define COEF_INTERP 2
#define PHASE_BITS 6
#define FIR_LENGTH u100_l
#define CONVOLVE(n) poly_fir_convolve_u100
#include "poly-fir.h"

#endif

#define u100_1_b 8
#define u100_2_b 6



/* Integrate the old cr32s/cr64s SIMD-FIR accumulation directly in cr32/cr64.
 * Coefficient storage/layout stays exactly as in Reference v2. One stage
 * callback substitution at initialization; no separate cr32s/cr64s core.
 * This path optimizes non-interpolated polyphase FIR only. */
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#if !(CORE_TYPE & CORE_DBL)
static void vpoly0_simd(stage_t * p, fifo_t * output_fifo)
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
      int j; __m128 vs = _mm_setzero_ps(); float sum;
      sample_t const * x = input + div, *h = coefs + taps * rem;
      for(j = 0; j < veclen; j += 4)
        vs = _mm_add_ps(vs, _mm_mul_ps(_mm_loadu_ps(h + j), _mm_loadu_ps(x + j)));
      { __m128 sh = _mm_movehl_ps(vs, vs);
        vs = _mm_add_ps(vs, sh); sh = _mm_shuffle_ps(vs, vs, 1);
        vs = _mm_add_ss(vs, sh); sum = _mm_cvtss_f32(vs); }
      for(; j < taps; ++j) sum += h[j] * x[j];
      output[i] = sum;
      div += sd; rem += sr; if(rem >= p->L) { rem -= p->L; ++div; }
    }
    assert(i == nout); fifo_read(&p->fifo, div, NULL); p->at.integer = rem;
  }
}
#endif
/* Only Float32 uses an inline SIMD vpoly0 implementation. Fast64 always
 * dispatches to the external AVX vpoly0 kernel when AVX/AVX2 is selected. */
#if !(CORE_TYPE & CORE_DBL)
#define SOXR_HAVE_INLINE_SIMD_POLY0 1
#endif
#endif

static poly_fir_t const poly_firs[] = {
  { -1, {{ 0, vpoly0 }, { 7.2f, vpoly1 }, { 5.0f, vpoly2 } } },
  { -1, {{ 0, vpoly0 }, { 9.4f, vpoly1 }, { 6.7f, vpoly2 } } },
  { -1, {{ 0, vpoly0 }, { 12.4f, vpoly1 }, { 7.8f, vpoly2 } } },
  { -1, {{ 0, vpoly0 }, { 13.6f, vpoly1 }, { 9.3f, vpoly2 } } },
  { -1, {{ 0, vpoly0 }, { 10.5f, vpoly2 }, { 8.4f, vpoly3 } } },
  { -1, {{ 0, vpoly0 }, { 11.85f, vpoly2 }, { 9.0f, vpoly3 } } },

  { -1, {{ 0, vpoly0 }, { 8.0f, vpoly1 }, { 5.3f, vpoly2 } } },
  { -1, {{ 0, vpoly0 }, { 8.6f, vpoly1 }, { 5.7f, vpoly2 } } },
  { -1, {{ 0, vpoly0 }, { 10.6f, vpoly1 }, { 6.75f, vpoly2 } } },
  { -1, {{ 0, vpoly0 }, { 12.6f, vpoly1 }, { 8.6f, vpoly2 } } },
  { -1, {{ 0, vpoly0 }, { 9.6f, vpoly2 }, { 7.6f, vpoly3 } } },
  { -1, {{ 0, vpoly0 }, { 11.4f, vpoly2 }, { 8.65f, vpoly3 } } },

#if CORE_TYPE & CORE_SIMD_POLY
  { 10.62f, {{ 0, vpoly0 }, { 0, 0 }, { 0, 0 } } },
  { -1, {{ 0, vpoly0 }, { u100_1_b, vpoly1 }, { u100_2_b, vpoly2 } } },
#else
  { 10.62f, {{ U100_l, U100_0 }, { 0, 0 }, { 0, 0 } } },
  { 11.28f, {{ u100_l, u100_0 }, { u100_1_b, u100_1 }, { u100_2_b, u100_2 } } },
#endif
  { -1, {{ 0, vpoly0 }, { 9, vpoly1 }, { 6, vpoly2 } } },
  { -1, {{ 0, vpoly0 }, { 11, vpoly1 }, { 7, vpoly2 } } },
  { -1, {{ 0, vpoly0 }, { 13, vpoly1 }, { 8, vpoly2 } } },
  { -1, {{ 0, vpoly0 }, { 10, vpoly2 }, { 8, vpoly3 } } },
  { -1, {{ 0, vpoly0 }, { 12, vpoly2 }, { 9, vpoly3 } } },
};



static cr_core_t const cr_core = {

#if CORE_TYPE & CORE_SIMD_POLY
  { SIMD_ALIGNED_MALLOC, SIMD_ALIGNED_CALLOC, SIMD_ALIGNED_FREE },
#else
  { malloc, calloc, free },
#endif
  half_firs, array_length(half_firs),
  0, 0,
  cubic_stage_fn,
  poly_firs, &RDFT_CB
};



#if defined SOXR_LIB

#include "soxr.h"

static char const * rate_create(void * channel, void * shared, double io_ratio,
    soxr_quality_spec_t * q_spec, soxr_runtime_spec_t * r_spec, double scale)
{
  rate_t * engine = (rate_t *) channel;
  engine->selected_core = cr_core;
  /* One CPU selection per FFT/rate engine, no global state or locks. */
  if(soxr_fft_pick_backend((CORE_TYPE & CORE_DBL) != 0,
          &engine->selected_core.rdft_cb))
    return "USE_SIMD requires an unavailable CPU instruction set";
  {
    int simd_fft = engine->selected_core.rdft_cb != 0 &&
      (engine->selected_core.rdft_cb->flags() & RDFT_IS_SIMD);
    char const * error = _soxr_init(channel, shared, io_ratio, q_spec, r_spec, scale,
      &engine->selected_core, CORE_TYPE | (simd_fft?CORE_SIMD_DFT : 0));
#if defined(SOXR_HAVE_INLINE_SIMD_POLY0)
    if(!error && simd_fft) {
      int k;
      for(k = 0; k < engine->num_stages; ++k)
        if(engine->stages[k].fn == vpoly0)
          engine->stages[k].fn = vpoly0_simd;
    }
#endif
#if CORE_TYPE & CORE_DBL
    extern soxr_rdft_callbacks_t const _soxr_pffft_sse2_f64_cb;
    if (!error && engine->selected_core.rdft_cb == &_soxr_pffft_sse2_f64_cb) {
      int k;
      for (k = 0; k < engine->num_stages; ++k) {
        stage_t * s = &engine->stages[k];
        int order = s->fn == vpoly1 ? 1 : s->fn == vpoly2 ? 2 :
                    s->fn == vpoly3 ? 3 : 0;
        if (s->fn == vpoly0)
          s->fn = soxr_fast64_vpoly0_sse2;
        else if (order && soxr_fast64_interp_pack(s, order))
          s->fn = order == 1 ? soxr_fast64_vpoly1_sse2 :
                  order == 2 ? soxr_fast64_vpoly2_sse2 :
                               soxr_fast64_vpoly3_sse2;
      }
    }
#endif
#if CORE_TYPE & CORE_DBL
    /* Fast64 AVX and AVX2 always select the external AVX vpoly0 kernel.
     * Interpolated FIR uses the matching AVX/AVX2 implementation. */
    if (!error && engine->selected_core.rdft_cb) {
      extern soxr_rdft_callbacks_t const _soxr_pffft_avx_f64_cb;
      extern soxr_rdft_callbacks_t const _soxr_pffft_avx2_f64_cb;
      int avx2 = engine->selected_core.rdft_cb == &_soxr_pffft_avx2_f64_cb;
      int avx = engine->selected_core.rdft_cb == &_soxr_pffft_avx_f64_cb;
      if (avx || avx2) {
        int k;
        for (k = 0; k < engine->num_stages; ++k) {
          stage_t * s = &engine->stages[k];
          int order = s->fn == vpoly1 ? 1 : s->fn == vpoly2 ? 2 :
                      s->fn == vpoly3 ? 3 : 0;
          if (s->fn == vpoly0)
            s->fn = soxr_fast64_vpoly0_avx;
          else if (order && soxr_fast64_interp_pack(s, order)) {
            if (avx2)
              s->fn = order == 1 ? soxr_fast64_vpoly1_avx2 :
                      order == 2 ? soxr_fast64_vpoly2_avx2 :
                                   soxr_fast64_vpoly3_avx2;
            else
              s->fn = order == 1 ? soxr_fast64_vpoly1_avx :
                      order == 2 ? soxr_fast64_vpoly2_avx :
                                   soxr_fast64_vpoly3_avx;
          }
        }
      }
    }
#endif
    return error;
  }
}



static char const * id(void) { return CORE_STR; }

static void * typed_input(void * state, void const * samples, size_t n)
{
  return _soxr_input((rate_t *) state, (real const *) samples, n);
}
static void typed_process(void * state, size_t n)
{
  _soxr_process((rate_t *) state, n);
}
static void const * typed_output(void * state, void * samples, size_t * n)
{
  return _soxr_output((rate_t *) state, (real *) samples, n);
}
static void typed_flush(void * state)
{
  _soxr_flush((rate_t *) state);
}
static void typed_close(void * state)
{
  _soxr_close((rate_t *) state);
}
static double typed_delay(void * state)
{
  return _soxr_delay((rate_t *) state);
}
static char const * typed_status(void * state)
{
  return _soxr_runtime_error((rate_t *) state);
}

soxr_rate_callbacks_t const RATE_CB = {
  typed_input, typed_process, typed_output, typed_flush, typed_close,
  typed_delay, _soxr_sizes, rate_create, NULL, id, typed_status
};

#endif
