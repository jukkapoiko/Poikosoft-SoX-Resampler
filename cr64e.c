/* SoXR Extreme: dedicated double-double implementation of the existing
 * constant-rate stage graph. No double sample buffers or double FFT. */

/*
 * SoXR-Poikosoft | Enhanced SoX Resampler Library
 *
 * Poikosoft integration and enhancements Copyright (c) 2026
 * Jukka Poikolainen / Poikosoft | https://www.poikosoft.com
 *
 * Module: Extreme precision processing engine
 * Features:
 *  - Dedicated double-double stage arithmetic and FIR processing.
 *  - Uses the double-double FFT rather than Fast FFT dispatch.
 *
 * Original third-party copyrights and license terms are preserved.
 */


#include <stdlib.h>
#include <limits.h>
#include <string.h>
#include <math.h>
#include "soxr-config.h"
#include "internal.h"
#include "cr.h"
#include "rate-callbacks.h"
#include "filter.h"
#include "soxr-dd-fft.h"
#include "pffft-runtime.h"
#include "soxr.h"

#define CORE_TYPE CORE_DBL
#define sample_t double
#include "half-coefs.h"
#undef sample_t

typedef soxr_dd_t dd;
typedef soxr_dd_complex zdd;
static dd zero(void) { return soxr_dd_from(0); }
static dd getc(dd const * a, int order, int taps, int phase, int power, int tap) {
  return coef(a, order, taps, phase, power, tap);
}

soxr_dd_t * soxr_prepare_poly_full_dd(dd const * h, int taps, int phases, int order,
                                     double multiplier, int normalize) {
  size_t count;
  dd *out, fm1, f1 = zero(), f2 = zero();
  int i, j;
  if(!h || taps <= 0 || phases <= 0 || order < 0 || order > 3 ||
     (size_t)taps > SIZE_MAX / (size_t)phases ||
     (size_t)taps * (size_t)phases > SIZE_MAX / (size_t)(order + 1) /
                                           sizeof(*out)) return NULL;
  count = (size_t)taps * (size_t)phases * (size_t)(order + 1);
  out = calloc(count, sizeof(*out));
  if(!out) return NULL;
  fm1 = h[0];
  for(i = taps - 1; i >= 0; --i) for(j = phases - 1; j >= 0; --j) {
    int pos = i * phases + j - 1;
    dd f0 = fm1, b = zero(), c = zero(), d = zero();
    fm1 = pos > 0?soxr_dd_mul_d(h[pos - 1], multiplier) : zero();
    switch(order) {
    case 1: b = soxr_dd_sub(f1, f0); break;
    case 2: c = soxr_dd_sub(soxr_dd_mul_d(soxr_dd_add(f2, f0), .5), f1);
      b = soxr_dd_sub(soxr_dd_sub(f1, c), f0); break;
    case 3: c = soxr_dd_sub(soxr_dd_mul_d(soxr_dd_add(f1, fm1), .5), f0);
      d = soxr_dd_div(soxr_dd_sub(soxr_dd_sub(soxr_dd_add(f2, fm1),
         soxr_dd_add(f1, f0)), soxr_dd_mul_d(c, 4)), soxr_dd_from(6.));
      b = soxr_dd_sub(soxr_dd_sub(soxr_dd_sub(f1, f0), d), c); break;
    }
    if(order > 2) coef(out, order, taps, j, 3, taps - 1 - i) = d;
    if(order > 1) coef(out, order, taps, j, 2, taps - 1 - i) = c;
    if(order > 0) coef(out, order, taps, j, 1, taps - 1 - i) = b;
    coef(out, order, taps, j, 0, taps - 1 - i) = f0;
    f2 = f1; f1 = f0;
  }
  if(normalize && order == 0) {
    int r, t, other;
    for(r = 0; r <= phases / 2; ++r) {
      int mirror = r?phases - r : 0;
      dd sum = zero(), corr;
      for(t = 0; t < taps; ++t)
        sum = soxr_dd_add(sum, coef(out, 0, taps, r, 0, t));
      corr = soxr_dd_sub(soxr_dd_from(multiplier), sum);
      t = (taps - 1) / 2;
      if(mirror != r) {
        coef(out, 0, taps, r, 0, t) = soxr_dd_add(coef(out, 0, taps, r, 0, t), corr);
        coef(out, 0, taps, mirror, 0, taps - 1 - t) = soxr_dd_add(
             coef(out, 0, taps, mirror, 0, taps - 1 - t), corr);
      } else {
        other = r == 0?taps - 2 : taps - 1;
        t = other / 2;
        if(t != other - t) corr = soxr_dd_mul_d(corr, .5);
        coef(out, 0, taps, r, 0, t) = soxr_dd_add(coef(out, 0, taps, r, 0, t), corr);
        if(t != other - t) coef(out, 0, taps, r, 0, other - t) = soxr_dd_add(
             coef(out, 0, taps, r, 0, other - t), corr);
      }
    }
  }
  return out;
}

static void dd_half(stage_t * p, fifo_t * outfifo) {
  if(p->extreme_fft == soxr_dd_fft_avx2) {
    soxr_dd_half_avx2(p, outfifo);
    return;
  }
  int n = min(stage_occupancy(p), p->input_size), i, j, amount = (n + 1) >> 1;
  dd const * input = (dd const *) fifo_read_ptr(&p->fifo) + p->pre;
  dd * output = fifo_reserve(outfifo, amount);
  if(!output) { soxr_stage_oom(p); return; }
  const double * coefs = (const double *) p->coefs;
  for(i = 0; i < amount; ++i, input += 2) {
    dd sum = soxr_dd_mul_d(input[0], .5);
    for(j = 0; j < p->n; ++j) {
      dd pair = soxr_dd_add(input[- (2 * j + 1)], input[2 * j + 1]);
      sum = soxr_dd_add(sum, soxr_dd_mul_d(pair, coefs[j]));
    }
    output[i] = sum;
  }
  fifo_read(&p->fifo, 2 * amount, NULL);
}

static void dd_poly0(stage_t * p, fifo_t * outfifo) {
  if(p->extreme_fft == soxr_dd_fft_avx2) {
    soxr_dd_poly0_avx2(p, outfifo);
    return;
  }
  if(p->extreme_use_avx) {
    soxr_dd_poly0_avx(p, outfifo);
    return;
  }
  int n = min(stage_occupancy(p), p->input_size);
  if(n) {
    dd const * input = (dd const *) fifo_read_ptr(&p->fifo) + p->pre;
    int at = p->at.integer, step = p->step.integer, i, j;
    int amount = (n * p->L - at + step - 1) / step;
    dd * output = fifo_reserve(outfifo, amount);
    if(!output) { soxr_stage_oom(p); return; }
  if(!output) { soxr_stage_oom(p); return; }
    for(i = 0; at < n * p->L; ++i, at += step) {
      int d = at / p->L, rem = at % p->L;
      dd const * in = input + d;
      dd const * coefs = p->shared->poly_dd + (size_t) p->n * rem;
      dd sum = zero();
      for(j = 0; j < p->n; ++j)
        sum = soxr_dd_add(sum, soxr_dd_mul(in[j], coefs[j]));
      output[i] = sum;
    }
    fifo_read(&p->fifo, at / p->L, NULL);
    p->at.integer = at % p->L;
  }
}

static void dd_poly_interp(stage_t * p, fifo_t * outfifo, int order) {
  if(p->extreme_fft == soxr_dd_fft_avx2) {
    soxr_dd_poly_interp_avx2(p, outfifo, order);
    return;
  }
  int n = min(stage_occupancy(p), p->input_size), i, j;
  int max_out = 1 + (int)(n * p->out_in_ratio);
  dd const * input = (dd const *) fifo_read_ptr(&p->fifo) + p->pre;
  dd * output = fifo_reserve(outfifo, max_out);
  if(!output) { soxr_stage_oom(p); return; }
  step_t at = p->at;
  for(i = 0; at.integer < n; ++i) {
    uint32_t frac = at.fraction;
    int phase = (int)(frac >> (32 - p->phase_bits));
    uint32_t residue = frac << p->phase_bits;
    dd x = soxr_dd_mul_d(soxr_dd_from((double) residue), 1. / 4294967296.);
    dd sum = zero();
    dd const * in = input + at.integer;
    for(j = 0; j < p->n; ++j) {
      dd v = getc(p->shared->poly_dd, order, p->n, phase, order, j);
      int k;
      for(k = order - 1; k >= 0; --k)
        v = soxr_dd_add(soxr_dd_mul(v, x),
             getc(p->shared->poly_dd, order, p->n, phase, k, j));
      sum = soxr_dd_add(sum, soxr_dd_mul(v, in[j]));
    }
    output[i] = sum;
    if(p->use_hi_prec_clock) {
      uint64_t was = at.fix.ls.all;
      at.fix.ls.all += p->step.fix.ls.all;
      at.whole += p->step.whole + (at.fix.ls.all < was);
    } else at.whole += p->step.whole;
  }
  fifo_trim_by(outfifo, max_out - i);
  fifo_read(&p->fifo, at.integer, NULL);
  p->at.whole = at.fraction;
  if(p->use_hi_prec_clock) p->at.fix.ls = at.fix.ls;
}
static void dd_poly1(stage_t * p, fifo_t * out) { dd_poly_interp(p, out, 1); }
static void dd_poly2(stage_t * p, fifo_t * out) { dd_poly_interp(p, out, 2); }
static void dd_poly3(stage_t * p, fifo_t * out) { dd_poly_interp(p, out, 3); }

static void dd_cubic(stage_t * p, fifo_t * outfifo) {
  int n = min(stage_occupancy(p), p->input_size), i, max_out = 1 + (int)(n * p->out_in_ratio);
  dd const * input = (dd const *) fifo_read_ptr(&p->fifo) + p->pre;
  dd * output = fifo_reserve(outfifo, max_out);
  if(!output) { soxr_stage_oom(p); return; }
  for(i = 0; p->at.integer < n; ++i, p->at.whole += p->step.whole) {
    dd const * s = input + p->at.integer;
    dd x = soxr_dd_mul_d(soxr_dd_from((double) p->at.fraction), 1. / 4294967296.);
    dd b = soxr_dd_sub(soxr_dd_mul_d(soxr_dd_add(s[1], s[-1]), .5), s[0]);
    dd a = soxr_dd_div(soxr_dd_sub(soxr_dd_sub(
      soxr_dd_add(soxr_dd_sub(s[2], s[1]), soxr_dd_sub(s[-1], s[0])),
      soxr_dd_mul_d(b, 4)), zero()), soxr_dd_from(6.));
    dd c = soxr_dd_sub(soxr_dd_sub(soxr_dd_sub(s[1], s[0]), a), b);
    output[i] = soxr_dd_mul_d(soxr_dd_add(soxr_dd_mul(
      soxr_dd_add(soxr_dd_mul(soxr_dd_add(soxr_dd_mul(a, x), b), x), c), x), s[0]), p->mult);
  }
  fifo_trim_by(outfifo, max_out - i);
  fifo_read(&p->fifo, p->at.integer, NULL);
  p->at.integer = 0;
}

static void dd_convolve_scalar(zdd * x, zdd const * h, size_t n) {
  for(size_t i = 0; i < n; ++i) {
    dd re = soxr_dd_sub(soxr_dd_mul(x[i].re, h[i].re),
                        soxr_dd_mul(x[i].im, h[i].im));
    dd im = soxr_dd_add(soxr_dd_mul(x[i].re, h[i].im),
                        soxr_dd_mul(x[i].im, h[i].re));
    x[i].re = re;
    x[i].im = im;
  }
}

void soxr_dd_dft_stage(stage_t * p, fifo_t * outfifo) {
  dft_filter_t const * f =&p->shared->dft_filter[p->dft_filter_num];
  int n = f->dft_length, overlap = f->num_taps - 1, num_in = fifo_occupancy(&p->fifo);
  if(p->at.integer + p->L * num_in >= n) {
    div_t divd = div(n - overlap - p->at.integer + p->L - 1, p->L);
    dd const * input = fifo_read_ptr(&p->fifo);
    zdd * x = p->extreme_dft_workspace;
    zdd const * h = (zdd const *) f->dd_spectrum;
    dd * output;
    int i, j;
    if(!x || !h) { soxr_stage_oom(p); return; }
    /* Match the original calloc() contents at the start of every DFT block. */
    memset(x, 0, (size_t) n * sizeof(*x));
    if(p->L == 1) { for(i = 0; i < n; ++i) x[i].re = input[i]; }
    else {
      for(i = p->at.integer, j = 0; i < n; i += p->L, ++j) x[i].re = input[j];
      p->at.integer = p->L - 1 - divd.rem;
    }
    if(p->extreme_fft_plan) soxr_dd_fft_avx2_planned(x, (size_t)n, 0, p->extreme_fft_plan);
    else if(p->extreme_scalar_plan && p->extreme_use_avx)
      soxr_dd_fft_avx_planned(x, (size_t)n, 0, p->extreme_scalar_plan);
    else if(p->extreme_scalar_plan && p->extreme_use_sse2)
      soxr_dd_fft_sse2_planned(x, (size_t)n, 0, p->extreme_scalar_plan);
    else if(p->extreme_scalar_plan)
      soxr_dd_fft_planned(x, (size_t)n, 0, p->extreme_scalar_plan);
    else p->extreme_fft(x, (size_t)n, 0);
    p->extreme_convolve(x, h, (size_t)n);
    if(p->extreme_fft_plan) soxr_dd_fft_avx2_planned(x, (size_t)n, 1, p->extreme_fft_plan);
    else if(p->extreme_scalar_plan && p->extreme_use_avx)
      soxr_dd_fft_avx_planned(x, (size_t)n, 1, p->extreme_scalar_plan);
    else if(p->extreme_scalar_plan && p->extreme_use_sse2)
      soxr_dd_fft_sse2_planned(x, (size_t)n, 1, p->extreme_scalar_plan);
    else if(p->extreme_scalar_plan)
      soxr_dd_fft_planned(x, (size_t)n, 1, p->extreme_scalar_plan);
    else p->extreme_fft(x, (size_t)n, 1);
    output = fifo_reserve(outfifo, n);
    if(!output) { soxr_stage_oom(p); return; }
    if(p->step.integer == 1) {
      for(i = 0; i < n - overlap; ++i) output[i] = x[i].re;
      fifo_trim_by(outfifo, overlap);
    } else {
      for(i = p->remM, j = 0; i < n - overlap; i += p->step.integer, ++j)
        output[j] = x[i].re;
      p->remM = i - (n - overlap);
      fifo_trim_by(outfifo, n - j);
    }
    fifo_read(&p->fifo, divd.quot, NULL);
  }
  p->input_size = (n - p->at.integer + p->L - 1) / p->L;
}


#define PE(a,b,c,d,e) {-1,{{0,dd_poly0},{a,dd_poly1},{b,dd_poly2}}}
static poly_fir_t const dd_poly_firs[] = {
 PE(7.2f, 5.0f, 0, 0, 0), PE(9.4f, 6.7f, 0, 0, 0),
 PE(12.4f, 7.8f, 0, 0, 0), PE(13.6f, 9.3f, 0, 0, 0),
 { -1, {{ 0, dd_poly0 }, { 10.5f, dd_poly2 }, { 8.4f, dd_poly3 } } },
 { -1, {{ 0, dd_poly0 }, { 11.85f, dd_poly2 }, { 9.0f, dd_poly3 } } },
 PE(8.0f, 5.3f, 0, 0, 0), PE(8.6f, 5.7f, 0, 0, 0),
 PE(10.6f, 6.75f, 0, 0, 0), PE(12.6f, 8.6f, 0, 0, 0),
 { -1, {{ 0, dd_poly0 }, { 9.6f, dd_poly2 }, { 7.6f, dd_poly3 } } },
 { -1, {{ 0, dd_poly0 }, { 11.4f, dd_poly2 }, { 8.65f, dd_poly3 } } },
 { 10.62f, {{ 42, dd_poly0 }, { 0, NULL }, { 0, NULL } } },
 { 11.28f, {{ 11, dd_poly0 }, { 8, dd_poly1 }, { 6, dd_poly2 } } },
 PE(9.0f, 6.0f, 0, 0, 0), PE(11.0f, 7.0f, 0, 0, 0),
 PE(13.0f, 8.0f, 0, 0, 0),
 { -1, {{ 0, dd_poly0 }, { 10.f, dd_poly2 }, { 8.f, dd_poly3 } } },
 { -1, {{ 0, dd_poly0 }, { 12.f, dd_poly2 }, { 9.f, dd_poly3 } } }
};
#undef PE
static half_fir_info_t const dd_half_firs[] = {
 { 7, half_fir_coefs_7, dd_half, 0, 120.65f },
 { 8, half_fir_coefs_8, dd_half, 0, 136.51f },
 { 9, half_fir_coefs_9, dd_half, 0, 152.32f },
 { 10, half_fir_coefs_10, dd_half, 0, 168.08f },
 { 11, half_fir_coefs_11, dd_half, 0, 183.79f },
 { 12, half_fir_coefs_12, dd_half, 0, 199.46f },
 { 13, half_fir_coefs_13, dd_half, 0, 215.12f }
};
static const cr_core_t dd_core = {
 { malloc, calloc, free }, dd_half_firs, 7, 0, 0, dd_cubic,
 dd_poly_firs, NULL
};
static char const * dd_create(void * ch, void * shared, double ratio,
   soxr_quality_spec_t * q, soxr_runtime_spec_t * r, double scale) {
 rate_t * engine = (rate_t *) ch;
 /* Select once, before expensive filter/stage setup.  A forced AVX2
  * request must fail safely on an unsupported CPU/OS; AUTO falls back. */
 int const mode = EXTREME_USE_SIMD;
 int const use_avx2 = (mode == EXTREME_SIMD_AUTO || mode == EXTREME_SIMD_AVX2) &&
                      soxr_cpu_has_avx2_fma3();
 int const use_avx = (mode == EXTREME_SIMD_AUTO || mode == EXTREME_SIMD_AVX) &&
                     !use_avx2 && soxr_cpu_has_avx();
 int const use_sse2 = (mode == EXTREME_SIMD_AUTO || mode == EXTREME_SIMD_SSE2) &&
                      !use_avx2 && !use_avx && soxr_cpu_has_sse2();
 if(mode == EXTREME_SIMD_AVX2 && !use_avx2)
   return "EXTREME_USE_SIMD requires unavailable AVX2 + FMA3 CPU/OS support";
 if(mode == EXTREME_SIMD_AVX && !use_avx)
   return "EXTREME_USE_SIMD requires available AVX CPU/OS support";
 if(mode == EXTREME_SIMD_SSE2 && !use_sse2)
   return "EXTREME_USE_SIMD requires SSE2 CPU support";
 char const * error = _soxr_init(ch, shared, ratio, q, r, scale, &dd_core,
                                 CORE_DBL | CORE_DD);
 if(!error) {
   /* The Fast engine's USE_SIMD setting is independent. */
   for(int i = 0; i < engine->num_stages; ++i) {
     stage_t * stage = &engine->stages[i];
     stage->extreme_fft = use_avx2 ? soxr_dd_fft_avx2 : soxr_dd_fft;
     stage->extreme_convolve = use_avx2 ? soxr_dd_convolve_avx2 : dd_convolve_scalar;
     stage->extreme_use_sse2 = !!use_sse2;
     stage->extreme_use_avx = !!use_avx;
     if(use_avx2 && stage->fn == dd_poly0)
       soxr_dd_poly0_pack_create(stage);
     if(stage->fn == soxr_dd_dft_stage) {
       size_t n = (size_t) stage->shared->dft_filter[stage->dft_filter_num].dft_length;
       if(!n || n > SIZE_MAX / sizeof(*stage->extreme_dft_workspace))
         return "EXTREME FFT workspace size invalid";
       stage->extreme_dft_workspace = calloc(n, sizeof(*stage->extreme_dft_workspace));
       if(!stage->extreme_dft_workspace) return "out of memory";
       if(!use_avx2) {
         unsigned const idx = stage->dft_filter_num;
         if(idx >= 2) return "invalid EXTREME FFT plan index";
         if(!stage->shared->extreme_scalar_plans[idx])
           stage->shared->extreme_scalar_plans[idx] = soxr_dd_scalar_plan_create(n);
         if(!stage->shared->extreme_scalar_plans[idx]) return "out of memory";
         stage->extreme_scalar_plan = stage->shared->extreme_scalar_plans[idx];
       }
       if(use_avx2) {
         unsigned const idx = stage->dft_filter_num;
         if(idx >= 2) return "invalid EXTREME FFT plan index";
         if(!stage->shared->extreme_fft_plans[idx])
           stage->shared->extreme_fft_plans[idx] = soxr_dd_avx2_plan_create(n);
         stage->extreme_fft_plan = stage->shared->extreme_fft_plans[idx];
         /* Allocation failure automatically retains the tested legacy FFT. */
       }
     }
   }
 }
 return error;
}
static real * _soxr_dd_input(rate_t * p, real const * samples, size_t n) {
 size_t i;
 dd * dest;
 if(p->flushing || p->runtime_error) return NULL;
 if(n > INT_MAX || n > SIZE_MAX / sizeof(dd)) { p->runtime_error = "input block too large"; return NULL; }
 p->samples_in += (int64_t) n;
 if(!samples) {
   if(n > p->ext_ic) {
     double * t = realloc(p->ext_input, n * sizeof(*t));
     if(!t) { p->runtime_error = "out of memory"; return NULL; }
     p->ext_input = t; p->ext_ic = n;
   }
   p->ext_pending = n;
   return (real *) p->ext_input;
 }
 dest = fifo_reserve(&p->stages[0].fifo, (int) n);
 if(!dest) { p->runtime_error = "out of memory"; return NULL; }
 for(i = 0; i < n; ++i) dest[i] = soxr_dd_from(((const double *) samples)[i]);
 return (real *) p->ext_input;
}
static void _soxr_dd_process(rate_t * p, size_t n) {
 if(p->runtime_error) return;
 if(p->ext_pending) {
   size_t i, sz = p->ext_pending;
   dd * dest = fifo_reserve(&p->stages[0].fifo, (int) sz);
   if(!dest) { p->runtime_error = "out of memory"; return; }
   for(i = 0; i < sz; ++i) dest[i] = soxr_dd_from(p->ext_input[i]);
   p->ext_pending = 0;
 }
 _soxr_process(p, n);
}
static real const * _soxr_dd_output(rate_t * p, real * samples, size_t * n) {
 if(p->runtime_error) { *n = 0; return NULL; }
 fifo_t * f =&p->stages[p->num_stages].fifo;
 int size = p->flushing?min(- (int) p->samples_out, (int) * n) :(int) * n;
 int k = min(size, fifo_occupancy(f)), i;
 dd const * src = fifo_read_ptr(f);
 double * out;
 if(k < 0) k = 0;
 if(samples) out = (double *) samples;
 else {
   if((size_t) k > p->ext_oc) {
     double * t = realloc(p->ext_output, (size_t) k * sizeof(double));
     if(!t) { *n = 0; p->runtime_error = "out of memory"; return NULL; }
     p->ext_output = t; p->ext_oc = (size_t) k;
   }
   out = p->ext_output;
 }
 for(i = 0; i < k; ++i) out[i] = soxr_dd_value(src[i]);
 fifo_read(f, k, NULL);
 p->samples_out += k;
 *n = k;
 return (real const *) out;
}
static char const * dd_name(void) { return "cr64e-full-dd"; }
static void * dd_typed_input(void * state, void const * samples, size_t n)
{
  return _soxr_dd_input((rate_t *) state, (real const *) samples, n);
}
static void dd_typed_process(void * state, size_t n)
{
  _soxr_dd_process((rate_t *) state, n);
}
static void const * dd_typed_output(void * state, void * samples, size_t * n)
{
  return _soxr_dd_output((rate_t *) state, (real *) samples, n);
}
static void dd_typed_flush(void * state)
{
  _soxr_flush((rate_t *) state);
}
static void dd_typed_close(void * state)
{
  _soxr_close((rate_t *) state);
}
static double dd_typed_delay(void * state)
{
  return _soxr_delay((rate_t *) state);
}
static char const * dd_typed_status(void * state)
{
  return _soxr_runtime_error((rate_t *) state);
}
soxr_rate_callbacks_t const _soxr_rate64e_cb = {
  dd_typed_input, dd_typed_process, dd_typed_output, dd_typed_flush,
  dd_typed_close, dd_typed_delay, _soxr_sizes, dd_create, NULL,
  dd_name, dd_typed_status
};
