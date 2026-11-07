/* SoX Resampler Library      Copyright (c) 2007-16 robs@users.sourceforge.net
 * Licence for this file: LGPL v2.1                  See LICENCE for details.
 *
 * Constant-rate resampling common code. */

/*
 * SoXR-Poikosoft | Enhanced SoX Resampler Library
 *
 * Poikosoft integration and enhancements Copyright (c) 2026
 * Jukka Poikolainen / Poikosoft | https://www.poikosoft.com
 *
 * Module: Fast coefficient preparation and stage setup
 * Features:
 *  - Precision-focused FIR coefficient preparation.
 *  - Polyphase and FFT stages for constant-rate conversion.
 *
 * Original third-party copyrights and license terms are preserved.
 */



#include <math.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <limits.h>

#include "filter.h"
#include "soxr-config.h"

  typedef double soxr_coeff_work_t;

#if defined SOXR_LIB
  #include "internal.h"
  #define STATIC
#endif

#include "cr.h"
#include "soxr-dd-fft.h"
#include "pffft-backend-util.h"

#define num_coefs4 ((core_flags&CORE_SIMD_POLY)? ((num_coefs+3)&~3) : num_coefs)

#define coef_coef(C,T,x) \
  C((T*)result, interp_order, num_coefs4, j, x, num_coefs4 - 1 - i)

#define STORE(C,T) { \
  if (interp_order > 2) coef_coef(C,T,3) = (T)d; \
  if (interp_order > 1) coef_coef(C,T,2) = (T)c; \
  if (interp_order > 0) coef_coef(C,T,1) = (T)b; \
  coef_coef(C,T,0) = (T)f0;}

#if 0
// Original SoXR 0.1.3 use this
static real * prepare_poly_fir_coefs(soxr_coeff_work_t const * coefs, int num_coefs,
    int num_phases, int interp_order, double multiplier,
    core_flags_t core_flags, alloc_t const * mem, bool normalize_dc)
{
  int i, j, length;
  size_t const coef_bytes = (size_t)(1u << LOG2_SIZEOF_REAL(core_flags));
  real *result;
  if(!coefs || num_coefs <= 0 || num_phases <= 0 || interp_order < 0 ||
      interp_order > 3 || num_coefs4 > INT_MAX / num_phases /
                              (interp_order + 1)) return NULL;
  length = num_coefs4 * num_phases * (interp_order + 1);
  result = mem->calloc((size_t) length, coef_bytes);
  if(!result) return NULL;
  soxr_coeff_work_t fm1 = coefs[0], f1 = 0, f2 = 0;

  for(i = num_coefs - 1; i >= 0; --i)
    for(j = num_phases - 1; j >= 0; --j) {
      soxr_coeff_work_t f0 = fm1, b = 0, c = 0, d = 0;
      int pos = i * num_phases + j - 1;
      fm1 = pos > 0 ?(soxr_coeff_work_t) coefs[pos - 1] * multiplier : 0;
      switch(interp_order) {
        case 1: b = f1 - f0; break;
        case 2: b = f1 - (.5 * (f2 + f0) - f1) - f0; c = .5 * (f2 + f0) - f1; break;
        case 3: c = .5 * (f1 + fm1) - f0; d = (1.0L / 6) * (f2 - f1 + fm1 - f0 - 4 * c); b = f1 - f0 - d - c; break;
        default: assert(!interp_order);
      }
      switch(core_flags & 3) {
        case 0: if(WITH_CR32) STORE(coef, float); break;
        case 1: if(WITH_CR64) {
            STORE(coef, double)
          } break;
        default: break; /* Only cr32/cr64 coefficient layouts are used. */
      }
      f2 = f1, f1 = f0;
    }
  return result;
}
#endif


static real * prepare_poly_fir_coefs_dd(const soxr_dd_t * coefs,
    int num_coefs, int num_phases, int interp_order, double multiplier,
    alloc_t const * mem, bool normalize_dc, real **low_out, core_flags_t core_flags)
{
  int i, j, length;
  double *hi;
  if(!coefs || num_coefs <= 0 || num_phases <= 0 || interp_order < 0 ||
     interp_order > 3 || num_coefs > INT_MAX / num_phases /
                                  (interp_order + 1)) return NULL;
  length = num_coefs * num_phases * (interp_order + 1);
  hi = mem->calloc((size_t) length, sizeof(*hi));
  double * lo = NULL;
  soxr_dd_t fm1 = coefs[0], f1 = soxr_dd_from(0), f2 = soxr_dd_from(0);
  if(!hi) return NULL;
#define SETDD(k, v) do { \
    int pos = (&coef(hi, interp_order, num_coefs, j, k, num_coefs - 1 - i)) - hi; \
    soxr_dd_t tmp = (v); \
    hi[pos] = tmp.hi; \
    if(lo) lo[pos] = tmp.lo; \
  } while(0)
  for(i = num_coefs - 1; i >= 0; --i) for(j = num_phases - 1; j >= 0; --j) {
    soxr_dd_t f0 = fm1, b = soxr_dd_from(0), c = soxr_dd_from(0), d = soxr_dd_from(0);
    int pos = i * num_phases + j - 1;
    fm1 = pos > 0?soxr_dd_mul_d(coefs[pos - 1], multiplier) : soxr_dd_from(0);
    switch(interp_order) {
      case 1: b = soxr_dd_sub(f1, f0); break;
      case 2: c = soxr_dd_sub(soxr_dd_mul_d(soxr_dd_add(f2, f0), .5), f1);
        b = soxr_dd_sub(soxr_dd_sub(f1, c), f0); break;
      case 3: c = soxr_dd_sub(soxr_dd_mul_d(soxr_dd_add(f1, fm1), .5), f0);
        d = soxr_dd_mul_d(soxr_dd_sub(soxr_dd_sub(soxr_dd_add(f2, fm1), soxr_dd_add(f1, f0)), soxr_dd_mul_d(c, 4)), 1. / 6.);
        b = soxr_dd_sub(soxr_dd_sub(soxr_dd_sub(f1, f0), d), c); break;
    }
    if(interp_order > 2) SETDD(3, d);
    if(interp_order > 1) SETDD(2, c);
    if(interp_order > 0) SETDD(1, b);
    SETDD(0, f0);
    f2 = f1; f1 = f0;
  }
#undef SETDD
  /* Exact rational rows: pairwise mirror-symmetric DC correction.
     In paired mode retain the sub-ulp residual in the low coefficient. */
  if(normalize_dc && interp_order == 0) {
    int r, t;
    for(r = 0; r <= num_phases / 2; ++r) {
      int mirror = r?num_phases - r : 0, other;
      soxr_dd_t sum = soxr_dd_from(0), corr;
      for(t = 0; t < num_coefs; ++t) {
        int idx = (&coef(hi, 0, num_coefs, r, 0, t)) - hi;
        sum = soxr_dd_add(sum, soxr_dd_make(hi[idx], lo?lo[idx] : 0));
      }
      corr = soxr_dd_sub(soxr_dd_from(multiplier), sum);
      t = (num_coefs - 1) / 2;
      if(mirror != r) {
        int a = (&coef(hi, 0, num_coefs, r, 0, t)) - hi;
        int b = (&coef(hi, 0, num_coefs, mirror, 0, num_coefs - 1 - t)) - hi;
        soxr_dd_t v = soxr_dd_add(soxr_dd_make(hi[a], lo?lo[a] : 0), corr);
        hi[a] = v.hi;
        if(lo) lo[a] = v.lo;
        v = soxr_dd_add(soxr_dd_make(hi[b], lo?lo[b] : 0), corr);
        hi[b] = v.hi;
        if(lo) lo[b] = v.lo;
      } else {
        other = r == 0?num_coefs - 2 : num_coefs - 1;
        t = other / 2;
        if(t != other - t) corr = soxr_dd_mul_d(corr, .5);
        for(i = 0; i < (t == other - t?1 : 2); ++i) {
          int idx = (&coef(hi, 0, num_coefs, r, 0, i?other - t : t)) - hi;
          soxr_dd_t v = soxr_dd_add(soxr_dd_make(hi[idx], lo?lo[idx] : 0), corr);
          hi[idx] = v.hi;
          if(lo) lo[idx] = v.lo;
        }
      }
    }
  }
  /* The original DD designer produces double coefficients.  Scalar
     CORE_FLOAT32 needs a properly packed float array, not a double*
     reinterpreted as float* (which produces Inf/NaN and huge samples). */
  if(!(core_flags & CORE_DBL)) {
    float * packed = mem->calloc((size_t) length, sizeof(float));
    if(!packed) { mem->free(hi); if(lo) mem->free(lo); return NULL; }
    for(int k = 0; k < length; ++k) packed[k] = (float)(hi[k] + (lo?lo[k] : 0));
    mem->free(hi);
    if(lo) mem->free(lo);
    *low_out = NULL;
    return (real *) packed;
  }
  *low_out = (real *) lo;
  return (real *) hi;
}


#undef STORE
#undef coef_coef

#define IS_FLOAT32 WITH_CR32 && \
    (!(WITH_CR64) || sizeof_real == sizeof(float))
#define WITH_FLOAT64 WITH_CR64


static void dft_stage_fn(stage_t * p, fifo_t * output_fifo)
{
  real * output, *dft_out;
  int i, j, num_in = max(0, fifo_occupancy(&p->fifo));
  rate_shared_t const * s = p->shared;
  dft_filter_t const * f = &s->dft_filter[p->dft_filter_num];
  int const overlap = f->num_taps - 1;
  int l2_convolved = 0;

  if(p->at.integer + p->L * num_in >= f->dft_length) {
    soxr_rdft_callbacks_t const * const RDFT_CB = p->rdft_cb;
    size_t const sizeof_real = sizeof(char) << LOG2_SIZEOF_REAL(p->core_flags);
    div_t divd = div(f->dft_length - overlap - p->at.integer + p->L - 1, p->L);
    real const * input = fifo_read_ptr(&p->fifo);
    fifo_read(&p->fifo, divd.quot, NULL);
    num_in -= divd.quot;

    output = fifo_reserve(output_fifo, f->dft_length);
    if(!output) { soxr_stage_oom(p); return; }
    dft_out = (p->core_flags & CORE_SIMD_DFT)? p->dft_out : output;

    if(lsx_is_power_of_2(p->L)) { /* F-domain */
      int portion = f->dft_length / p->L;
      memcpy(dft_out, input, (unsigned) portion * sizeof_real);
      rdft_oforward(portion, f->dft_forward_setup, dft_out, p->dft_scratch);
      if((rdft_flags() & RDFT_L2_FUSED) && p->L == 2 && p->step.integer > 0) {
        rdft_l2_fused(f->dft_length, f->dft_backward_setup,
            dft_out, p->dft_scratch, f->coefs);
        dft_out = (real *) ((char *) p->dft_scratch +
                   (size_t) f->dft_length * sizeof_real);
        l2_convolved = 1;
      }
      else {
        if(IS_FLOAT32) {
#define dft_out ((float *)dft_out)
          for(i = portion + 2; i < (portion << 1); i += 2) /* Mirror image. */
            dft_out[i] = dft_out[(portion << 1) - i],
              dft_out[i + 1] = -dft_out[(portion << 1) - i + 1];
          dft_out[portion] = dft_out[1];
          dft_out[portion + 1] = 0;
          dft_out[1] = dft_out[0];
#undef dft_out
        }
        else if(WITH_FLOAT64) {
#define dft_out ((double *)dft_out)
          for(i = portion + 2; i < (portion << 1); i += 2) /* Mirror image. */
            dft_out[i] = dft_out[(portion << 1) - i],
              dft_out[i + 1] = -dft_out[(portion << 1) - i + 1];
          dft_out[portion] = dft_out[1];
          dft_out[portion + 1] = 0;
          dft_out[1] = dft_out[0];
#undef dft_out
        }

        for(portion <<= 1; i < f->dft_length; i += portion, portion <<= 1) {
          memcpy((char *) dft_out + (size_t) i * sizeof_real, dft_out, (size_t) portion * sizeof_real);
          if(IS_FLOAT32)
        #define dft_out ((float *)dft_out)
            dft_out[i + 1] = 0;
        #undef dft_out
          else if(WITH_FLOAT64)
        #define dft_out ((double *)dft_out)
            dft_out[i + 1] = 0;
        #undef dft_out
        }
        if(p->step.integer > 0)
          rdft_reorder_back(f->dft_length, f->dft_backward_setup, dft_out, p->dft_scratch);
      }
    } else {
      if(p->L == 1)
        memcpy(dft_out, input, (size_t) f->dft_length * sizeof_real);
      else {
        memset(dft_out, 0, (size_t) f->dft_length * sizeof_real);
        if(IS_FLOAT32)
          for(j = 0, i = p->at.integer; i < f->dft_length; ++j, i += p->L)
            ((float *) dft_out)[i] = ((float *) input)[j];
        else if(WITH_FLOAT64)
          for(j = 0, i = p->at.integer; i < f->dft_length; ++j, i += p->L)
            ((double *) dft_out)[i] = ((double *) input)[j];
        p->at.integer = p->L - 1 - divd.rem;
      }
      if(p->step.integer > 0)
        rdft_forward(f->dft_length, f->dft_forward_setup, dft_out, p->dft_scratch);
      else
        rdft_oforward(f->dft_length, f->dft_forward_setup, dft_out, p->dft_scratch);
    }

    if(p->step.integer > 0) {
      if(!l2_convolved)
        rdft_convolve(f->dft_length, f->dft_backward_setup, dft_out, f->coefs);
      rdft_backward(f->dft_length, f->dft_backward_setup, dft_out, p->dft_scratch);
      if((p->core_flags & CORE_SIMD_DFT) && p->step.integer == 1)
        memcpy(output, dft_out, (size_t) f->dft_length * sizeof_real);
      if(p->step.integer != 1) {
        if(IS_FLOAT32)
          for(j = 0, i = p->remM; i < f->dft_length - overlap; ++j,
              i += p->step.integer)
            ((float *) output)[j] = ((float *) dft_out)[i];
        else if(WITH_FLOAT64)
          for(j = 0, i = p->remM; i < f->dft_length - overlap; ++j,
              i += p->step.integer)
            ((double *) output)[j] = ((double *) dft_out)[i];
        p->remM = i - (f->dft_length - overlap);
        fifo_trim_by(output_fifo, f->dft_length - j);
      }
      else fifo_trim_by(output_fifo, overlap);
    }
    else { /* F-domain */
      int m = -p->step.integer;
      rdft_convolve_portion(f->dft_length >> m, dft_out, f->coefs);
      rdft_obackward(f->dft_length >> m, f->dft_backward_setup, dft_out, p->dft_scratch);
      if(p->core_flags & CORE_SIMD_DFT)
        memcpy(output, dft_out, (size_t)(f->dft_length >> m) * sizeof_real);
      fifo_trim_by(output_fifo, (((1 << m) - 1) * f->dft_length + overlap) >> m);
    }
    if(rdft_oom_status()) { soxr_stage_oom(p); return; }
    (void) RDFT_CB;
  }
  p->input_size = (f->dft_length - p->at.integer + p->L - 1) / p->L;
}



/* Set to 4 x nearest power of 2 or half of that */
/* if danger of causing too many cache misses. */
static int set_dft_length(int num_taps, int min, int large)
{
  double d = log((double) num_taps) / log(2.);
  return 1 << range_limit((int)(d + 2.77), min, max((int)(d + 1.77), large));
}

static void dft_stage_init(
    unsigned instance, double Fp, double Fs, double Fn, double att,
    double phase_response, stage_t * p, int L, int M, double * multiplier,
    unsigned min_dft_size, unsigned large_dft_size, core_flags_t core_flags,
    soxr_rdft_callbacks_t const * RDFT_CB)
{
  dft_filter_t * f = &p->shared->dft_filter[instance];
  int num_taps = 0, dft_length = f->dft_length, i, offset;
  /* Partial-spectrum shortcut depends on the FFT backend packing.
     PFFFT's ordered/unordered transforms support this only via the paired
     full-spectrum preparation and partial-convolution handlers. */
  bool f_domain_m = abs(3 - M) == 1 && Fs <= 1 &&
      !(core_flags & CORE_DD);
  size_t const sizeof_real = sizeof(char) << LOG2_SIZEOF_REAL(core_flags);

  {
  if(!dft_length) {
    int k = phase_response == 50 && lsx_is_power_of_2(L) && Fn == L? L << 1 : 4;
    double m, *h = NULL;
    if(core_flags & CORE_DD) {
      soxr_dd_t * hdd;
      soxr_dd_t sum = soxr_dd_from(0), gain;
      soxr_dd_complex * sp;
      int n = 0;
      if(phase_response != 50) { p->fn = NULL; return; } /* DD needs linear phase */
      hdd = lsx_design_lpf_extreme_dd(Fp, Fs, Fn, att, &n, -k, -1., 0, 0);
      if(!hdd) return;
      dft_length = set_dft_length(n, (int) min_dft_size, (int) large_dft_size);
      sp = calloc((size_t) dft_length, sizeof(*sp));
      if(!sp) { free(hdd); return; }
      for(i = 0; i < n; ++i) sum = soxr_dd_add(sum, hdd[i]);
      gain = soxr_dd_div(soxr_dd_from(L * *multiplier), sum);
      offset = dft_length - n + 1;
      for(i = 0; i < n; ++i) sp[(i + offset) & (dft_length - 1)].re = soxr_dd_mul(hdd[i], gain);
      free(hdd);
      soxr_dd_fft(sp, (size_t) dft_length, 0);
      f->dd_spectrum = (soxr_dd_t *) sp;
      f->num_taps = n; f->dft_length = dft_length; f->post_peak = n / 2;
    }
    if(!(core_flags & CORE_DD)) {

    soxr_dd_t * hdd = NULL;
    if((core_flags & CORE_DBL) && phase_response == 50)
      hdd = lsx_design_lpf_dd(Fp, Fs, Fn, att, &num_taps, -k, -1.,
           0, 0);
    if(!hdd && (core_flags & CORE_DBL) && phase_response == 50) {
      p->fn = NULL; return;
    }
    if(hdd) {
      int j; h = malloc((size_t) num_taps * sizeof(*h));
      if(!h) { free(hdd); p->fn = NULL; return; }
      for(j = 0; j < num_taps; ++j) h[j] = soxr_dd_value(hdd[j]);
    }
    else
    h = lsx_design_lpf(Fp, Fs, Fn, att, &num_taps, -k, -1.);

    if(!h) { free(hdd); p->fn = NULL; return; }
    if(phase_response != 50) {
      if(!lsx_fir_to_phase(&h, &num_taps, &f->post_peak, phase_response)) {
        free(h); free(hdd); p->fn = NULL; return;
      }
    }
    else f->post_peak = num_taps / 2;

    dft_length = set_dft_length(num_taps, (int) min_dft_size, (int) large_dft_size);
    f->coefs = rdft_calloc((size_t) dft_length, sizeof_real);
    if(!f->coefs) { free(h); free(hdd); p->fn = NULL; return; }
    offset = dft_length - num_taps + 1;
    m = (1. / dft_length) * rdft_multiplier() * L * *multiplier;
    if(hdd && (core_flags & CORE_DBL)) {
      soxr_dd_t sum = soxr_dd_from(0), fac;
      for(i = 0; i < num_taps; ++i) sum = soxr_dd_add(sum, hdd[i]);
      fac = soxr_dd_div(soxr_dd_from(rdft_multiplier() * L * *multiplier),
            soxr_dd_mul_d(sum, dft_length));
      for(i = 0; i < num_taps; ++i) {
        int idx = (i + offset) & (dft_length - 1);
        soxr_dd_t v = soxr_dd_mul(hdd[i], fac);
        ((double *) f->coefs)[idx] = v.hi;
      }
    } else
    if(IS_FLOAT32) for(i = 0; i < num_taps; ++i)
        ((float *) f->coefs)[(i + offset) & (dft_length - 1)] = (float)(h[i] * m);
    else if(WITH_FLOAT64) for(i = 0; i < num_taps; ++i)
        ((double *) f->coefs)[(i + offset) & (dft_length - 1)] = h[i] * m;
    free(h);
    free(hdd);
    } /* normal FIR spectral preparation */
  }

  if((core_flags & CORE_DD)) {
    f->dft_length = dft_length;
    p->dft_out = NULL; p->dft_scratch = NULL;
  } else if(rdft_flags() & RDFT_IS_SIMD)
    p->dft_out = rdft_malloc(sizeof_real * (size_t) dft_length);
  if(!(core_flags & CORE_DD) && (rdft_flags() & RDFT_NEEDS_SCRATCH))
    p->dft_scratch = rdft_malloc(2 * sizeof_real * (size_t) dft_length);

  if(!(core_flags & CORE_DD) &&
      ((rdft_flags() & RDFT_IS_SIMD && !p->dft_out) ||
       (rdft_flags() & RDFT_NEEDS_SCRATCH && !p->dft_scratch))) {
    p->fn = NULL; return;
  }
  if(!(core_flags & CORE_DD) && !f->dft_length) {
    void * coef_setup = rdft_forward_setup(dft_length);
    int Lp = lsx_is_power_of_2(L)? L : 1;
    int Mp = f_domain_m? M : 1;
    f->dft_forward_setup = rdft_forward_setup(dft_length / Lp);
    f->dft_backward_setup = rdft_backward_setup(dft_length / Mp);
    if(!coef_setup || !f->dft_forward_setup || !f->dft_backward_setup) {
      rdft_delete_setup(coef_setup);
      p->fn = NULL; return;
    }
    if(Mp == 1)
    {
      rdft_forward(dft_length, coef_setup, f->coefs, p->dft_scratch);
    }
    else
    {
      rdft_oforward(dft_length, coef_setup, f->coefs, p->dft_scratch);
    }
    rdft_delete_setup(coef_setup);
    if(rdft_oom_status()) { p->fn = NULL; return; }
    f->num_taps = num_taps;
    f->dft_length = dft_length;
    lsx_debug("fir_len=%i dft_length=%i Fp=%g Fs=%g Fn=%g att=%g %i/%i",
        num_taps, dft_length, Fp, Fs, Fn, att, L, M);
  }
  }
  *multiplier = 1;
  p->out_in_ratio = (double) L / M;
  p->core_flags = core_flags;
  p->rdft_cb = RDFT_CB;
    p->fn = (core_flags & CORE_DD) ? soxr_dd_dft_stage : dft_stage_fn;
  p->preload = f->post_peak / L;
  p->at.integer = f->post_peak % L;
  p->L = L;
  p->step.integer = f_domain_m? -M / 2 : M;
  p->dft_filter_num = instance;
  p->block_len = f->dft_length - (f->num_taps - 1);
  p->phase0 = p->at.integer / p->L;
  p->input_size = (f->dft_length - p->at.integer + p->L - 1) / p->L;
}

static struct half_fir_info const * find_half_fir(
    struct half_fir_info const * firs, size_t len, double att)
{
  size_t i;
  for(i = 0; i + 1 < len && att > firs[i].att; ++i);
  return &firs[i];
}

#define have_pre_stage  (preM  * preL  != 1)
#define have_arb_stage  (arbM  * arbL  != 1)
#define have_post_stage (postM * postL != 1)

#include "soxr.h"

STATIC char const * _soxr_init(
  rate_t * const p, /* Per audio channel. */
  rate_shared_t * const shared, /* By channels undergoing same rate change. */
  double const io_ratio, /* Input rate divided by output rate. */
  soxr_quality_spec_t const * const q_spec,
  soxr_runtime_spec_t const * const r_spec,
  double multiplier, /* Linear gain to apply during conversion. */
  cr_core_t const * const core,
  core_flags_t const core_flags)
{
  size_t const sizeof_real = sizeof(char) << LOG2_SIZEOF_REAL(core_flags);
  double const tolerance = 1 + 1e-5;

  double bits = q_spec->precision;
  rolloff_t const rolloff = (rolloff_t)(q_spec->flags & 3);
  int interpolator = (int)(r_spec->flags & 3) - 1;
  double const Fp0 = q_spec->passband_end, Fs0 = q_spec->stopband_begin;
  double const phase_response = q_spec->phase_response, tbw0 = Fs0 - Fp0;
  if((core_flags & CORE_DD) && phase_response != 50)
    return "SOXR_EXTREME requires a linear-phase filter";

  bool const maintain_3dB_pt = !!(q_spec->flags & SOXR_MAINTAIN_3DB_PT);
  double tbw_tighten = 1, alpha;
  #define tighten(x) (Fs0-(Fs0-(x))*tbw_tighten)

  /* Filter attenuation and interpolation topology are independent choices.
     The old 33-bit clamp guarded a SIX-entry mode table, not FIR precision.
     Keep the valid 1..6 mode indices (6 selects the high-quality rational
     polyphase entry), while using the full requested bits for attenuation.
     Extending the integer mode beyond 6 would index the wrong filter table. */
  double arbM = io_ratio, Fn1, Fp1 = Fp0, Fs1 = Fs0;
  /* The mode-index lookup is independently bounded to 1..6.  Preserve
     the complete 80-bit *attenuation target* only in the DD engine. */
  double const bits1 = min(bits, (core_flags & CORE_DD) ? 80 : 64);
  double att = (bits1 + 1) * linear_to_dB(2.), attArb = att;
  int preL = 1, preM = 1, shr = 0, arbL = 1, postL = 1, postM = 1;
  bool upsample = false, rational = false, iOpt =!(r_spec->flags & SOXR_NOSMALLINTOPT);
  bool lq_bits = (q_spec->flags & SOXR_PROMOTE_TO_LQ)? bits <= 16 : bits == 16;
  bool lq_Fp0 = (q_spec->flags & SOXR_PROMOTE_TO_LQ)? Fp0 <= lq_bw0 : Fp0 == lq_bw0;
  int n = 0, i;
  int mode = lq_bits && rolloff == rolloff_medium ? io_ratio > 1 ||
    phase_response != 50 || !lq_Fp0 || Fs0 != 1:
    min(6, (int) ceil(2 + (bits1 - 17) / 4));

  struct half_fir_info const * half_fir_info;
  stage_t * s;

  if(io_ratio < 1 && Fs0 - 1 > 1 - Fp0 / tolerance)
    return "imaging greater than rolloff";
  if(.002 / tolerance > tbw0 || tbw0 > .5 * tolerance)
    return "transition bandwidth not in [0.2,50] % of nyquist";
  if(.5 / tolerance > Fp0 || Fs0 > 1.5 * tolerance)
    return "transition band not within [50,150] % of nyquist";

  if(bits != 0 && (15 > bits || bits > ((core_flags & CORE_DD) ? 80 : 64)))
    return (core_flags & CORE_DD) ?
       "Extreme filter design precision not in [15,80] bits":
       "Fast filter design precision not in [15,64] bits";

  if(io_ratio <= 0)
    return "resampling factor not positive";
  if(0 > phase_response || phase_response > 100)
    return "phase response not in [0=min-phase,100=max-phase] %";

  p->core = core;
  p->io_ratio = io_ratio;
  if(bits != 0) while(!n++) { /* Determine stages: */
    int try, L, M, x, maxL = interpolator > 0? 1 : mode? 2048 :
      (int) ceil(r_spec->coef_size_kbytes * 1000. / (U100_l * (int) sizeof_real));
    double d, epsilon = 0, frac;
    upsample = arbM < 1;
    for(i = (int)(.5 * arbM), shr = 0; i >>= 1; arbM *= .5, ++shr);
    preM = upsample || (arbM > 1.5 && arbM < 2);
    postM = 1 + (arbM > 1 && preM), arbM /= postM;
    preL = 1 + (!preM && arbM < 2) + (upsample && mode), arbM *= preL;
    if((frac = arbM - (int) arbM) != 0)
      epsilon = fabs(floor(frac * MULT32 + .5) / (frac * MULT32) - 1);
    for(i = 1, rational = frac == 0; i <= maxL && !rational; ++i) {
      d = frac * i, try = (int)(d + .5);
      if((rational = fabs(try / d - 1) <= epsilon)) { /* No long doubles! */
        if(try == i)
          arbM = ceil(arbM), shr += x = arbM > 3, arbM /= 1 + x;
        else arbM = i * (int) arbM + try, arbL = i;
      }
    }
    L = preL * arbL, M = (int)(arbM * postM), x = (L | M) & 1, L >>= !x, M >>= !x;
    if(iOpt && postL == 1 && (d = preL * arbL / arbM) > 4 && d != 5) {
      for(postL = 4, i = (int)(d / 16); (i >>= 1) && postL < 256; postL <<= 1);
      arbM = arbM * postL / arbL / preL, arbL = 1, n = 0;
    } else if(rational && (max(L, M) < 3 + 2 * iOpt || L * M < 6 * iOpt))
      preL = L, preM = M, arbM = arbL = postM = 1;
    if(!mode && (!rational || !n))
      ++mode, n = 0;
  }

  p->num_stages = shr + have_pre_stage + have_arb_stage + have_post_stage;
  if(!p->num_stages && multiplier != 1) {
    bits = arbL = 0; /* Use cubic_stage in this case. */
    ++p->num_stages;
  }
  p->stages = calloc((size_t) p->num_stages + 1, sizeof(*p->stages));
  if(!p->stages)
    return "out of memory";
  for(i = 0; i <= p->num_stages; ++i)
    p->stages[i].runtime_error = &p->runtime_error;
  for(i = 0; i < p->num_stages; ++i) {
    p->stages[i].num = i;
    p->stages[i].shared = shared;
    p->stages[i].input_size = 8192;
  }
  p->stages[0].is_input = true;

  alpha = postM / (io_ratio * (postL << 0));

  if((n = p->num_stages) > 1) { /* Att. budget: */
    if(have_arb_stage)
      att += linear_to_dB(2.), attArb = att, --n;
    att += linear_to_dB((double) n);
  }

  half_fir_info = find_half_fir(core->half_firs, core->half_firs_len, att);
  for(i = 0, s = p->stages; i < shr; ++i, ++s) {
    s->fn = half_fir_info->fn;
    s->coefs = half_fir_info->coefs;
    s->n = half_fir_info->num_coefs;
    s->pre_post = 4 * s->n;
    s->preload = s->pre = s->pre_post >> 1;
  }

  if(have_pre_stage) {
    if(maintain_3dB_pt && have_post_stage) { /* Trans. bands overlapping. */
      double x = tbw0 * lsx_inv_f_resp(-3.,
          (core_flags & CORE_DD) ? min(att, 400.) : att);
      x = -lsx_f_resp(x / (max(2 * alpha - Fs0, alpha) - Fp0),
          (core_flags & CORE_DD) ? min(att, 400.) : att);
      if(x > .035) {
        tbw_tighten = ((4.3074e-3 - 3.9121e-4 * x) * x - .040009) * x + 1.0014;
        lsx_debug("tbw_tighten=%g (%gdB)", tbw_tighten, x);
      }
    }
    Fn1 = preM? max(preL, preM) : arbM / arbL;
    dft_stage_init(0, tighten(Fp1), Fs1, Fn1,
        att, phase_response, s++, preL,
        max(preM, 1), &multiplier, r_spec->log2_min_dft_size,
        r_spec->log2_large_dft_size, core_flags, core->rdft_cb);
    if(!p->stages[s - p->stages - 1].fn) return "out of memory";
    Fp1 /= Fn1, Fs1 /= Fn1;
  }

  if(bits == 0 && have_arb_stage) { /* `Quick' cubic arb stage: */
    s->fn = core->cubic_stage_fn;
    s->mult = multiplier, multiplier = 1;
    s->step.whole = (int64_t)(arbM * MULT32 + .5);
    s->pre_post = max(3, s->step.integer);
    s->preload = s->pre = 1;
    s->out_in_ratio = MULT32 / (double) s->step.whole;
  }
  else if(have_arb_stage) { /* Higher quality arb stage: */
    static const float rolloffs[] = { -.01f, -.3f, 0, -.103f };
    poly_fir_t const * f = &core->poly_firs[6 * (upsample +!!preM) + mode -!upsample];
    int order, num_coefs = (int) f->interp[0].scalar, phase_bits, phases;
    size_t coefs_size;
    double at, Fp = Fp1, Fs, Fn, mult = upsample? 1 : arbM / arbL;
    poly_fir1_t const * f1;

    if(!upsample && preM)
      Fn = 2 * mult, Fs = 3 + fabs(Fs1 - 1);
    else Fn = 1, Fs = 2 - (mode? Fp1 + (Fs1 - Fp1) * .7 : Fs1);

    if(mode)
      /* The original SoX response-fit polynomial is not valid when its
       * attenuation argument approaches 500 dB. Reuse its calibrated
       * transition shape, NOT a reduced filter-rejection budget. */
      Fp = Fs - (Fs - Fp) / (1 - lsx_inv_f_resp(rolloffs[rolloff],
            (core_flags & CORE_DD) ? min(attArb, 400.) : attArb));

    i = (interpolator < 0? !rational : max(interpolator, !rational)) - 1;
    /* Exact rational ratios such as 96k -> 44.1k (147/320) use
       phase taps rather than polynomial interpolation. */
    do {
      f1 = &f->interp[++i];
      assert(f1->fn);
      if(i)
        arbM /= arbL, arbL = 1, rational = false;
      phase_bits = (int) ceil(f1->scalar - log(mult) / log(2.));
      phases = !rational?(1 << phase_bits) : arbL;
      if(f->interp[0].scalar == 0) {
        int phases0 = max(phases, 19), n0 = 0;
        lsx_design_lpf(Fp, Fs, -Fn, attArb, &n0, phases0, f->beta);
        num_coefs = n0 / phases0 + 1, num_coefs += num_coefs & !preM;
      }
      if((num_coefs & 1) && rational && (arbL & 1))
        phases <<= 1, arbL <<= 1, arbM *= 2;
      at = arbL * (s->phase0 = .5 * (num_coefs & 1));
      order = i + (i && mode > 4);
      coefs_size = (size_t)(num_coefs4 * phases * (order + 1)) *
          sizeof_real;
    } while(interpolator < 0 && i < 2 && f->interp[i + 1].fn &&
        coefs_size / 1000 > r_spec->coef_size_kbytes);

    if(!s->shared->poly_fir_coefs &&
        !((core_flags & CORE_DD) && s->shared->poly_dd)) {
      if(core_flags & CORE_DD) {
        soxr_dd_t * input_dd;
        int num_taps = num_coefs * phases - 1;
        input_dd = lsx_design_lpf_extreme_dd(Fp, Fs, Fn, attArb, &num_taps, phases, f->beta, 0, 0);
        if(!input_dd) return "Extreme DD filter design failed";
        s->shared->poly_dd = soxr_prepare_poly_full_dd(input_dd, num_coefs, phases, order, multiplier, bits >= 64);
        free(input_dd);
        if(!s->shared->poly_dd) return "Extreme DD coefficient allocation failed";
      } else {
      int num_taps = num_coefs * phases - 1;
      soxr_dd_t * ddcoefs = lsx_design_lpf_dd(Fp, Fs, Fn, attArb,
          &num_taps, phases, f->beta, 0, 0);
      if(ddcoefs)
        s->shared->poly_fir_coefs = prepare_poly_fir_coefs_dd(ddcoefs,
            num_coefs, phases, order, multiplier, &core->mem, bits >= 64,
            & (real *) { 0 }
            , core_flags);
      free(ddcoefs);
      if(!s->shared->poly_fir_coefs) return "out of memory";
    }
    } /* normal double coefficient route */
    multiplier = 1;
    s->fn = f1->fn;
    s->pre_post = num_coefs4 - 1;
    s->preload = ((num_coefs - 1) >> 1) + (num_coefs4 - num_coefs);
    s->n = num_coefs4;
    s->phase_bits = phase_bits;
    s->L = arbL;
    s->use_hi_prec_clock =
        mode > 1 && (q_spec->flags & SOXR_HI_PREC_CLOCK) && !rational;
#if WITH_FLOAT_STD_PREC_CLOCK
    if(order && !s->use_hi_prec_clock) {
      s->at.flt = at;
      s->step.flt = arbM;
      s->out_in_ratio = (double)(arbL / s->step.flt);
    } else
#endif
    {
      s->at.whole = (int64_t)(at * MULT32 + .5);
#if WITH_HI_PREC_CLOCK
      if(s->use_hi_prec_clock) {
        double M = arbM * MULT32;
        s->at.fix.ls.parts.ms = 0x80000000ul;
        s->step.whole = (int64_t) M;
        M -= (double) s->step.whole;
        M *= MULT32 * MULT32;
        s->step.fix.ls.all = (uint64_t) M;
      } else
#endif
        s->step.whole = (int64_t)(arbM * MULT32 + .5);
      s->out_in_ratio = MULT32 * arbL / (double) s->step.whole;
    }
    ++s;
  }

  if(have_post_stage)
    dft_stage_init(1, tighten(Fp0 / (upsample? alpha : 1)), upsample? max(2 -
        Fs0 / alpha, 1): Fs0, (double) max(postL, postM),
        att, phase_response,
        s++, postL, postM, &multiplier, r_spec->log2_min_dft_size,
        r_spec->log2_large_dft_size, core_flags, core->rdft_cb);
  if(have_post_stage && !(s-1)->fn) return "out of memory";

  lsx_debug("%g: >>%i %i/%i %i/%g %i/%i (%x)", 1 / io_ratio,
      shr, preL, preM, arbL, arbM, postL, postM, core_flags);

  for(i = 0, s = p->stages; i < p->num_stages; ++i, ++s) {
    void * preload;
    if(fifo_create(&s->fifo, (int) sizeof_real)) return "out of memory";
    preload = fifo_reserve(&s->fifo, s->preload);
    if(!preload) return "out of memory";
    memset(preload, 0, sizeof_real * (size_t) s->preload);
    lsx_debug_more("%5i|%-5i preload=%i remL=%i",
        s->pre, s->pre_post - s->pre, s->preload, s->at.integer);
  }
  if(fifo_create(&s->fifo, (int) sizeof_real)) return "out of memory";
  return 0;
}

bool soxr_stage_process(stage_t * stage, bool flushing)
{
  fifo_t * fifo = &stage->fifo;
  bool done = false;
  int want;
  if(stage->runtime_error && *stage->runtime_error) return true;
  while(!done && (want = stage->input_size - fifo_occupancy(fifo)) > 0) {
    if(stage->is_input) {
      if(flushing) {
        void *zeros = fifo_reserve(fifo, want);
        if(!zeros) { soxr_stage_oom(stage); return true; }
        memset(zeros, 0, fifo->item_size * (size_t) want);
      }
      else done = true;
    }
    else done = soxr_stage_process(stage - 1, flushing);
    if(stage->runtime_error && *stage->runtime_error) return true;
  }
  if(!stage->fn) { soxr_stage_oom(stage); return true; }
  stage->fn(stage, &stage[1].fifo);
  if(stage->runtime_error && *stage->runtime_error) return true;
  return done && fifo_occupancy(fifo) < stage->input_size;
}

STATIC void _soxr_process(rate_t * p, size_t olen)
{
  if(p->runtime_error) return;
  int const n = p->flushing? min(- (int) p->samples_out, (int) olen) :(int) olen;
  stage_t * stage = &p->stages[p->num_stages];
  fifo_t * fifo = &stage->fifo;
  bool done = false;
  while(!done && fifo_occupancy(fifo) < (int) n)
    done = stage->is_input || soxr_stage_process(stage - 1, p->flushing);
}

STATIC real * _soxr_input(rate_t * p, real const * samples, size_t n)
{
  real *result;
  if(p->flushing || p->runtime_error) return NULL;
  if(n > INT_MAX) { p->runtime_error = "input block too large"; return NULL; }
  result = fifo_write(&p->stages[0].fifo, (int)n, samples);
  if(!result) { p->runtime_error = "out of memory"; return NULL; }
  p->samples_in += (int64_t)n;
  return result;
}

STATIC real const * _soxr_output(rate_t * p, real * samples, size_t * n0)
{
  fifo_t * fifo = &p->stages[p->num_stages].fifo;
  if(p->runtime_error) { *n0 = 0; return NULL; }
  int n = p->flushing? min(- (int) p->samples_out, (int) * n0) :(int) * n0;
  p->samples_out += n = min(n, fifo_occupancy(fifo));
  return fifo_read(fifo, (int)(*n0 = (size_t) n), samples);
}

STATIC void _soxr_flush(rate_t * p)
{
  if(p->flushing) return;
  p->samples_out -= (int64_t)((double) p->samples_in / p->io_ratio + .5);
  p->samples_in = 0;
  p->flushing = true;
}

STATIC void _soxr_close(rate_t * p)
{
  if(p->stages) {
    soxr_rdft_callbacks_t const * const RDFT_CB = p->core->rdft_cb;
    rate_shared_t * shared = p->stages[0].shared;
    int i;

    for(i = 0; i <= p->num_stages; ++i) {
      stage_t * s = &p->stages[i];
      /* AVX2-only functions must never be entered on a fallback CPU. */
      free(s->extreme_dft_workspace);
      /* EXTREME plans are immutable and owned by the per-instance shared state. */
      if(s->extreme_poly0_packed_coefs)
        soxr_dd_poly0_pack_destroy(s->extreme_poly0_packed_coefs);
      if(RDFT_CB) {
        rdft_free(s->dft_scratch);
        rdft_free(s->dft_out);
      }
      fifo_delete(&s->fifo);
    }
    if(shared) {
      for(i = 0; i < 2; ++i) {
        dft_filter_t * f = &shared->dft_filter[i];
        soxr_dd_avx2_plan_destroy(shared->extreme_fft_plans[i]);
        soxr_dd_scalar_plan_destroy(shared->extreme_scalar_plans[i]);
        free(f->dd_spectrum);
        if(RDFT_CB) rdft_free(f->coefs);
        else free(f->coefs);
        if(RDFT_CB) {
          rdft_delete_setup(f->dft_forward_setup);
          rdft_delete_setup(f->dft_backward_setup);
        }
      }
      p->core->mem.free(shared->poly_fir_coefs);
      soxr_pffft_aligned_free(shared->poly_fir_simd_coefs);
      free(shared->poly_dd);
      memset(shared, 0, sizeof(*shared));
    }
    free(p->stages);
    free(p->ext_input);
    free(p->ext_output);
    (void) RDFT_CB;
  }
}

#if defined SOXR_LIB
char const * _soxr_runtime_error(rate_t *p) { return p->runtime_error; }

STATIC double _soxr_delay(rate_t * p)
{
  return (double) p->samples_in / p->io_ratio - (double) p->samples_out;
}

STATIC void _soxr_sizes(size_t * shared, size_t * channel)
{
  *shared = sizeof(rate_shared_t);
  *channel = sizeof(rate_t);
}
#endif
