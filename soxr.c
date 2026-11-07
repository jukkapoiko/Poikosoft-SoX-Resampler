/* SoX Resampler Library      Copyright (c) 2007-18 robs@users.sourceforge.net
 * Licence for this file: LGPL v2.1                  See LICENCE for details. */

/*
 * SoXR-Poikosoft | Enhanced SoX Resampler Library
 *
 * Poikosoft integration and enhancements Copyright (c) 2026
 * Jukka Poikolainen / Poikosoft | https://www.poikosoft.com
 *
 * Module: Public SoXR API and resampler lifecycle
 * Features:
 *  - Quality-profile and Fast/Extreme engine selection.
 *  - High-precision sample clock support and runtime initialization.
 *
 * Original third-party copyrights and license terms are preserved.
 */



#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "soxr.h"
#include "data-io.h"
#include "internal.h"

#if defined(_MSC_VER) && !defined(__clang__)
  #define SOXR_RATE_ISFINITE(x) _finite(x)
#else
  #define SOXR_RATE_ISFINITE(x) isfinite(x)
#endif




#if WITH_DEV_TRACE

#include <stdarg.h>
#include <stdio.h>

/* No process-global trace level or mutable tracing configuration. */

void _soxr_trace(char const * fmt, ...)
{
  va_list args;
  va_start(args, fmt);
  vfprintf(stderr, fmt, args);
  fputc('\n', stderr);
  va_end(args);
}

#endif



char const * soxr_version(void)
{
  return "libsoxr-" SOXR_THIS_VERSION_STR;
}




typedef void sample_t; /* float or double */
#include "rate-callbacks.h"
typedef soxr_rate_callbacks_t control_block_t;

#define resampler_input        p->control_block.input
#define resampler_process      p->control_block.process
#define resampler_output       p->control_block.output
#define resampler_flush        p->control_block.flush
#define resampler_close        p->control_block.close
#define resampler_delay        p->control_block.delay
#define resampler_sizes        p->control_block.sizes
#define resampler_create       p->control_block.create
#define resampler_set_io_ratio p->control_block.set_io_ratio
#define resampler_id           p->control_block.id
#define resampler_status       p->control_block.status

typedef void * resampler_t; /* For one channel. */
typedef void * resampler_shared_t; /* Between channels. */
typedef void(*deinterleave_t)(sample_t * *dest,
    soxr_datatype_t data_type, void const * *src0, size_t n, unsigned ch);
typedef size_t(*interleave_t)(soxr_datatype_t data_type, void * *dest,
    sample_t const * const * src, size_t, unsigned, unsigned long *);

/* Format-specific adapters keep indirect I/O calls ABI-compatible. */
static void typed_deinterleave_f(sample_t ** dest, soxr_datatype_t type,
    void const ** src, size_t n, unsigned channels)
{
  _soxr_deinterleave_f((float **) dest, type, src, n, channels);
}
static void typed_deinterleave_d(sample_t ** dest, soxr_datatype_t type,
    void const ** src, size_t n, unsigned channels)
{
  _soxr_deinterleave((double **) dest, type, src, n, channels);
}
static size_t typed_interleave_f(soxr_datatype_t type, void ** dest,
    sample_t const * const * src, size_t n, unsigned channels,
    unsigned long * seed)
{
  return _soxr_interleave_f(type, dest,
      (float const * const *) src, n, channels, seed);
}
static size_t typed_interleave_d(soxr_datatype_t type, void ** dest,
    sample_t const * const * src, size_t n, unsigned channels,
    unsigned long * seed)
{
  return _soxr_interleave(type, dest,
      (double const * const *) src, n, channels, seed);
}

struct soxr {
  unsigned num_channels;
  double io_ratio;
  soxr_error_t error;
  soxr_quality_spec_t q_spec;
  soxr_io_spec_t io_spec;
  soxr_runtime_spec_t runtime_spec;

  void * input_fn_state;
  soxr_input_fn_t input_fn;
  size_t max_ilen;

  resampler_shared_t shared;
  resampler_t * resamplers;
  control_block_t control_block;
  deinterleave_t deinterleave;
  interleave_t interleave;

  void * *channel_ptrs;
  size_t clips;
  unsigned long seed;
  int flushing;
#if SOXR_AUTO_LPC_EXTRAPOLATION
  struct soxr_lpc_state *lpc;
  double lpc_input_rate, lpc_output_rate;
#endif
};

#if SOXR_AUTO_LPC_EXTRAPOLATION
struct soxr_lpc_state;
static void soxr_lpc_free(struct soxr_lpc_state *);
static soxr_error_t soxr_lpc_new(soxr_t,double,double);
#endif

#if WITH_CR32 || WITH_CR64
  #include "filter.h"
#else
  #define lsx_to_3dB(x) ((x)/(x))
#endif

soxr_quality_spec_t soxr_quality_spec(unsigned long recipe, unsigned long flags)
{
    soxr_quality_spec_t spec, *p = &spec;
    unsigned q = recipe & 0xf; /* TODO: move to soxr-lsr.c: */
    unsigned long const extreme_recipe = recipe & 0x300000UL;
    double rej;

    memset(p, 0, sizeof(*p));
    /* These explicit EXTREME quality recipes use the original Kaiser-sinc
     * filter, DD coefficient design and complete DD signal path. */
    if(extreme_recipe && extreme_recipe != 0x100000UL)
      p->e = "unsupported SOXR_EXTREME quality recipe";
    else if(extreme_recipe == 0x100000UL) {
      if(q != SOXR_64_BITQ)
        p->e = "invalid SOXR_EXTREME quality recipe";
      else
        flags |= SOXR_EXTREME;
    }

    flags |= q < SOXR_LSR0Q || q == SOXR_64_BITQ ? RESET_ON_CLEAR : 0;
    p->phase_response = "\62\31\144"[(recipe & 0x30) >> 4];
    p->stopband_begin = 1.0;
    
    switch(q) {

        case SOXR_16_BITQ:
            p->precision = 16;
            break;
        case SOXR_20_BITQ:
            p->precision = 20;
            break;
        case SOXR_24_BITQ:
            p->precision = 24;
            break;
        case SOXR_28_BITQ:
            p->precision = 28;
            break;
        case SOXR_32_BITQ:
            p->precision = 32;
            break;
        case SOXR_40_BITQ:
            p->precision = 40;
            break;        
        case SOXR_56_BITQ:
            p->precision = 56;
            break;
        case SOXR_64_BITQ:
            p->precision = 64;
            break;
        default:
            p->precision = 32;
            break;
    }

    /* The attenuation-profile goal is independent of DD arithmetic. */
    rej = p->precision * linear_to_dB(2.);
    
    /* Fast: use high-precision clock for non-rational resampling ratios.
     * cr.c only enables the additional clock arithmetic for interpolated
     * FIR stages whose phase cannot be represented exactly as rational. */
    p->flags = flags | SOXR_HI_PREC_CLOCK;
    if(flags & SOXR_EXTREME)
      p->flags |= SOXR_DOUBLE_PRECISION | SOXR_HI_PREC_CLOCK;
    if(q == SOXR_64_BITQ)
        p->flags |= SOXR_DOUBLE_PRECISION | SOXR_HI_PREC_CLOCK;

    if(q <= SOXR_56_BITQ || q == SOXR_64_BITQ) {
        
        #define LOW_Q_BW0     (1385 / 2048.) /* 0.67625 rounded to be a FP exact. */
        p->passband_end = q == 1 ? LOW_Q_BW0 : 1 - .05 / lsx_to_3dB(rej);
        if(q <= 2)
            p->flags &= ~SOXR_ROLLOFF_NONE, p->flags |= SOXR_ROLLOFF_MEDIUM;
    }
    else if(q >= SOXR_LSR0Q && q <= SOXR_LSR2Q) { /* TODO: move to soxr-lsr.c */
        static float const bw[] = { .931f, .832f, .663f };
        p->passband_end = bw[q - SOXR_LSR0Q];
        if(q == SOXR_LSR2Q) {
            p->flags &= ~SOXR_ROLLOFF_NONE;
            p->flags |= SOXR_ROLLOFF_LSR2Q | SOXR_PROMOTE_TO_LQ;
        }
    }
    else {
        p->e = "invalid quality recipe";
    }
    /* At most one explicit steepness recipe selection. */

    {
      unsigned long steep = recipe & (SOXR_STEEP_99 | 
                                    SOXR_STEEP_995 | SOXR_STEEP_9955 | 
                                    SOXR_STEEP_996 | SOXR_STEEP_9965 | 
                                    SOXR_STEEP_997 | SOXR_STEEP_9975 | 
                                    SOXR_STEEP_998);
      if(steep && (steep & (steep - 1)))
        p->e = "conflicting SOXR_STEEP_* flags";
      else if(steep == SOXR_STEEP_99)
        p->passband_end = 1 - .01 / lsx_to_3dB(rej);
      else if(steep == SOXR_STEEP_995)
        p->passband_end = 1 - .005 / lsx_to_3dB(rej);
      else if(steep == SOXR_STEEP_9955)
        p->passband_end = 1 - .0045 / lsx_to_3dB(rej);
      else if(steep == SOXR_STEEP_996)
        p->passband_end = 1 - .004 / lsx_to_3dB(rej);
      else if(steep == SOXR_STEEP_9965)
        p->passband_end = 1 - .0035 / lsx_to_3dB(rej);
      else if(steep == SOXR_STEEP_997)
        p->passband_end = 1 - .003 / lsx_to_3dB(rej);
      else if(steep == SOXR_STEEP_9975)
        p->passband_end = 1 - .0025 / lsx_to_3dB(rej);
      else if(steep == SOXR_STEEP_998)
        p->passband_end = 1 - .002 / lsx_to_3dB(rej);
    }
    if(recipe & SOXR_ALLOW_ALIASING) {
        p->stopband_begin = 1 - (1 - p->passband_end) * lsx_to_3dB(rej);
        p->stopband_begin = 2 - p->stopband_begin / 1;
    }
    return spec;
}




char const * soxr_engine(soxr_t p)
{
  return resampler_id();
}



size_t * soxr_num_clips(soxr_t p)
{
  return &p->clips;
}



soxr_error_t soxr_error(soxr_t p)
{
  return p->error;
}



soxr_runtime_spec_t soxr_runtime_spec(unsigned num_threads)
{
  soxr_runtime_spec_t spec, *p = &spec;
  memset(p, 0, sizeof(*p));
  p->log2_min_dft_size = 10;
  p->log2_large_dft_size = 17;
  p->coef_size_kbytes = 400;
  p->num_threads = 1;
  (void) num_threads;
  return spec;
}



soxr_io_spec_t soxr_io_spec(
  soxr_datatype_t itype,
  soxr_datatype_t otype)
{
  soxr_io_spec_t spec, *p = &spec;
  memset(p, 0, sizeof(*p));
  if((itype | otype) >= SOXR_SPLIT * 2)
    p->e = "invalid io datatype(s)";
  else {
    p->itype = itype;
    p->otype = otype;
    p->scale = 1;
  }
  return spec;
}

extern control_block_t const
  _soxr_rate32_cb,
  _soxr_rate64_cb,
  _soxr_rate64e_cb;


static void runtime_num(char const * env_name,
    int min, int max, unsigned * field)
{
    /*
  char const * e = getenv(env_name);
  if (e) {
    int i = atoi(e);
    if (i >= min && i <= max)
      *field = (unsigned)i;
  }
  */
}




soxr_t soxr_create(
  double input_rate, double output_rate,
  unsigned num_channels,
  soxr_error_t * error0,
  soxr_io_spec_t const * io_spec,
  soxr_quality_spec_t const * q_spec,
  soxr_runtime_spec_t const * runtime_spec)
{
  double io_ratio = output_rate != 0? input_rate != 0?
    input_rate / output_rate: -1: input_rate != 0? -1 : 0;
  static const float datatype_full_scale[] = { 1, 1, 65536. * 32768, 32768 };
  soxr_t p = 0;
  soxr_error_t error = 0;

  if(!SOXR_RATE_ISFINITE(input_rate) || !SOXR_RATE_ISFINITE(output_rate))
    error = "sample rates must be finite";
  else if(input_rate < 0 || output_rate < 0)
    error = "sample rates must not be negative";
  else if(input_rate > 0 && output_rate > 0 &&
          (!(io_ratio > 0) || !SOXR_RATE_ISFINITE(io_ratio)))
    error = "invalid sample-rate ratio";

#if WITH_DEV_TRACE
#define _(x) (char)(sizeof(x)>=10? 'a'+(char)(sizeof(x)-10):'0'+(char)sizeof(x))
  char const * e = getenv("SOXR_TRACE");
  (void) e;
  {
    static char const arch[] = { _(char), _(short), _(int), _(long), _(long long)
      , ' ', _(float), _(double), _(long double)
      , ' ', _(int *), _(int(*)(int))
      , ' ', HAVE_BIGENDIAN ? 'B' : 'L'

      , 0 };
#undef _
    lsx_debug("arch: %s", arch);
  }
#endif

  if(!error && q_spec && q_spec->e) error = q_spec->e;
  else if(!error && io_spec && (io_spec->itype | io_spec->otype) >= SOXR_SPLIT * 2)
    error = "invalid io datatype(s)";

  if(!error && !(p = calloc(sizeof(*p), 1))) error = "malloc failed";

  if(p) {
    control_block_t const * control_block;

    p->q_spec = q_spec? *q_spec : soxr_quality_spec(SOXR_HQ, 0);

    if(q_spec) { /* Backwards compatibility with original API: */
      if(p->q_spec.passband_end > 2)
        p->q_spec.passband_end /= 100;
      if(p->q_spec.stopband_begin > 2)
        p->q_spec.stopband_begin = 2 - p->q_spec.stopband_begin / 100;
    }

    p->io_ratio = io_ratio;
#if SOXR_AUTO_LPC_EXTRAPOLATION
    p->lpc_input_rate=input_rate;
    p->lpc_output_rate=output_rate;
#endif
    p->num_channels = num_channels;
    if(io_spec)
      p->io_spec = *io_spec;
    else
      p->io_spec.scale = 1;

    p->runtime_spec = runtime_spec? *runtime_spec : soxr_runtime_spec(1);

    runtime_num("SOXR_MIN_DFT_SIZE", 8, 15, &p->runtime_spec.log2_min_dft_size);
    runtime_num("SOXR_LARGE_DFT_SIZE", 8, 20, &p->runtime_spec.log2_large_dft_size);
    runtime_num("SOXR_COEFS_SIZE", 100, 800, &p->runtime_spec.coef_size_kbytes);
    p->runtime_spec.num_threads = 1;

    p->io_spec.scale *= datatype_full_scale[p->io_spec.otype & 3] /
                        datatype_full_scale[p->io_spec.itype & 3];

    p->seed = (unsigned long) time(0) ^ (unsigned long)(size_t) p;

    /* Automatic Fast32 for <=20-bit filter targets; otherwise Fast64.
     * The Extreme recipe always selects the full DD implementation.
     * Input/output sample format is independent of the internal engine. */
    if(p->q_spec.precision <= 20 &&
        !(p->q_spec.flags & (SOXR_DOUBLE_PRECISION | SOXR_EXTREME))) {
      p->deinterleave = typed_deinterleave_f;
      p->interleave = typed_interleave_f;
      control_block = &_soxr_rate32_cb;
    } else {
      p->deinterleave = typed_deinterleave_d;
      p->interleave = typed_interleave_d;
      control_block = (p->q_spec.flags & SOXR_EXTREME)
          ? &_soxr_rate64e_cb : &_soxr_rate64_cb;
    }
    memcpy(&p->control_block, control_block, sizeof(p->control_block));

    if(p->num_channels && io_ratio != 0)
      error = soxr_set_io_ratio(p, io_ratio, 0);
#if SOXR_AUTO_LPC_EXTRAPOLATION
    if(!error && p->num_channels && io_ratio > 0 &&
        !(p->io_spec.itype & SOXR_SPLIT) &&
        !(p->io_spec.otype & SOXR_SPLIT) &&
        (p->io_spec.itype == SOXR_FLOAT32_I || p->io_spec.itype == SOXR_FLOAT64_I) &&
        (p->io_spec.otype == SOXR_FLOAT32_I || p->io_spec.otype == SOXR_FLOAT64_I) &&
        
#if defined(_MSC_VER) && !defined(__clang__)
        _finite(input_rate) && _finite(output_rate) &&
#else
        isfinite(input_rate) && isfinite(output_rate) &&
#endif
        input_rate>0 && output_rate>0)
      error=soxr_lpc_new(p,input_rate,output_rate);
#endif
  }
  if(error)
    soxr_delete(p), p = 0;
  if(error0)
    *error0 = error;
  return p;
}



soxr_error_t soxr_set_input_fn(soxr_t p,
    soxr_input_fn_t input_fn, void * input_fn_state, size_t max_ilen)
{
#if SOXR_AUTO_LPC_EXTRAPOLATION
  if(p->lpc && input_fn) return "LPC extrapolation does not support input callbacks";
#endif
  p->input_fn_state = input_fn_state;
  p->input_fn = input_fn;
  p->max_ilen = max_ilen? max_ilen :(size_t) - 1;
  return 0;
}



static void soxr_delete0(soxr_t p)
{
  unsigned i;

  if(p->resamplers) for(i = 0; i < p->num_channels; ++i) {
    if(p->resamplers[i])
      resampler_close(p->resamplers[i]);
    free(p->resamplers[i]);
  }
  free(p->resamplers);
  free(p->channel_ptrs);
  free(p->shared);
#if SOXR_AUTO_LPC_EXTRAPOLATION
  soxr_lpc_free(p->lpc);
#endif

  memset(p, 0, sizeof(*p));
}



double soxr_delay(soxr_t p)
{
  return
    (p && !p->error && p->resamplers)? resampler_delay(p->resamplers[0]) : 0;
}



static soxr_error_t fatal_error(soxr_t p, soxr_error_t error)
{
  soxr_delete0(p);
  return p->error = error;
}



static soxr_error_t initialise(soxr_t p)
{
  unsigned i;
  size_t shared_size, channel_size;

  resampler_sizes(&shared_size, &channel_size);
  p->channel_ptrs = calloc(sizeof(*p->channel_ptrs), p->num_channels);
  p->shared = calloc(shared_size, 1);
  p->resamplers = calloc(sizeof(*p->resamplers), p->num_channels);
  if(!p->shared || !p->channel_ptrs || !p->resamplers)
    return fatal_error(p, "malloc failed");

  for(i = 0; i < p->num_channels; ++i) {
    soxr_error_t error;
    if(!(p->resamplers[i] = calloc(channel_size, 1)))
      return fatal_error(p, "malloc failed");
    error = resampler_create(
        p->resamplers[i],
        p->shared,
        p->io_ratio,
        &p->q_spec,
        &p->runtime_spec,
        p->io_spec.scale);
    if(error)
      return fatal_error(p, error);
  }
  return 0;
}



soxr_error_t soxr_set_num_channels(soxr_t p, unsigned num_channels)
{
  if(!p) return "invalid soxr_t pointer";
  if(num_channels == p->num_channels) return p->error;
  if(!num_channels) return "invalid # of channels";
  if(p->resamplers) return "# of channels can't be changed";
  p->num_channels = num_channels;
  return soxr_set_io_ratio(p, p->io_ratio, 0);
}



soxr_error_t soxr_set_io_ratio(soxr_t p, double io_ratio, size_t slew_len)
{
  unsigned i;
  soxr_error_t error;
  if(!p) return "invalid soxr_t pointer";
  if((error = p->error)) return error;
  if(!p->num_channels) return "must set # channels before O/I ratio";
  if(!(io_ratio > 0) || !SOXR_RATE_ISFINITE(io_ratio)) return "I/O ratio out-of-range";
  if(!p->channel_ptrs) {
    p->io_ratio = io_ratio;
    return initialise(p);
  }
#if SOXR_AUTO_LPC_EXTRAPOLATION
  if(p->lpc && fabs(p->io_ratio-io_ratio) >= 1e-15)
    return "LPC extrapolation requires fixed input/output rates";
#endif
  if(p->control_block.set_io_ratio) {
    for(i = 0; !error && i < p->num_channels; ++i)
      resampler_set_io_ratio(p->resamplers[i], io_ratio, slew_len);
    return error;
  }
  return fabs(p->io_ratio - io_ratio) < 1e-15? 0 :
    "varying O/I ratio is not supported with this quality level";
}



void soxr_delete(soxr_t p)
{
  if(p)
    soxr_delete0(p), free(p);
}



soxr_error_t soxr_clear(soxr_t p) /* TODO: this, properly. */
{
  if(p) {
    struct soxr tmp = *p;
    soxr_delete0(p);
    memset(p, 0, sizeof(*p));
    p->input_fn = tmp.input_fn;
#if SOXR_AUTO_LPC_EXTRAPOLATION
    p->lpc_input_rate=tmp.lpc_input_rate;
    p->lpc_output_rate=tmp.lpc_output_rate;
#endif
    p->runtime_spec = tmp.runtime_spec;
    p->q_spec = tmp.q_spec;
    p->io_spec = tmp.io_spec;
    p->num_channels = tmp.num_channels;
    p->input_fn_state = tmp.input_fn_state;
    p->control_block = tmp.control_block;
    p->deinterleave = tmp.deinterleave;
    p->interleave = tmp.interleave;
    {
      soxr_error_t err=(p->q_spec.flags & RESET_ON_CLEAR)?
        soxr_set_io_ratio(p, tmp.io_ratio, 0): 0;
#if SOXR_AUTO_LPC_EXTRAPOLATION
      if(!err && tmp.lpc) err=soxr_lpc_new(p,tmp.lpc_input_rate,tmp.lpc_output_rate);
#endif
      if(err) p->error=err;
      return err;
    }
  }
  return "invalid soxr_t pointer";
}



/* Collect per-channel errors on the calling thread. Different resampler
 * instances can be used simultaneously by independent application threads. */
static bool soxr_check_runtime_errors(soxr_t p)
{
  unsigned i;
  if(p->error) return true;
  for(i = 0; i < p->num_channels; ++i)
    if(p->resamplers && p->resamplers[i]) {
      char const * e = resampler_status(p->resamplers[i]);
      if(e) { p->error = e; return true; }
    }
  return false;
}

static void soxr_input_1ch(soxr_t p, unsigned i, soxr_cbuf_t src, size_t len)
{
  sample_t * dest = resampler_input(p->resamplers[i], NULL, len);
  if(!dest) return; /* Stage status is collected before output begins. */
  (*p->deinterleave)(&dest, p->io_spec.itype, &src, len, 1);
}



static size_t soxr_input(soxr_t p, void const * in, size_t len)
{
  bool separated = !!(p->io_spec.itype & SOXR_SPLIT);
  unsigned i;
  if(!p || p->error) return 0;
  if(!in && len) { p->error = "null input buffer pointer"; return 0; }
  if(!len) {
    p->flushing = true;
    return 0;
  }
  if(separated)
    for(i = 0; i < p->num_channels; ++i)
      soxr_input_1ch(p, i, ((soxr_cbufs_t) in)[i], len);
  else {
    for(i = 0; i < p->num_channels; ++i)
      p->channel_ptrs[i] = resampler_input(p->resamplers[i], NULL, len);
    if(soxr_check_runtime_errors(p)) return 0;
    (*p->deinterleave)(
        (sample_t **) p->channel_ptrs, p->io_spec.itype, &in, len, p->num_channels);
  }
  if(soxr_check_runtime_errors(p)) return 0;
  return len;
}



static size_t soxr_output_1ch(soxr_t p, unsigned i, soxr_buf_t dest, size_t len, bool separated)
{
  sample_t const * src;
  if(p->flushing)
    resampler_flush(p->resamplers[i]);
  resampler_process(p->resamplers[i], len);
  if(resampler_status(p->resamplers[i])) return 0;
  src = resampler_output(p->resamplers[i], NULL, &len);
  if(resampler_status(p->resamplers[i])) return 0;
  if(!src && len) return 0;
  if(separated)
    p->clips += (p->interleave)(p->io_spec.otype, &dest, &src,
      len, 1, (p->io_spec.flags & SOXR_NO_DITHER)? 0 : &p->seed);
  else p->channel_ptrs[i] = (void /* const */ *)src;
  return len;
}



static size_t soxr_output_no_callback(soxr_t p, soxr_buf_t out, size_t len)
{
  unsigned u;
  size_t done = 0;
  bool separated = !!(p->io_spec.otype & SOXR_SPLIT);
  for(u = 0; u < p->num_channels; ++u)
    done = soxr_output_1ch(p, u, separated ? ((soxr_bufs_t) out)[u] : out, len, separated);

  if(soxr_check_runtime_errors(p)) return 0;
  if(!separated)
    p->clips += (p->interleave)(p->io_spec.otype, &out, (sample_t const * const *) p->channel_ptrs,
        done, p->num_channels, (p->io_spec.flags & SOXR_NO_DITHER)? 0 : &p->seed);
  return done;
}



size_t soxr_output(soxr_t p, void * out, size_t len0)
{
  size_t odone, odone0 = 0, olen = len0, osize, idone;
  size_t ilen;
  void const * in = out; /* Set to !=0, so that caller may leave unset. */
  bool was_flushing;
  void * const split_output = out;
  bool const separated = p && !!(p->io_spec.otype & SOXR_SPLIT);

  if(!p || p->error) return 0;
  ilen = min(p->max_ilen, (size_t) ceil((double) olen * p->io_ratio));
  if(!out && len0) { p->error = "null output buffer pointer"; return 0; }

  do {
    if(separated) {
      unsigned j;
      for(j = 0; j < p->num_channels; ++j)
        p->channel_ptrs[j] = (char *)((soxr_bufs_t)split_output)[j] +
            odone0 * soxr_datatype_size(p->io_spec.otype);
      odone = soxr_output_no_callback(p, p->channel_ptrs, olen);
    } else
      odone = soxr_output_no_callback(p, out, olen);
    if(soxr_check_runtime_errors(p)) return 0;
    odone0 += odone;
    if(odone0 == len0 || !p->input_fn || p->flushing)
      break;

    osize = soxr_datatype_size(p->io_spec.otype) * p->num_channels;
    if(!separated) out = (char *) out + osize * odone;
    olen -= odone;
    idone = p->input_fn(p->input_fn_state, &in, ilen);
    was_flushing = p->flushing;
    if(!in)
      p->error = "input function reported failure";
    else soxr_input(p, in, idone);
  } while(odone || idone || (!was_flushing && p->flushing));
  return odone0;
}



static size_t soxr_i_for_o(soxr_t p, size_t olen, size_t ilen)
{
  size_t result;
    result = (size_t) ceil((double) olen * p->io_ratio);
  return min(result, ilen);
}






static soxr_error_t soxr_process_raw(soxr_t p,
    void const * in, size_t ilen0, size_t * idone0,
    void * out, size_t olen, size_t * odone0)
{
  size_t ilen, idone, odone = 0;
  unsigned u;
  bool flush_requested = false;

  if(!p) return "null pointer";
  if(p->error) return p->error;

  if(!in)
    flush_requested = true, ilen = ilen0 = 0;
  else {
    if((ptrdiff_t) ilen0 < 0)
      flush_requested = true, ilen0 = ~ilen0;
    if(idone0 && (1 || flush_requested))
      ilen = soxr_i_for_o(p, olen, ilen0);
    else
      ilen = ilen0/*, olen = soxr_o_for_i(p, ilen, olen)*/;
  }
  p->flushing |= ilen == ilen0 && flush_requested;

  if(!out && !in)
    idone = ilen;
  else if(p->io_spec.itype & p->io_spec.otype & SOXR_SPLIT) { /* Both i & o */
    for(u = 0; u < p->num_channels; ++u) {
      if(in)
        soxr_input_1ch(p, u, ((soxr_cbufs_t) in)[u], ilen);
      if(soxr_check_runtime_errors(p)) break;
      odone = soxr_output_1ch(p, u, ((soxr_bufs_t) out)[u], olen, true);
      if(soxr_check_runtime_errors(p)) break;
    }
    idone = ilen;
  }
  else {
    idone = ilen? soxr_input(p, in, ilen) : 0;
    odone = soxr_output(p, out, olen);
  }
  
  if(soxr_check_runtime_errors(p)) {
    /* A channel may have consumed input before another channel failed.
     * Never expose an incomplete multi-channel output block as valid. */
    idone = 0;
    odone = 0;
  }
  if(idone0) *idone0 = idone;
  if(odone0) *odone0 = odone;
  return p->error;
}

#if SOXR_AUTO_LPC_EXTRAPOLATION
#include "soxr-lpc-extrap.h"
#endif

soxr_error_t soxr_process(soxr_t p,
  void const *in, size_t ilen, size_t *idone,
  void *out, size_t olen, size_t *odone)
{
#if SOXR_AUTO_LPC_EXTRAPOLATION
  if(p && p->lpc) {
    if(p->error) return p->error;
    {
      soxr_error_t err=soxr_lpc_process(p,in,ilen,idone,out,olen,odone);
      if(err && !p->error) p->error=err;
      return err;
    }
  }
#endif
  return soxr_process_raw(p,in,ilen,idone,out,olen,odone);
}



soxr_error_t soxr_oneshot(
    double irate, double orate,
    unsigned num_channels,
    void const * in, size_t ilen, size_t * idone,
    void * out, size_t olen, size_t * odone,
    soxr_io_spec_t const * io_spec,
    soxr_quality_spec_t const * q_spec,
    soxr_runtime_spec_t const * runtime_spec)
{
  soxr_t resampler;
  soxr_error_t error = q_spec? q_spec->e : 0;
  if(!error) {
    soxr_quality_spec_t q_spec1;
    if(!q_spec)
      q_spec1 = soxr_quality_spec(SOXR_LQ, 0), q_spec = &q_spec1;
    resampler = soxr_create(irate, orate, num_channels,
        &error, io_spec, q_spec, runtime_spec);
  }
  if(!error) {
    {
      size_t idone_local = 0;
      error = soxr_process(resampler, in, ~ilen,
          idone ? idone : &idone_local, out, olen, odone);
    }
    soxr_delete(resampler);
  }
  return error;
}



soxr_error_t soxr_set_error(soxr_t p, soxr_error_t error)
{
  if(!p) return "null pointer";
  if(!p->error && p->error != error) return p->error;
  p->error = error;
  return 0;
}
