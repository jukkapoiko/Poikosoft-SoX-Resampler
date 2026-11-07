/* SoX Resampler Library      Copyright (c) 2007-16 robs@users.sourceforge.net
 * Licence for this file: LGPL v2.1                  See LICENCE for details. */

/*
 * SoXR-Poikosoft | Enhanced SoX Resampler Library
 *
 * Poikosoft integration and enhancements Copyright (c) 2026
 * Jukka Poikolainen / Poikosoft | https://www.poikosoft.com
 *
 * Module: Constant-rate engine declarations
 * Features:
 *  - Stage data structures, phase clocks, and FIR coefficient indexing.
 *
 * Original third-party copyrights and license terms are preserved.
 */



#if !defined soxr_cr_included
#define soxr_cr_included

#include "soxr-dd-fft.h"

#define  FIFO_SIZE_T int
#include "fifo.h"

typedef void real; /* float or double */
struct stage;
typedef void(*stage_fn_t)(struct stage * input, fifo_t * output);
typedef struct half_fir_info {
  int num_coefs;
  real const * coefs;
  stage_fn_t fn, dfn;
  float att;
} half_fir_info_t;
typedef struct { float scalar; stage_fn_t fn; } poly_fir1_t;
typedef struct { float beta; poly_fir1_t interp[3]; } poly_fir_t;

#define U100_l 42
#define MULT32 (65536. * 65536.)

/* Original cr64s four-tap vector layout for an independent SIMD copy. */
#define coef4(coef_p, interp_order, fir_len, phase_num, coef_interp_num, fir_coef_num) (coef_p)[\
  (fir_len) * ((interp_order) + 1) * (phase_num) + \
  ((interp_order) + 1) * ((fir_coef_num) & ~3) + \
  4 * ((interp_order) - (coef_interp_num)) + \
  ((fir_coef_num) & 3)]

/* Conceptually: coef_p is &coefs[num_phases][fir_len][interp_order+1]: */
#define coef(coef_p, interp_order, fir_len, phase_num, coef_interp_num, fir_coef_num) (coef_p)[\
  (fir_len) * ((interp_order) + 1) * (phase_num) + \
  ((interp_order) + 1) * (fir_coef_num) + \
  ((interp_order) - (coef_interp_num))]

typedef union { /* Int64 in parts */
  #if HAVE_BIGENDIAN
  struct { int32_t ms; uint32_t ls; } parts;
  #else
  struct { uint32_t ls; int32_t ms; } parts;
  #endif
  int64_t all;
} int64p_t;

typedef union { /* Uint64 in parts */
  #if HAVE_BIGENDIAN
  struct { uint32_t ms, ls; } parts;
  #else
  struct { uint32_t ls, ms; } parts;
  #endif
  uint64_t all;
} uint64p_t;

typedef struct {
  int dft_length, num_taps, post_peak;
  void * dft_forward_setup, *dft_backward_setup;
  real * coefs;
  soxr_dd_t * dd_spectrum; /* length dft_length, DD complex spectrum is separate */
} dft_filter_t;

typedef struct { /* So generated filter coefs may be shared between channels */
  real * poly_fir_coefs;
  real * poly_fir_simd_coefs; /* Optional AVX or AVX2 interpolation layout. */
  soxr_dd_t * poly_dd;
  dft_filter_t dft_filter[2];
  soxr_dd_scalar_plan * extreme_scalar_plans[2];
  soxr_dd_avx2_plan * extreme_fft_plans[2]; /* Immutable, shared within this resampler only. */
} rate_shared_t;

typedef double float_step_t; /* Or long double or __float128. */

typedef union { /* Fixed point arithmetic */
  struct { uint64p_t ls; int64p_t ms; } fix; /* Hi-prec has ~96 bits. */
  float_step_t flt;
} step_t;

#define integer  fix.ms.parts.ms
#define fraction fix.ms.parts.ls
#define whole    fix.ms.all

#define CORE_DBL       1
#define CORE_SIMD_POLY 2
#define CORE_SIMD_HALF 4
#define CORE_SIMD_DFT  8
#define CORE_DD        16 /* Extreme: 2-double samples and coefficients */
#define LOG2_SIZEOF_REAL(core_flags) (((core_flags) & CORE_DD) ? 4 : 2 + ((core_flags) & 1))

typedef int core_flags_t;

#if defined SOXR_LIB
#include "rdft_t.h"
#else
#include "rdft_t.h"
#endif

typedef struct stage {
  int num;
  /* Extreme-only SIMD callbacks; initialized once per channel engine. */
  soxr_dd_fft_fn extreme_fft;
  soxr_dd_convolve_fn extreme_convolve;
  bool extreme_use_sse2;
  bool extreme_use_avx;
  soxr_dd_scalar_plan * extreme_scalar_plan; /* Owned by rate_shared_t. */
  soxr_dd_avx2_plan * extreme_fft_plan; /* Engine-local; immutable after init. */
  soxr_dd_complex * extreme_dft_workspace; /* Reused by this DFT stage. */
  void *extreme_poly0_packed_coefs; /* Stage-local AVX2 packed DD coefficient plan. */
  bool extreme_poly0_pack_attempted;
  char const ** runtime_error; /* Parent engine sticky OOM state. */

  /* Common to all stage types: */
  core_flags_t core_flags;
  stage_fn_t fn;
  fifo_t fifo;
  int pre; /* Number of past samples to store */
  int pre_post; /* pre + number of future samples to store */
  int preload; /* Number of zero samples to pre-load the fifo */
  double out_in_ratio; /* For buffer management. */
  int input_size;
  bool is_input;

  /* For a stage with variable (run-time generated) filter coefs: */
  soxr_rdft_callbacks_t const * rdft_cb;
  rate_shared_t * shared;
  unsigned dft_filter_num; /* Which, if any, of the 2 DFT filters to use */
  real * dft_scratch;
  float * dft_out;
  real const * coefs;

  /* For a stage with variable L/M: */
  step_t at, step;
  bool use_hi_prec_clock;
  int L, remM;
  int n, phase_bits, block_len;
  double mult, phase0;
} stage_t;

static __inline void soxr_stage_oom(stage_t *p)
{
  if(p->runtime_error && !*p->runtime_error)
    *p->runtime_error = "out of memory";
}

#define stage_occupancy(s) max(0, fifo_occupancy(&(s)->fifo) - (s)->pre_post)
#define stage_read_p(s) ((sample_t *)fifo_read_ptr(&(s)->fifo) + (s)->pre)

#define lq_bw0  (1385/2048.) /* ~.67625, FP exact. */

typedef enum { rolloff_small, rolloff_medium, rolloff_none } rolloff_t;

typedef struct {
  void * (*alloc)(size_t);
  void * (*calloc)(size_t, size_t);
  void(*free)(void *);
} alloc_t;

typedef struct {
  alloc_t mem;
  half_fir_info_t const * half_firs;
  size_t half_firs_len;
  half_fir_info_t const * doub_firs;
  size_t doub_firs_len;
  stage_fn_t cubic_stage_fn;
  poly_fir_t const * poly_firs;
  soxr_rdft_callbacks_t const * rdft_cb;
} cr_core_t;

typedef struct rate rate_t;
struct rate {
  cr_core_t selected_core; /* Initialized once per engine; never global */
  cr_core_t const * core;
  double io_ratio;
  int64_t samples_in, samples_out;
  int num_stages, flushing;
  stage_t * stages;
  double * ext_input, *ext_output;
  size_t ext_ic, ext_oc, ext_pending;
  char const * runtime_error;
};

#if defined SOXR_LIB

#include "soxr.h"

char const * _soxr_init(
  rate_t * const p, /* Per audio channel.                            */
  rate_shared_t * const shared, /* Between channels (undergoing same rate change)*/
  double const io_ratio, /* Input rate divided by output rate.            */
  soxr_quality_spec_t const * const q_spec,
  soxr_runtime_spec_t const * const r_spec,
  double multiplier, /* Linear gain to apply during conversion.   1   */
  cr_core_t const * const core,
  core_flags_t const);

void _soxr_process(struct rate * p, size_t olen);
bool soxr_stage_process(stage_t *, bool);
void soxr_dd_dft_stage(stage_t *, fifo_t *);
/* Optional AVX / AVX2 polynomial FIR, preserving scalar remainder taps. */
int soxr_fast64_interp_pack(stage_t *, int);
void soxr_fast64_vpoly0_sse2(stage_t *, fifo_t *);
void soxr_fast64_vpoly1_sse2(stage_t *, fifo_t *);
void soxr_fast64_vpoly2_sse2(stage_t *, fifo_t *);
void soxr_fast64_vpoly3_sse2(stage_t *, fifo_t *);
void soxr_fast64_vpoly0_avx(stage_t *, fifo_t *);
void soxr_fast64_vpoly1_avx(stage_t *, fifo_t *);
void soxr_fast64_vpoly2_avx(stage_t *, fifo_t *);
void soxr_fast64_vpoly3_avx(stage_t *, fifo_t *);
void soxr_fast64_vpoly1_avx2(stage_t *, fifo_t *);
void soxr_fast64_vpoly2_avx2(stage_t *, fifo_t *);
void soxr_fast64_vpoly3_avx2(stage_t *, fifo_t *);
/* Dedicated AVX2+FMA3 Extreme FIR kernels; never called on unsupported CPUs. */
void soxr_dd_half_avx2(stage_t *, fifo_t *);
void soxr_dd_poly0_avx2(stage_t *, fifo_t *);
void soxr_dd_poly0_avx(stage_t *, fifo_t *);
void soxr_dd_poly0_pack_destroy(void *);
void soxr_dd_poly_interp_avx2(stage_t *, fifo_t *, int);
soxr_dd_t * soxr_prepare_poly_full_dd(const soxr_dd_t *, int, int, int, double, int);
real * _soxr_input(struct rate * p, real const * samples, size_t n);
real const * _soxr_output(struct rate * p, real * samples, size_t * n0);
void _soxr_flush(struct rate * p);
void _soxr_close(struct rate * p);
double _soxr_delay(struct rate * p);
char const * _soxr_runtime_error(struct rate * p);
void _soxr_sizes(size_t * shared, size_t * channel);
#endif

#endif
