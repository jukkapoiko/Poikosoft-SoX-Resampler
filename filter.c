/* SoX Resampler Library      Copyright (c) 2007-16 robs@users.sourceforge.net
 * Licence for this file: LGPL v2.1                  See LICENCE for details. */

/*
 * SoXR-Poikosoft | Enhanced SoX Resampler Library
 *
 * Poikosoft integration and enhancements Copyright (c) 2026
 * Jukka Poikolainen / Poikosoft | https://www.poikosoft.com
 *
 * Module: Precision-oriented FIR filter design
 * Features:
 *  - Kaiser-window and low-pass filter design.
 *  - Double-double precision coefficient-generation helpers.
 *
 * Original third-party copyrights and license terms are preserved.
 */



#include "filter.h"
#include "soxr-config.h"

#include "math-wrap.h"
#include <assert.h>
#include <string.h>
#include <stdlib.h>

#include "fft4g.h"
#include "internal.h"


static soxr_dd_t dd_pi(void)
{ return soxr_dd_make(3.1415926535897931160, 1.2246467991473532072e-16); }
static soxr_dd_t dd_sin(soxr_dd_t x)
{
  /* Reduce to +/-pi before the alternating Taylor series. */
  soxr_dd_t pi = dd_pi(), twopi = soxr_dd_mul_d(pi, 2);
  double n = nearbyint(x.hi / (2 * 3.14159265358979323846));
  soxr_dd_t r = soxr_dd_sub(x, soxr_dd_mul_d(twopi, n));
  soxr_dd_t xx = soxr_dd_mul(r, r), term = r, sum = r;
  int k;
  for(k = 1; k < 45; ++k) {
    term = soxr_dd_mul(term, soxr_dd_div(soxr_dd_neg(xx),
               soxr_dd_from((double)(2 * k) * (2 * k + 1))));
    sum = soxr_dd_add(sum, term);
    if(fabs(term.hi) < 1e-35) break;
  }
  return sum;
}
static soxr_dd_t dd_sinpi(soxr_dd_t a)
{ return dd_sin(soxr_dd_mul(dd_pi(), a)); }
static soxr_dd_t dd_sinh_over_x(soxr_dd_t d)
{
  /* Entire function F(d) = sinh(sqrt(d))/sqrt(d) analytically
     continued for d<0. Avoid huge cancellation in the sine region. */
  soxr_dd_t x, term, sum;
  int k;
  if(d.hi < -0.1) {
    x = soxr_dd_sqrt(soxr_dd_neg(d));
    return soxr_dd_div(dd_sin(x), x);
  }
  term = sum = soxr_dd_from(1);
  for(k = 1; k < 110; ++k) {
    term = soxr_dd_mul(term, soxr_dd_div(d,
             soxr_dd_from((double)(2 * k) * (2 * k + 1))));
    sum = soxr_dd_add(sum, term);
    if(fabs(term.hi) < 1e-34) break;
  }
  return sum;
}
static soxr_dd_t dd_i0(soxr_dd_t x)
{
  soxr_dd_t x2 = soxr_dd_mul_d(soxr_dd_mul(x, x), 0.25);
  soxr_dd_t sum = soxr_dd_from(1), term = sum;
  int k;
  for(k = 1; k < 150; ++k) {
    term = soxr_dd_mul(term, soxr_dd_div(x2, soxr_dd_from((double) k * k)));
    sum = soxr_dd_add(sum, term);
    if(fabs(term.hi) < fabs(sum.hi) * 2e-34) break;
  }
  return sum;
}
static soxr_dd_t * dd_spectral_kaiser(int taps, double fp, double fs, double scale)
{
  const int m = taps - 1;
  const soxr_dd_t beta = { 30, 0 };
  soxr_dd_t mid = soxr_dd_mul_d(soxr_dd_add(soxr_dd_from(fp), soxr_dd_from(fs)), 0.5);
  soxr_dd_t halfspan = soxr_dd_mul_d(soxr_dd_sub(soxr_dd_from(fs), soxr_dd_from(fp)), 0.5);
  soxr_dd_t normalizer = soxr_dd_div(soxr_dd_from(30),
      soxr_dd_mul(beta, dd_sinh_over_x(soxr_dd_mul(beta, beta))));
  soxr_dd_t * h;
  int i;
  if(taps < 31 || taps > 65535 || !(fs > fp) || fs >= 1 || fp <= 0) return NULL;
  h = malloc((size_t) taps * sizeof(*h)); if(!h) return NULL;
  for(i = 0; i <= m / 2; ++i) {
    soxr_dd_t z = soxr_dd_from((double) i - 0.5 * m);
    soxr_dd_t omega = soxr_dd_mul(soxr_dd_mul(dd_pi(), z), halfspan);
    soxr_dd_t d = soxr_dd_sub(soxr_dd_mul(beta, beta), soxr_dd_mul(omega, omega));
    soxr_dd_t sh = dd_sinh_over_x(d);
    soxr_dd_t v = (z.hi == 0)?mid :
      soxr_dd_div(dd_sinpi(soxr_dd_mul(mid, z)), soxr_dd_mul(dd_pi(), z));
    h[i] = h[m - i] = soxr_dd_mul_d(soxr_dd_mul(soxr_dd_mul(v, sh), normalizer), scale);
  }
  return h;
}
soxr_dd_t * lsx_design_lpf_dd(double Fp, double Fs, double Fn, double att,
    int * num_taps, int k, double beta, int accurate_sinc, int spectral_taper)
{
  int n =*num_taps, phases = k > 0?k : 1, modulo = k < 0?-k : 1, i;
  double fp = Fp / fabs(Fn), fs = Fs / fabs(Fn);
  double tr_bw = 0.5 * (fs - fp) / phases;
  double fc = fs / phases - tr_bw, rho = phases == 1?.5 : att < 120?.63 : .75;
  soxr_dd_t * h = NULL;
  (void) accurate_sinc;
  if(tr_bw > 0.5 * fs / phases) tr_bw = 0.5 * fs / phases;
  lsx_kaiser_params(att, fc, tr_bw, &beta, num_taps);
  if(!n) * num_taps = phases > 1?
    *num_taps / phases * phases + phases - 1:
    (*num_taps + modulo - 2) / modulo * modulo + 1;
  if(Fn < 0) return NULL;
  if(spectral_taper && k < 0 && phases == 1)
    h = dd_spectral_kaiser(*num_taps, fp, fs, phases);
  if(!h) {
    const int m =*num_taps - 1;
    soxr_dd_t denom = dd_i0(soxr_dd_from(beta));
    soxr_dd_t gain = soxr_dd_div(soxr_dd_from((double) phases), denom);
    soxr_dd_t inv = soxr_dd_div(soxr_dd_from(1),
        soxr_dd_add(soxr_dd_from(0.5 * m), soxr_dd_from(rho)));
    h = malloc((size_t)(*num_taps) * sizeof(*h)); if(!h) return NULL;
    for(i = 0; i <= m / 2; ++i) {
      soxr_dd_t z = soxr_dd_from((double) i - 0.5 * m);
      soxr_dd_t window_arg = soxr_dd_sub(soxr_dd_from(1),
          soxr_dd_mul(soxr_dd_mul(z, inv), soxr_dd_mul(z, inv)));
      soxr_dd_t sinc = z.hi == 0?soxr_dd_from(fc) :
        soxr_dd_div(dd_sinpi(soxr_dd_mul_d(z, fc)), soxr_dd_mul(dd_pi(), z));
      soxr_dd_t window = dd_i0(soxr_dd_mul_d(soxr_dd_sqrt(window_arg), beta));
      h[i] = h[m - i] = soxr_dd_mul(soxr_dd_mul(sinc, window), gain);
    }
  }
  return h;
}

/* Extreme-only high-precision cutoff design. */
soxr_dd_t * lsx_design_lpf_extreme_dd(double Fp, double Fs, double Fn, double att,
    int * num_taps, int k, double beta, int accurate_sinc, int spectral_taper)
{
  int n =*num_taps, phases = k > 0?k : 1, modulo = k < 0?-k : 1, i;
  /* Input API limits are binary64. Compute all ensuing cutoff arithmetic
     in DD instead of rounding the transition width to double first. */
  soxr_dd_t fpdd = soxr_dd_div(soxr_dd_from(Fp), soxr_dd_from(fabs(Fn)));
  soxr_dd_t fsdd = soxr_dd_div(soxr_dd_from(Fs), soxr_dd_from(fabs(Fn)));
  soxr_dd_t trdd = soxr_dd_div(soxr_dd_sub(fsdd, fpdd), soxr_dd_from(2. * phases));
  soxr_dd_t fcdd = soxr_dd_sub(soxr_dd_div(fsdd, soxr_dd_from(phases)), trdd);
  double fp = soxr_dd_value(fpdd), fs = soxr_dd_value(fsdd);
  double tr_bw = soxr_dd_value(trdd), fc = soxr_dd_value(fcdd);
  double rho = phases == 1?.5 : att < 120?.63 : .75;
  soxr_dd_t * h = NULL;
  (void) accurate_sinc;
  if(tr_bw > 0.5 * fs / phases) {
    trdd = soxr_dd_div(fsdd, soxr_dd_from(2. * phases));
    tr_bw = soxr_dd_value(trdd);
    fcdd = soxr_dd_sub(soxr_dd_div(fsdd, soxr_dd_from(phases)), trdd);
    fc = soxr_dd_value(fcdd);
  }
  lsx_kaiser_params(att, fc, tr_bw, &beta, num_taps);
  if(!n) * num_taps = phases > 1?
    *num_taps / phases * phases + phases - 1:
    (*num_taps + modulo - 2) / modulo * modulo + 1;
  if(Fn < 0) return NULL;
  if(spectral_taper && k < 0 && phases == 1)
    h = dd_spectral_kaiser(*num_taps, fp, fs, phases);
  if(!h) {
    const int m =*num_taps - 1;
    soxr_dd_t denom = dd_i0(soxr_dd_from(beta));
    soxr_dd_t gain = soxr_dd_div(soxr_dd_from((double) phases), denom);
    soxr_dd_t inv = soxr_dd_div(soxr_dd_from(1),
        soxr_dd_add(soxr_dd_from(0.5 * m), soxr_dd_from(rho)));
    h = malloc((size_t)(*num_taps) * sizeof(*h)); if(!h) return NULL;
    for(i = 0; i <= m / 2; ++i) {
      soxr_dd_t z = soxr_dd_from((double) i - 0.5 * m);
      soxr_dd_t window_arg = soxr_dd_sub(soxr_dd_from(1),
          soxr_dd_mul(soxr_dd_mul(z, inv), soxr_dd_mul(z, inv)));
      soxr_dd_t sinc = z.hi == 0?soxr_dd_from(fc) :
        soxr_dd_div(dd_sinpi(soxr_dd_mul(z, fcdd)), soxr_dd_mul(dd_pi(), z));
      soxr_dd_t window = dd_i0(soxr_dd_mul_d(soxr_dd_sqrt(window_arg), beta));
      h[i] = h[m - i] = soxr_dd_mul(soxr_dd_mul(sinc, window), gain);
    }
  }
  return h;
}

#if WITH_CR64 || !SOXR_LIB
#define DFT_FLOAT double
#define SOXR_LDBL_CONVOLVE 0
#define ORDERED_CONVOLVE lsx_ordered_convolve
#define ORDERED_PARTIAL_CONVOLVE lsx_ordered_partial_convolve
#include "rdft.h"
#undef SOXR_LDBL_CONVOLVE
#endif

#if WITH_CR32
#define DFT_FLOAT float
#define ORDERED_CONVOLVE lsx_ordered_convolve_f
#define ORDERED_PARTIAL_CONVOLVE lsx_ordered_partial_convolve_f
#include "rdft.h"
#endif

double lsx_kaiser_beta(double att, double tr_bw)
{
  if(att >= 60) {
    static const double coefs[][4] = {
      { -6.784957e-10, 1.02856e-05, 0.1087556, -0.8988365 + .001 },
      { -6.897885e-10, 1.027433e-05, 0.10876, -0.8994658 + .002 },
      { -1.000683e-09, 1.030092e-05, 0.1087677, -0.9007898 + .003 },
      { -3.654474e-10, 1.040631e-05, 0.1087085, -0.8977766 + .006 },
      { 8.106988e-09, 6.983091e-06, 0.1091387, -0.9172048 + .015 },
      { 9.519571e-09, 7.272678e-06, 0.1090068, -0.9140768 + .025 },
      { -5.626821e-09, 1.342186e-05, 0.1083999, -0.9065452 + .05 },
      { -9.965946e-08, 5.073548e-05, 0.1040967, -0.7672778 + .085 },
      { 1.604808e-07, -5.856462e-05, 0.1185998, -1.34824 + .1 },
      { -1.511964e-07, 6.363034e-05, 0.1064627, -0.9876665 + .18 },
    };
    double realm = log(tr_bw / .0005) / log(2.);
    double const * c0 = coefs[range_limit((int) realm, 0, (int) array_length(coefs) - 1)];
    double const * c1 = coefs[range_limit(1 + (int) realm, 0, (int) array_length(coefs) - 1)];
    double b0 = ((c0[0] * att + c0[1]) * att + c0[2]) * att + c0[3];
    double b1 = ((c1[0] * att + c1[1]) * att + c1[2]) * att + c1[3];
    return b0 + (b1 - b0) * (realm - (int) realm);
  }
  if(att > 50) return .1102 * (att - 8.7);
  if(att > 20.96) return .58417 * pow(att - 20.96, .4) + .07886 * (att - 20.96);
  return 0;
}



double * lsx_make_lpf(
    int num_taps, double Fc, double beta, double rho, double scale)
{
  int i, m = num_taps - 1;
  double * h = malloc((size_t) num_taps * sizeof(*h));
  double mult = scale / lsx_bessel_I_0(beta), mult1 = 1 / (.5 * m + rho);
  assert(Fc >= 0 && Fc <= 1);
  lsx_debug("make_lpf(n=%i Fc=%.7g beta=%g rho=%g scale=%g)",
      num_taps, Fc, beta, rho, scale);

  if(h) for(i = 0; i <= m / 2; ++i) {
    double z = i - .5 * m, x = z * M_PI, y = z * mult1;
    h[i] = x != 0? sin(Fc * x) / x : Fc;
    h[i] *= lsx_bessel_I_0(beta * sqrt(1 - y * y)) * mult;
    if(m - i != i)
      h[m - i] = h[i];
  }
  return h;
}

void lsx_kaiser_params(double att, double Fc, double tr_bw, double * beta, int * num_taps)
{
  *beta = *beta < 0? lsx_kaiser_beta(att, tr_bw * .5 / Fc) : *beta;
  att = att < 60?(att - 7.95) / (2.285 * M_PI * 2) :
    ((.0007528358 - 1.577737e-05 **beta) **beta + .6248022) **beta + .06186902;
  *num_taps = !*num_taps?(int) ceil(att / tr_bw + 1) : *num_taps;
}

double * lsx_design_lpf(
    double Fp, /* End of pass-band */
    double Fs, /* Start of stop-band */
    double Fn, /* Nyquist freq; e.g. 0.5, 1, PI */
    double att, /* Stop-band attenuation in dB */
    int * num_taps, /* 0: value will be estimated */
    int k, /* >0: number of phases; <0: num_taps = 1 (mod -k) */
    double beta) /* <0: value will be estimated */
{
  int n = *num_taps, phases = max(k, 1), modulo = max(-k, 1);
  double tr_bw, Fc, rho = phases == 1? .5 : att < 120? .63 : .75;

  lsx_debug_more("./sinctest %-12.7g %-12.7g %g 0 %-5g %i %i 50 %g %g -4 >1",
      Fp, Fs, Fn, att, *num_taps, k, beta, rho);

  Fp /= fabs(Fn), Fs /= fabs(Fn); /* Normalise to Fn = 1 */
  tr_bw = .5 * (Fs - Fp); /* Transition band-width: 6dB to stop points */
  tr_bw /= phases, Fs /= phases;
  tr_bw = min(tr_bw, .5 * Fs);
  Fc = Fs - tr_bw;
  assert(Fc - tr_bw >= 0);
  lsx_kaiser_params(att, Fc, tr_bw, &beta, num_taps);
  if(!n)
    *num_taps = phases > 1? *num_taps / phases * phases + phases - 1 :
      (*num_taps + modulo - 2) / modulo * modulo + 1;
  return Fn < 0? 0 : lsx_make_lpf(*num_taps, Fc, beta, rho, (double) phases);
}

static double safe_log(double x)
{
  assert(x >= 0);
  if(x != 0)
    return log(x);
  lsx_debug("log(0)");
  return -26;
}

int lsx_fir_to_phase(double * *h, int * len, int * post_len, double phase)
{
  int * phase_ip = NULL;
  double * phase_w = NULL;
  double * pi_wraps, *work, phase1 = (phase > 50 ? 100 - phase : phase) / 50;
  int i, work_len, begin, end, imp_peak = 0, peak = 0;
  double imp_sum = 0, peak_imp_sum = 0;
  double prev_angle2 = 0, cum_2pi = 0, prev_angle1 = 0, cum_1pi = 0;

  for(i = *len, work_len = 2 * 2 * 8; i > 1; work_len <<= 1, i >>= 1);

  work = calloc((size_t) work_len + 2, sizeof(*work)); /* +2: (UN)PACK */
  pi_wraps = malloc((((size_t) work_len + 2) / 2) * sizeof(*pi_wraps));

  phase_ip = (int *) calloc(dft_br_len(work_len), sizeof(*phase_ip));
  phase_w = (double *) calloc(dft_sc_len(work_len), sizeof(*phase_w));
  if(!work || !pi_wraps || !phase_ip || !phase_w) {
    free(work); free(pi_wraps); free(phase_ip); free(phase_w); return 0;
  }
  memcpy(work, *h, (size_t) * len * sizeof(*work));
  lsx_rdft(work_len, 1, work, phase_ip, phase_w); /* Cepstral: */

  LSX_UNPACK(work, work_len);

  for(i = 0; i <= work_len; i += 2) {
    double angle = atan2(work[i + 1], work[i]);
    double detect = 2 * M_PI;
    double delta = angle - prev_angle2;
    double adjust = detect * ((delta < -detect * .7) - (delta > detect * .7));
    prev_angle2 = angle;
    cum_2pi += adjust;
    angle += cum_2pi;
    detect = M_PI;
    delta = angle - prev_angle1;
    adjust = detect * ((delta < -detect * .7) - (delta > detect * .7));
    prev_angle1 = angle;
    cum_1pi += fabs(adjust); /* fabs for when 2pi and 1pi have combined */
    pi_wraps[i >> 1] = cum_1pi;

    work[i] = safe_log(sqrt(sqr(work[i]) + sqr(work[i + 1])));
    work[i + 1] = 0;
  }
  LSX_PACK(work, work_len);
  lsx_rdft(work_len, -1, work, phase_ip, phase_w);

  for(i = 0; i < work_len; ++i) work[i] *= 2. / work_len;

  for(i = 1; i < work_len / 2; ++i) { /* Window to reject acausal components */
    work[i] *= 2;
    work[i + work_len / 2] = 0;
  }
  lsx_rdft(work_len, 1, work, phase_ip, phase_w);


  for(i = 2; i < work_len; i += 2) /* Interpolate between linear & min phase */
    work[i + 1] = phase1 * i / work_len * pi_wraps[work_len >> 1] +
        (1 - phase1) * (work[i + 1] + pi_wraps[i >> 1]) - pi_wraps[i >> 1];

  work[0] = exp(work[0]), work[1] = exp(work[1]);
  for(i = 2; i < work_len; i += 2) {
    double x = exp(work[i]);
    work[i] = x * cos(work[i + 1]);
    work[i + 1] = x * sin(work[i + 1]);
  }

  lsx_rdft(work_len, -1, work, phase_ip, phase_w);

  for(i = 0; i < work_len; ++i) work[i] *= 2. / work_len;

  /* Find peak pos. */
  for(i = 0; i <= (int)(pi_wraps[work_len >> 1] / M_PI + .5); ++i) {
    imp_sum += work[i];
    if(fabs(imp_sum) > fabs(peak_imp_sum)) {
      peak_imp_sum = imp_sum;
      peak = i;
    }
    if(work[i] > work[imp_peak]) /* For debug check only */
      imp_peak = i;
  }
  while(peak && fabs(work[peak - 1]) > fabs(work[peak]) && work[peak - 1] * work[peak] > 0)
    --peak;

  if(phase1 == 0)
    begin = 0;
  else if(phase1 == 1)
    begin = peak - *len / 2;
  else {
    begin = (int)((.997 - (2 - phase1) * .22) * *len + .5);
    end = (int)((.997 + (0 - phase1) * .22) * *len + .5);
    begin = peak - (begin & ~3);
    end = peak + 1 + ((end + 3) & ~3);
    int newlen = end - begin;
    void *tmp = realloc(*h, (size_t)newlen * sizeof(**h));
    if(!tmp) goto oom;
    *len = newlen;
    *h = tmp;
  }
  for(i = 0; i < *len; ++i)(*h)[i] =
    work[(begin + (phase > 50 ? *len - 1 - i : i) + work_len) & (work_len - 1)];
  *post_len = phase > 50 ? peak - begin : begin + *len - (peak + 1);

  lsx_debug("nPI=%g peak-sum@%i=%g (val@%i=%g); len=%i post=%i (%g%%)",
      pi_wraps[work_len >> 1] / M_PI, peak, peak_imp_sum, imp_peak,
      work[imp_peak], *len, *post_len, 100 - 100. * *post_len / (*len - 1));
  free(pi_wraps); free(work); free(phase_ip); free(phase_w);
  return 1;
oom:
  free(pi_wraps); free(work); free(phase_ip); free(phase_w);
  return 0;
}

#define F_x(F,expr) static double F(double x) {return expr;}
F_x(sinePhi, ((2.0517e-07 * x - 1.1303e-04) * x + .023154) * x + .55924)
F_x(sinePsi, ((9.0667e-08 * x - 5.6114e-05) * x + .013658) * x + 1.0977)
F_x(sinePow, log(.5) / log(sin(x * .5)))
#define dB_to_linear(x) exp((x) * (M_LN10 * 0.05))

double lsx_f_resp(double t, double a)
{
  double x;
  if(t > (a <= 160? .8 : .82)) {
    double a1 = a + 15;
    double p = .00035 * a + .375;
    double w = 1 / (1 - .597) * asin(pow((a1 - 10.6) / a1, 1 / p));
    double c = 1 + asin(pow(1 - a / a1, 1 / p)) / w;
    return a1 * (pow(sin((c - t) * w), p) - 1);
  }
  if(t > .5)
    x = sinePsi(a), x = pow(sin((1 - t) * x), sinePow(x));
  else
    x = sinePhi(a), x = 1 - pow(sin(t * x), sinePow(x));
  return linear_to_dB(x);
}

double lsx_inv_f_resp(double drop, double a)
{
  double x = sinePhi(a), s;
  drop = dB_to_linear(drop);
  s = drop > .5 ? 1 - drop : drop;
  x = asin(pow(s, 1 / sinePow(x))) / x;
  return drop > .5? x : 1 - x;
}
