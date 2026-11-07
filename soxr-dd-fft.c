/*
 * SoXR-Poikosoft | Enhanced SoX Resampler Library
 *
 * Poikosoft integration and enhancements Copyright (c) 2026
 * Jukka Poikolainen / Poikosoft | https://www.poikosoft.com
 *
 * Module: Extreme double-double FFT
 * Features:
 *  - Radix-2 FFT operating on double-double real/imaginary values.
 *
 * Original third-party copyrights and license terms are preserved.
 */

#include "soxr-dd-fft.h"
#include <stdint.h>
#include <stdlib.h>
#include <limits.h>

static soxr_dd_t dd_pi(void) {
  return soxr_dd_make(3.141592653589793116,
                      1.2246467991473532072e-16);
}
static void dd_sincos(soxr_dd_t x, soxr_dd_t * s, soxr_dd_t * c) {
  /* Used only for FFT stage angle in [-pi, pi]. */
  soxr_dd_t xx = soxr_dd_mul(x, x);
  soxr_dd_t st = x, ct = soxr_dd_from(1.);
  *s = st; *c = ct;
  for(int k = 1; k <= 45; ++k) {
    st = soxr_dd_mul(st, soxr_dd_div(soxr_dd_neg(xx),
         soxr_dd_from((double)(2 * k) * (2 * k + 1))));
    ct = soxr_dd_mul(ct, soxr_dd_div(soxr_dd_neg(xx),
         soxr_dd_from((double)(2 * k - 1) * (2 * k))));
    *s = soxr_dd_add(*s, st);
    *c = soxr_dd_add(*c, ct);
    if(st.hi == 0. && ct.hi == 0.) break;
  }
}
static soxr_dd_complex cmul(soxr_dd_complex a, soxr_dd_complex b) {
  soxr_dd_complex r;
  r.re = soxr_dd_sub(soxr_dd_mul(a.re, b.re), soxr_dd_mul(a.im, b.im));
  r.im = soxr_dd_add(soxr_dd_mul(a.re, b.im), soxr_dd_mul(a.im, b.re));
  return r;
}
void soxr_dd_scalar_plan_destroy(soxr_dd_scalar_plan * plan);

struct soxr_dd_scalar_plan {
  size_t n;
  size_t num_stages;
  size_t * offsets;
  soxr_dd_complex * forward;
  soxr_dd_complex * inverse;
};

soxr_dd_scalar_plan * soxr_dd_scalar_plan_create(size_t n)
{
  soxr_dd_scalar_plan * plan;
  size_t len, i;
  if(n < 2 || (n & (n - 1)) || n > (size_t)INT_MAX) return NULL;
  plan = calloc(1, sizeof(*plan));
  if(!plan) return NULL;
  for(len = n; len > 1; len >>= 1) ++plan->num_stages;
  /* Sum of lengths / 2 over radix-2 stages is n - 1. */
  if(n > SIZE_MAX / sizeof(*plan->forward)) {
    soxr_dd_scalar_plan_destroy(plan);
    return NULL;
  }
  plan->offsets = calloc(plan->num_stages, sizeof(*plan->offsets));
  plan->forward = calloc(n - 1, sizeof(*plan->forward));
  plan->inverse = calloc(n - 1, sizeof(*plan->inverse));
  if(!plan->offsets || !plan->forward || !plan->inverse) {
    soxr_dd_scalar_plan_destroy(plan);
    return NULL;
  }
  plan->n = n;
  len = 2;
  {
    size_t offset = 0;
    for(i = 0; i < plan->num_stages; ++i, len <<= 1) {
      soxr_dd_t a = soxr_dd_div(soxr_dd_mul_d(dd_pi(), -2.), soxr_dd_from((double) len));
      soxr_dd_t b = soxr_dd_div(soxr_dd_mul_d(dd_pi(), 2.), soxr_dd_from((double) len));
      soxr_dd_complex fwlen, bwlen;
      soxr_dd_complex fw = {{ 1., 0. }, { 0., 0. } };
      soxr_dd_complex bw = {{ 1., 0. }, { 0., 0. } };
      dd_sincos(a, &fwlen.im, &fwlen.re);
      dd_sincos(b, &bwlen.im, &bwlen.re);
      plan->offsets[i] = offset;
      for(size_t j = 0; j < len / 2; ++j) {
        plan->forward[offset + j] = fw;
        plan->inverse[offset + j] = bw;
        fw = cmul(fw, fwlen);
        bw = cmul(bw, bwlen);
      }
      offset += len / 2;
    }
  }
  return plan;
}

size_t soxr_dd_scalar_plan_length(soxr_dd_scalar_plan const * plan)
{
  return plan ? plan->n : 0;
}

soxr_dd_complex const * soxr_dd_scalar_plan_stage(soxr_dd_scalar_plan const * plan,
                                                  size_t stage, int inverse)
{
  return (inverse ? plan->inverse : plan->forward) + plan->offsets[stage];
}

void soxr_dd_scalar_plan_destroy(soxr_dd_scalar_plan * plan)
{
  if(!plan) return;
  free(plan->forward);
  free(plan->inverse);
  free(plan->offsets);
  free(plan);
}

int soxr_dd_fft_planned(soxr_dd_complex * a, size_t n, int inverse,
                         soxr_dd_scalar_plan const * plan) {
  size_t stage_index = 0;
  if(plan && plan->n != n) return -1;

  if(!a || n == 0 || (n & (n - 1))) return -1;
  for(size_t i = 1, j = 0; i < n; ++i) {
    size_t bit = n >> 1;
    for(; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if(i < j) { soxr_dd_complex t = a[i]; a[i] = a[j]; a[j] = t; }
  }
  for(size_t len = 2; len <= n; len<<=1) {
    soxr_dd_complex wlen;
    soxr_dd_complex const * twiddles = NULL;
    if(plan) twiddles = soxr_dd_scalar_plan_stage(plan, stage_index, inverse);
    else {
      soxr_dd_t theta = soxr_dd_div(soxr_dd_mul_d(dd_pi(), inverse ? 2. : -2.),
                                    soxr_dd_from((double) len));
      dd_sincos(theta, &wlen.im, &wlen.re);
    }
    ++stage_index;
    for(size_t i = 0; i < n; i += len) {
      if(twiddles) {
        for(size_t j = 0; j < len / 2; ++j) {
          soxr_dd_complex u = a[i + j],
                          v = cmul(a[i + j + len / 2], twiddles[j]);
          a[i + j].re = soxr_dd_add(u.re, v.re);
          a[i + j].im = soxr_dd_add(u.im, v.im);
          a[i + j + len / 2].re = soxr_dd_sub(u.re, v.re);
          a[i + j + len / 2].im = soxr_dd_sub(u.im, v.im);
        }
      } else {
        soxr_dd_complex w = {{ 1., 0. }, { 0., 0. } };
        for(size_t j = 0; j < len / 2; ++j) {
          soxr_dd_complex u = a[i + j], v = cmul(a[i + j + len / 2], w);
          a[i + j].re = soxr_dd_add(u.re, v.re);
          a[i + j].im = soxr_dd_add(u.im, v.im);
          a[i + j + len / 2].re = soxr_dd_sub(u.re, v.re);
          a[i + j + len / 2].im = soxr_dd_sub(u.im, v.im);
          w = cmul(w, wlen);
        }
      }
    }
    if(len == n) break;
  }
  if(inverse) {
    for(size_t i = 0; i < n; ++i) {
      a[i].re = soxr_dd_div(a[i].re, soxr_dd_from((double) n));
      a[i].im = soxr_dd_div(a[i].im, soxr_dd_from((double) n));
    }
  }
  return 0;
}

int soxr_dd_fft(soxr_dd_complex * a, size_t n, int inverse)
{
  return soxr_dd_fft_planned(a, n, inverse, NULL);
}
