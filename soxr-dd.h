/* Two-double expansion: no FP80, AVX or FMA required. All transforms
   assume IEEE round-to-nearest and no unsafe reassociation/fast-math. */

/*
 * SoXR-Poikosoft | Enhanced SoX Resampler Library
 *
 * Poikosoft integration and enhancements Copyright (c) 2026
 * Jukka Poikolainen / Poikosoft | https://www.poikosoft.com
 *
 * Module: Double-double arithmetic
 * Features:
 *  - Two-double hi/lo arithmetic without mandatory AVX/FMA.
 *  - Requires strict floating-point evaluation.
 *
 * Original third-party copyrights and license terms are preserved.
 */




#if !defined soxr_dd_included
#define soxr_dd_included

#include <math.h>
#include <float.h>

typedef struct { double hi, lo; } soxr_dd_t;
static inline soxr_dd_t soxr_dd_make(double hi, double lo)
{ soxr_dd_t r = { hi, lo }; return r; }
static inline soxr_dd_t soxr_dd_from(double x) { return soxr_dd_make(x, 0); }
static inline soxr_dd_t soxr_dd_add(soxr_dd_t a, soxr_dd_t b)
{
  double s = a.hi + b.hi, v = s - a.hi;
  double e = (a.hi - (s - v)) + (b.hi - v);
  double t = a.lo + b.lo + e, h = s + t;
  return soxr_dd_make(h, t - (h - s));
}
static inline soxr_dd_t soxr_dd_neg(soxr_dd_t a)
{ return soxr_dd_make(-a.hi, -a.lo); }
static inline soxr_dd_t soxr_dd_sub(soxr_dd_t a, soxr_dd_t b)
{ return soxr_dd_add(a, soxr_dd_neg(b)); }
/* FMA TwoProd: exact residual for finite operands without overflow/underflow.
   Strict IEEE-754 rounding is mandatory: disable fast-math/reassociation. */
static inline soxr_dd_t soxr_dd_product(double a, double b)
{
  double p = a * b;
  return soxr_dd_make(p, fma(a, b, -p));
}
static inline soxr_dd_t soxr_dd_mul(soxr_dd_t a, soxr_dd_t b)
{
  soxr_dd_t p = soxr_dd_product(a.hi, b.hi);
  p = soxr_dd_add(p, soxr_dd_product(a.hi, b.lo));
  p = soxr_dd_add(p, soxr_dd_product(a.lo, b.hi));
  return soxr_dd_add(p, soxr_dd_product(a.lo, b.lo));
}
static inline soxr_dd_t soxr_dd_mul_d(soxr_dd_t a, double b)
{ return soxr_dd_mul(a, soxr_dd_from(b)); }
static inline soxr_dd_t soxr_dd_div(soxr_dd_t a, soxr_dd_t b)
{
  double q = a.hi / b.hi;
  soxr_dd_t v = soxr_dd_from(q);
  v = soxr_dd_add(v, soxr_dd_from(soxr_dd_sub(a, soxr_dd_mul(b, v)).hi / b.hi));
  return soxr_dd_add(v, soxr_dd_from(soxr_dd_sub(a, soxr_dd_mul(b, v)).hi / b.hi));
}
static inline soxr_dd_t soxr_dd_sqrt(soxr_dd_t a)
{
  if(a.hi <= 0) return soxr_dd_from(0);
  soxr_dd_t v = soxr_dd_from(sqrt(a.hi));
  return soxr_dd_add(v, soxr_dd_div(soxr_dd_sub(a, soxr_dd_mul(v, v)), soxr_dd_mul_d(v, 2)));
}
static inline double soxr_dd_value(soxr_dd_t a) { return a.hi + a.lo; }

#endif