/* SoX Resampler Library      Copyright (c) 2007-13 robs@users.sourceforge.net
 * Licence for this file: LGPL v2.1                  See LICENCE for details. */

/*
 * SoXR-Poikosoft | Enhanced SoX Resampler Library
 *
 * Poikosoft integration and enhancements Copyright (c) 2026
 * Jukka Poikolainen / Poikosoft | https://www.poikosoft.com
 *
 * Module: Float64 Ooura FFT adapter
 * Features:
 *  - Float64 FFT fallback backend and callback interface.
 *
 * Original third-party copyrights and license terms are preserved.
 */



#include <stdlib.h>
#include "filter.h"
#include "rdft_t.h"
#include "fft4g.c"
#include "soxr-config.h"

#if WITH_CR64
#define DFT_FLOAT double
#define OOURA_RDFT lsx_rdft
#include "fft4g_plan.h"
#undef OOURA_RDFT
#undef DFT_FLOAT
static void nothing(void) {}
static void forward(int length, void * setup, void * h, void * work)
{
  (void) work;
  ooura_transform(length, setup, (double *) h, 1);
}
static void backward(int length, void * setup, void * h, void * work)
{
  (void) work;
  ooura_transform(length, setup, (double *) h, -1);
}
static void convolve(int length, void * setup, void * h, void const * coefs)
{
  lsx_ordered_convolve(length, setup, (double *) h, (double const *) coefs);
}
static void partial(int length, void * h, void const * coefs)
{
  lsx_ordered_partial_convolve(length, (double *) h, (double const *) coefs);
}
static void reorder_noop(int length, void * setup, void * h, void * work)
{
  (void) length; (void) setup; (void) h; (void) work;
}
static int multiplier(void) { return 2; }
static int flags(void) { return 0; }
soxr_rdft_callbacks_t const _soxr_rdft64_cb = {
  ooura_setup, ooura_setup, ooura_delete,
  forward, forward, backward, backward, convolve, partial,
  multiplier, reorder_noop, malloc, calloc, free, flags, ooura_status, NULL
};
#endif
