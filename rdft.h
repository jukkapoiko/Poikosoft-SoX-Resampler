/* SoX Resampler Library      Copyright (c) 2007-13 robs@users.sourceforge.net
 * Licence for this file: LGPL v2.1                  See LICENCE for details. */

/*
 * SoXR-Poikosoft | Enhanced SoX Resampler Library
 *
 * Poikosoft integration and enhancements Copyright (c) 2026
 * Jukka Poikolainen / Poikosoft | https://www.poikosoft.com
 *
 * Module: Ooura real-FFT convolution adapters
 * Features:
 *  - Frequency-domain convolution and reordering for scalar FFT.
 *
 * Original third-party copyrights and license terms are preserved.
 */



void ORDERED_CONVOLVE(int n, void * not_used, DFT_FLOAT * a, const DFT_FLOAT * b)
{
  int i;
  a[0] *= b[0];
  a[1] *= b[1];
  for(i = 2; i < n; i += 2) {
#if defined(SOXR_LDBL_CONVOLVE) && SOXR_LDBL_CONVOLVE
    long double re = (long double) b[i] * (long double) a[i] -
                     (long double) b[i + 1] * (long double) a[i + 1];
    long double im = (long double) b[i + 1] * (long double) a[i] +
                     (long double) b[i] * (long double) a[i + 1];
    a[i] = (DFT_FLOAT) re;
    a[i + 1] = (DFT_FLOAT) im;
#else
    DFT_FLOAT tmp = a[i];
    a[i] = b[i] * tmp - b[i + 1] * a[i + 1];
    a[i + 1] = b[i + 1] * tmp + b[i] * a[i + 1];
#endif
  }
  (void) not_used;
}

void ORDERED_PARTIAL_CONVOLVE(int n, DFT_FLOAT * a, const DFT_FLOAT * b)
{
  int i;
  a[0] *= b[0];
  for(i = 2; i < n; i += 2) {
#if defined(SOXR_LDBL_CONVOLVE) && SOXR_LDBL_CONVOLVE
    long double re = (long double) b[i] * (long double) a[i] -
                     (long double) b[i + 1] * (long double) a[i + 1];
    long double im = (long double) b[i + 1] * (long double) a[i] +
                     (long double) b[i] * (long double) a[i + 1];
    a[i] = (DFT_FLOAT) re;
    a[i + 1] = (DFT_FLOAT) im;
#else
    DFT_FLOAT tmp = a[i];
    a[i] = b[i] * tmp - b[i + 1] * a[i + 1];
    a[i + 1] = b[i + 1] * tmp + b[i] * a[i + 1];
#endif
  }
#if defined(SOXR_LDBL_CONVOLVE) && SOXR_LDBL_CONVOLVE
  a[1] = (DFT_FLOAT)((long double) b[i] * a[i] - (long double) b[i + 1] * a[i + 1]);
#else
  a[1] = b[i] * a[i] - b[i + 1] * a[i + 1];
#endif
}

#undef ORDERED_CONVOLVE
#undef ORDERED_PARTIAL_CONVOLVE
#undef DFT_FLOAT
