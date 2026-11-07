/* SoX Resampler Library      Copyright (c) 2007-13 robs@users.sourceforge.net
 * Licence for this file: LGPL v2.1                  See LICENCE for details. */

/*
 * SoXR-Poikosoft | Enhanced SoX Resampler Library
 *
 * Poikosoft integration and enhancements Copyright (c) 2026
 * Jukka Poikolainen / Poikosoft | https://www.poikosoft.com
 *
 * Module: Filter-design declarations
 * Features:
 *  - FIR design, response measurement and convolution interfaces.
 *
 * Original third-party copyrights and license terms are preserved.
 */



#if !defined soxr_filter_included
#define soxr_filter_included

#include "soxr-config.h"
#include "aliases.h"

soxr_dd_t * lsx_design_lpf_extreme_dd(double Fp, double Fs, double Fn, double att,
    int * num_taps, int k, double beta, int accurate_sinc, int spectral_taper);
soxr_dd_t * lsx_design_lpf_dd(double Fp, double Fs, double Fn, double att,
    int * num_taps, int k, double beta, int accurate_sinc, int spectral_taper);
double lsx_bessel_I_0(double x);
#define lsx_is_power_of_2(x) (!( (x) < 2 || ((x) & ((x) - 1))))
void lsx_ordered_convolve(int n, void * not_used, double * a, const double * b);
void lsx_ordered_convolve_f(int n, void * not_used, float * a, const float * b);
void lsx_ordered_partial_convolve(int n, double * a, const double * b);
void lsx_ordered_partial_convolve_f(int n, float * a, const float * b);

double lsx_kaiser_beta(double att, double tr_bw);
double * lsx_make_lpf(int num_taps, double Fc, double beta, double rho,
    double scale);
void lsx_kaiser_params(double att, double Fc, double tr_bw, double * beta, int * num_taps);
double * lsx_design_lpf(
    double Fp, /* End of pass-band */
    double Fs, /* Start of stop-band */
    double Fn, /* Nyquist freq; e.g. 0.5, 1, PI; < 0: dummy run */
    double att, /* Stop-band attenuation in dB */
    int * num_taps, /* 0: value will be estimated */
    int k, /* >0: number of phases; <0: num_taps = 1 (mod -k) */
    double beta); /* <0: value will be estimated */

int lsx_fir_to_phase(double * *h, int * len,
    int * post_len, double phase0);

double lsx_f_resp(double t, double a);
double lsx_inv_f_resp(double drop, double a);
#define lsx_to_3dB(a) (1 - lsx_inv_f_resp(-3., a))

#endif
