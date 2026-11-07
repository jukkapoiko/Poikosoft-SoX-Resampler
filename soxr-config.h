/* SoX Resampler Library      Copyright (c) 2007-16 robs@users.sourceforge.net
 * Licence for this file: LGPL v2.1                  See LICENCE for details. */

/*
 * SoXR-Poikosoft | Enhanced SoX Resampler Library
 *
 * Poikosoft integration and enhancements Copyright (c) 2026
 * Jukka Poikolainen / Poikosoft | https://www.poikosoft.com
 *
 * Module: Build and CPU-path configuration
 * Features:
 *  - USE_SIMD: AUTO, DISABLE, SSE2, AVX or AVX2+FMA3.
 *  - Fast engine options and high-precision sample-clock settings.
 *  - EXTREME_USE_SIMD: AUTO, DISABLE, SSE2, AVX or AVX2+FMA3 for Extreme DD.
 *  - Extreme SIMD choice independent of the Fast SIMD selector.
 *
 * Original third-party copyrights and license terms are preserved.
 */



/* N.B. Pre-configured for modern MS-Windows systems.  However, the normal
 * procedure is to use the cmake configuration and build system. See INSTALL. */

#if !defined soxr_config_included
#define soxr_config_included

#include "soxr-dd.h"

#define HAVE_FENV_H 1
#define HAVE_STDBOOL_H 1
#define HAVE_STDINT_H 1
#define HAVE_LRINT 1
#define HAVE_BIGENDIAN 0

/* Automatic LPC boundary extrapolation.
 * 32-order Welch LPC, gamma 0.999, 15 ms target-rate analysis */
#define SOXR_AUTO_LPC_EXTRAPOLATION 1
#define SOXR_LPC_MEMORY_LIMIT_MIB 512


/* Never fake __AVX__: enable real AVX via -mavx or /arch:AVX. */

/* FFT backend selection (Fast Float32 and Fast Float64 only).
 * SIMD_AUTO    = default: CPUID + XGETBV, fastest supported FFT backend.
 * SIMD_DISABLE = force Ooura FFT (no explicit PFFFT SIMD kernels).
 * SIMD_SSE2    = force original SSE PFFFT for Float32 (SSE2 CPU baseline);
 *                paired-SSE2 PFFFT for Float64
 * SIMD_AVX     = force PFFFT AVX for Float32 and Float64.
 * SIMD_AVX2    = force PFFFT AVX2+FMA3 for Float32 and Float64.
 *
 * Forced SSE/AVX/AVX2 modes return an error from soxr_create() if the
 * selected CPU instruction set is unavailable. AVX2 REQUIRES FMA3.
 * Extreme uses its own double-double FFT and ignores this setting.
 * CPU selection runs during each Fast engine's initialization, not per FFT.
 *
 * SIMD_SSE2 is a Fast FFT dispatch enum value. The internal SIMD_SSE
 * preprocessor macro in the original FIR kernels is unrelated and unchanged.
 */
typedef enum soxr_fft_simd_mode {
  SIMD_DISABLE = 0,
  SIMD_AUTO = 1,
  SIMD_SSE2 = 2,
  SIMD_AVX = 3,
  SIMD_AVX2 = 4
} soxr_fft_simd_mode_t;

#ifndef USE_SIMD
#define USE_SIMD SIMD_AUTO
/* To force an FFT backend instead, replace the line above with ONE of:
 * #define USE_SIMD SIMD_DISABLE  // Ooura F32/F64
 * #define USE_SIMD SIMD_SSE2     // SSE PFFFT F32, SSE2 PFFFT F64
 * #define USE_SIMD SIMD_AVX      // PFFFT AVX F32/F64
 * #define USE_SIMD SIMD_AVX2     // PFFFT AVX2 + FMA3 F32/F64
 */
#endif
/* Select exactly one of SIMD_DISABLE/SIMD_AUTO/SIMD_SSE2/SIMD_AVX/SIMD_AVX2. */
typedef char soxr_use_simd_mode_must_be_valid[
  (USE_SIMD == SIMD_DISABLE || USE_SIMD == SIMD_AUTO ||
   USE_SIMD == SIMD_SSE2 || USE_SIMD == SIMD_AVX ||
   USE_SIMD == SIMD_AVX2) ? 1 : -1];

/* Fast64 FIR dispatch (always enabled for SIMD PFFFT backends):
 * SSE2: SSE2 vpoly0/1/2/3.
 * AVX:  external AVX vpoly0 and AVX vpoly1/2/3.
 * AVX2: external AVX vpoly0 and AVX2 vpoly1/2/3.
 * Scalar kernels remain available for SIMD_DISABLE and safe initialization.
 * ISA-specific code is isolated in separate translation units. */

/* Fast SIMD FIR is built into cr32/cr64 and enabled only when the engine
 * selects a supported SIMD FFT backend. No separate cr32s/cr64s engines. */

/* Extreme full double-double engine SIMD selector (independent of USE_SIMD).
 * EXTREME_SIMD_AUTO:    AVX2+FMA3, then AVX, then SSE2, then scalar DD.
 * EXTREME_SIMD_DISABLE: Force the complete original scalar DD engine.
 * EXTREME_SIMD_AVX2:    Require AVX2+FMA3; soxr_create() fails gracefully if absent.
 * EXTREME_SIMD_AVX:     Require AVX with OS YMM support; no AVX2/FMA3 required.
 *
 * The dedicated AVX-only DD backend does not require AVX2 or FMA3.
 * Detection occurs once per rate-engine initialization; no per-sample CPU
 * probing.  All three modes use the same DD filter/coefficient precision and
 * linear-phase topology.  AVX/AVX2 never become baseline ISA requirements.
 *
 * The AVX backend accelerates FFT butterflies, retaining scalar DD multiplication
 * and FIR arithmetic for exact results.  Use enum constants for the mode.
 */
typedef enum soxr_extreme_simd_mode {
  EXTREME_SIMD_DISABLE = 0,
  EXTREME_SIMD_AUTO = 1,
  EXTREME_SIMD_SSE2 = 2,
  EXTREME_SIMD_AVX = 3,
  EXTREME_SIMD_AVX2 = 4   
} soxr_extreme_simd_mode_t;

#ifndef EXTREME_USE_SIMD
#define EXTREME_USE_SIMD EXTREME_SIMD_AUTO
#endif
/* Validate the selected EXTREME SIMD mode at compilation. */
typedef char soxr_extreme_use_simd_mode_must_be_valid[
  (EXTREME_USE_SIMD == EXTREME_SIMD_DISABLE ||
   EXTREME_USE_SIMD == EXTREME_SIMD_AUTO ||
   EXTREME_USE_SIMD == EXTREME_SIMD_SSE2 ||
   EXTREME_USE_SIMD == EXTREME_SIMD_AVX ||
   EXTREME_USE_SIMD == EXTREME_SIMD_AVX2) ? 1 : -1];

#define WITH_CR32 1
#define WITH_CR64 1
#define WITH_VR32 0

#define WITH_HI_PREC_CLOCK 1
#define WITH_FLOAT_STD_PREC_CLOCK 0
#define WITH_DEV_TRACE 0

#endif
