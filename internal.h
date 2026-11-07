/* SoX Resampler Library      Copyright (c) 2007-16 robs@users.sourceforge.net
 * Licence for this file: LGPL v2.1                  See LICENCE for details. */

/*
 * SoXR-Poikosoft | Enhanced SoX Resampler Library
 *
 * Poikosoft integration and enhancements Copyright (c) 2026
 * Jukka Poikolainen / Poikosoft | https://www.poikosoft.com
 *
 * Module: Internal library interfaces
 * Features:
 *  - Common internal types, helpers and build declarations.
 *
 * Original third-party copyrights and license terms are preserved.
 */



#if !defined soxr_internal_included
#define soxr_internal_included

#include "std-types.h"



#undef min
#undef max
#define min(a, b) ((a) <= (b) ? (a) : (b))
#define max(a, b) ((a) >= (b) ? (a) : (b))



#define range_limit(x, lower, upper) (min(max(x, lower), upper))
#define linear_to_dB(x) (log10(x) * 20)
#define array_length(a) (sizeof(a)/sizeof(a[0]))
#if !defined AL
#define AL(a) array_length(a)
#endif
#define sqr(a) ((a) * (a))



#if defined __GNUC__
  #define UNUSED __attribute__ ((unused))
#else
  #define UNUSED
#endif



/* Tracing is disabled: no process-global tracing state. */
#define lsx_debug if(0) lsx_dummy
#define lsx_debug_more lsx_debug
#if defined(__GNUC__)
void lsx_dummy(char const *, ...);
#else
static __inline void lsx_dummy(char const * x, ...) { (void) x; }
#endif

/* soxr_quality_spec_t.flags: */

#define SOXR_ROLLOFF_LSR2Q     3u    /* Reserved for internal use. */
#define SOXR_MAINTAIN_3DB_PT   4u    /* Reserved for internal use. */
#define SOXR_PROMOTE_TO_LQ    64u    /* Reserved for internal use. */



/* soxr_runtime_spec_t.flags: */
#define SOXR_NOSMALLINTOPT     8u    /* For test purposes only. */



/* soxr_quality_spec recipe: */

//#define SOXR_PRECISIONQ         11   /* Quality specified by the precision parameter. */



/* soxr_quality_spec flags: */

#define RESET_ON_CLEAR   (1u<<31)



#endif
