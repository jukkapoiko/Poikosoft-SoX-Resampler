/* SoX Resampler Library      Copyright (c) 2007-16 robs@users.sourceforge.net
 * Licence for this file: LGPL v2.1                  See LICENCE for details. */

/*
 * SoXR-Poikosoft | Enhanced SoX Resampler Library
 *
 * Poikosoft integration and enhancements Copyright (c) 2026
 * Jukka Poikolainen / Poikosoft | https://www.poikosoft.com
 *
 * Module: Float32 Fast engine
 * Features:
 *  - Instantiates the shared Fast processing core with float samples.
 *
 * Original third-party copyrights and license terms are preserved.
 */



#define RATE_CB    _soxr_rate32_cb
#define CORE_STR   "cr32"

#define CORE_TYPE  0
#include "cr-core.c"
