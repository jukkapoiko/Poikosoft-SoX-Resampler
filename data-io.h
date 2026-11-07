/* SoX Resampler Library      Copyright (c) 2007-13 robs@users.sourceforge.net
 * Licence for this file: LGPL v2.1                  See LICENCE for details. */

/*
 * SoXR-Poikosoft | Enhanced SoX Resampler Library
 *
 * Poikosoft integration and enhancements Copyright (c) 2026
 * Jukka Poikolainen / Poikosoft | https://www.poikosoft.com
 *
 * Module: Audio I/O declarations
 * Features:
 *  - Internal input/output conversion interfaces.
 *
 * Original third-party copyrights and license terms are preserved.
 */



#if !defined soxr_data_io_included
#define soxr_data_io_included

#include "soxr.h"

void _soxr_deinterleave(
    double * *dest,
    soxr_datatype_t data_type,
    void const * *src0,
    size_t n,
    unsigned ch);

void _soxr_deinterleave_f(
    float * *dest,
    soxr_datatype_t data_type,
    void const * *src0,
    size_t n,
    unsigned ch);

size_t /* clips */ _soxr_interleave(
    soxr_datatype_t data_type,
    void * *dest,
    double const * const * src,
    size_t n,
    unsigned ch,
    unsigned long * seed);

size_t /* clips */ _soxr_interleave_f(
    soxr_datatype_t data_type,
    void * *dest,
    float const * const * src,
    size_t n,
    unsigned ch,
    unsigned long * seed);

#endif
