/* Private aligned allocator shared by PFFFT backend translation units. */

/*
 * SoXR-Poikosoft | Enhanced SoX Resampler Library
 *
 * Poikosoft integration and enhancements Copyright (c) 2026
 * Jukka Poikolainen / Poikosoft | https://www.poikosoft.com
 *
 * Module: PFFFT memory utilities
 * Features:
 *  - Aligned allocation helpers shared by the SIMD backends.
 *
 * Original third-party copyrights and license terms are preserved.
 */


#ifndef SOXR_PFFFT_BACKEND_UTIL_H
#define SOXR_PFFFT_BACKEND_UTIL_H
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
static void * soxr_pffft_aligned_malloc(size_t size) {
  void * base; uintptr_t p;
  if(size > SIZE_MAX - 64u - sizeof(void *)) return NULL;
  base = malloc(size + 64u + sizeof(void *));
  if(!base) return NULL;
  p = ((uintptr_t) base + sizeof(void *) + 63u) &~(uintptr_t) 63u;
  ((void **) p)[-1] = base;
  return (void *) p;
}
static void * soxr_pffft_aligned_calloc(size_t n, size_t s) {
  void * p;
  if(s && n > SIZE_MAX / s) return NULL;
  p = soxr_pffft_aligned_malloc(n * s);
  if(p) memset(p, 0, n * s);
  return p;
}
static void soxr_pffft_aligned_free(void * p) { if(p) free(((void **) p)[-1]); }
#endif
