/* SoX Resampler Library      Copyright (c) 2007-13 robs@users.sourceforge.net
 * Licence for this file: LGPL v2.1                  See LICENCE for details. */

/*
 * SoXR-Poikosoft | Enhanced SoX Resampler Library
 *
 * Poikosoft integration and enhancements Copyright (c) 2026
 * Jukka Poikolainen / Poikosoft | https://www.poikosoft.com
 *
 * Module: Streaming sample FIFO
 * Features:
 *  - Internal buffering primitives for resampler processing stages.
 *
 * Original third-party copyrights and license terms are preserved.
 */



#ifndef soxr_fifo_included
#define soxr_fifo_included

#include <stdint.h> /* SIZE_MAX */

#if !defined FIFO_SIZE_T
#define FIFO_SIZE_T size_t
#endif

#if !defined FIFO_REALLOC
#include <stdlib.h>
  #define FIFO_REALLOC(a,b,c) realloc(a,b)
  #undef FIFO_FREE
  #define FIFO_FREE free
  #undef FIFO_MALLOC
  #define FIFO_MALLOC malloc
#endif

typedef struct {
  char * data;
  size_t allocation; /* Number of bytes allocated for data. */
  size_t item_size; /* Size of each item in data */
  size_t begin; /* Offset of the first byte to read. */
  size_t end; /* 1 + Offset of the last byte byte to read. */
} fifo_t;

#if !defined FIFO_MIN
  #define FIFO_MIN 0x4000
#endif

#if !defined UNUSED
  #define UNUSED
#endif

UNUSED static void fifo_clear(fifo_t * f)
{
  f->end = f->begin = 0;
}

/* OOM-safe FIFO: preserve the live allocation on realloc failure.
 * Compaction may adjust offsets, but never loses unread samples. */
UNUSED static void * fifo_reserve(fifo_t * f, FIFO_SIZE_T n0)
{
  size_t n = (size_t)n0, total;
  void *new_data;
  if(!f->item_size || n > SIZE_MAX / f->item_size) return NULL;
  n *= f->item_size;
  if(f->begin == f->end) fifo_clear(f);
  if(f->end > SIZE_MAX - n) return NULL;
  if(f->end + n > f->allocation && f->begin > FIFO_MIN) {
    memmove(f->data, f->data + f->begin, f->end - f->begin);
    f->end -= f->begin;
    f->begin = 0;
  }
  if(f->end + n > f->allocation) {
    if(n > SIZE_MAX - f->allocation) return NULL;
    total = f->allocation + n;
    new_data = FIFO_REALLOC(f->data, total, f->allocation);
    if(!new_data) return NULL;
    f->data = (char *)new_data;
    f->allocation = total;
  }
  new_data = f->data ? f->data + f->end : NULL;
  f->end += n;
  return new_data;
}

//UNUSED static void * fifo_write(fifo_t * f, FIFO_SIZE_T n0, void const * data)
static __inline void * fifo_write(fifo_t * f, FIFO_SIZE_T n0, void const * data)
{
  size_t n = (size_t) n0;
  void * s = fifo_reserve(f, n0);
  if(data && s)
    memcpy(s, data, n * f->item_size);
  return s;
}


//UNUSED static void fifo_trim_to(fifo_t * f, FIFO_SIZE_T n0)
static __inline void fifo_trim_to(fifo_t * f, FIFO_SIZE_T n0)
{
  size_t n = (size_t) n0;
  n *= f->item_size;
  f->end = f->begin + n;
}

UNUSED static void fifo_trim_by(fifo_t * f, FIFO_SIZE_T n0)
{
  size_t n = (size_t) n0;
  n *= f->item_size;
  f->end -= n;
}

UNUSED static FIFO_SIZE_T fifo_occupancy(fifo_t * f)
{
  return (FIFO_SIZE_T)((f->end - f->begin) / f->item_size);
}

UNUSED static void * fifo_read(fifo_t * f, FIFO_SIZE_T n0, void * data)
{
  size_t n = (size_t) n0;
  char * ret = f->data ? f->data + f->begin : NULL;
  n *= f->item_size;
  if(n > (f->end - f->begin) || (n && !ret))
    return NULL;
  if(data)
    memcpy(data, ret, (size_t) n);
  f->begin += n;
  return ret;
}

#define fifo_read_ptr(f) fifo_read(f, (FIFO_SIZE_T)0, NULL)


//static void fifo_delete(fifo_t * f)
static __inline void fifo_delete(fifo_t * f)
{
  FIFO_FREE(f->data);
}


//UNUSED static int fifo_create(fifo_t * f, FIFO_SIZE_T item_size)
static __inline int fifo_create(fifo_t * f, FIFO_SIZE_T item_size)
{
  f->item_size = (size_t) item_size;
  f->allocation = FIFO_MIN;
  f->data = NULL;
  fifo_clear(f);
  f->data = FIFO_MALLOC(f->allocation);
  if(!f->data) { f->allocation = 0; return 1; }
  return 0;
}


#endif
