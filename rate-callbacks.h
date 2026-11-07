/* Typed engine callbacks. All function calls use their declared ABI. */
#ifndef SOXR_RATE_CALLBACKS_H
#define SOXR_RATE_CALLBACKS_H
#include <stddef.h>
#include "soxr.h"
typedef struct soxr_rate_callbacks {
  void * (*input)(void *, void const *, size_t);
  void (*process)(void *, size_t);
  void const * (*output)(void *, void *, size_t *);
  void (*flush)(void *);
  void (*close)(void *);
  double (*delay)(void *);
  void (*sizes)(size_t *, size_t *);
  char const * (*create)(void *, void *, double,
      soxr_quality_spec_t *, soxr_runtime_spec_t *, double);
  void (*set_io_ratio)(void *, double, size_t);
  char const * (*id)(void);
  char const * (*status)(void *);
} soxr_rate_callbacks_t;
#endif
