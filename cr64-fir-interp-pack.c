/* Optional Fast64 interpolating FIR coefficient packing.
 * Common scalar code (safe on all supported CPU instruction sets).
 * Original coefficients are retained unchanged for fallback.
 */
#include <stdint.h>
#include <string.h>
#include "soxr-config.h"
#include "internal.h"
typedef double sample_t;
#include "cr.h"
#include "pffft-backend-util.h"

int soxr_fast64_interp_pack(stage_t * p, int order)
{
  int phase, k, j;
  int const n = p->n, nvec = p->n & ~3, bits = p->phase_bits;
  int phases;
  size_t count;
  double * target;
  double const * source = (double const *) p->shared->poly_fir_coefs;
  if (!source || nvec < 4 || bits < 0 || bits > 16 || order < 1 || order > 3)
    return 0;
  if (p->shared->poly_fir_simd_coefs)
    return 1;
  phases = 1 << bits;
  count = (size_t) phases * nvec * (order + 1);
  if (count > SIZE_MAX / sizeof(double))
    return 0;
  target = (double *) soxr_pffft_aligned_calloc(count, sizeof(double));
  if (!target)
    return 0;
  for (phase = 0; phase < phases; ++phase)
    for (j = 0; j < nvec; ++j)
      for (k = 0; k <= order; ++k)
        coef4(target, order, nvec, phase, k, j) =
          coef(source, order, n, phase, k, j);
  p->shared->poly_fir_simd_coefs = (real *) target;
  return 1;
}

