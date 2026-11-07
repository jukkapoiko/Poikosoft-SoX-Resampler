/* Poikosoft automatic LPC boundary extrapolation, 2026.
 * Standalone implementation include of soxr.c, C11 / MSVC compatible.
 * Adapted from the LPC approach in lvqcl's lpc.cpp:
 *
 * Copyright (c) 2013, 2018 lvqcl
 *
 * Permission to use, copy, modify, and distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 *
 * Poikosoft additions: automatic streaming state, configurable LPC,
 * gamma^k bandwidth expansion, stable edge-hold fallback, bounded
 * rational-period padding, direct streaming output, fixed-memory boundary scratch, exact output-duration cropping.
 */
#ifndef SOXR_POIKOSOFT_LPC_EXTRAP_H
#define SOXR_POIKOSOFT_LPC_EXTRAP_H
#include <limits.h>
#include <stdint.h>
#include <float.h>

/* LPC analysis and padding defaults. Only the enable switch and memory
 * budget are configured in soxr-config.h. */
#define SOXR_LPC_ORDER 32
#define SOXR_LPC_GAMMA 0.999
#define SOXR_LPC_TARGET_WINDOW_MILLISECONDS 15
#define SOXR_LPC_PAD_MILLISECONDS 50

#if SOXR_LPC_MEMORY_LIMIT_MIB < 1 || SOXR_LPC_MEMORY_LIMIT_MIB > 4096
#error "SOXR_LPC_MEMORY_LIMIT_MIB must be between 1 and 4096"
#endif

#define SOXR_LPC_DISCARD_FRAMES 1024
#define SOXR_LPC_ERROR_ALLOC "malloc failed: LPC extrapolation buffers"
/* MSVC C exposes _finite rather than ISO isfinite on older toolsets. */
#if defined(_MSC_VER) && !defined(__clang__)
#define soxr_lpc_isfinite(x) (_finite(x) != 0)
#else
#define soxr_lpc_isfinite(x) isfinite(x)
#endif

struct soxr_lpc_state {
  size_t pre_n, post_n, prime_n, input_bytes, output_bytes;
  size_t nch, prime_used, tail_used, pre_pos, prime_pos, post_pos;
  size_t drop_front;
  size_t allocated_bytes, memory_limit_bytes;
  size_t real_in, total_out;
  double input_rate, output_rate;
  int is_double_input;
  int phase; /* 0 primer, 1 prefix, 2 primer feed, 3 stream, 4 suffix, 5 flush, 6 done */
  int eof_requested;
  int suffix_ready;
  unsigned char *prime, *tail, *prefix, *raw, *work_mem;
};

static int soxr_lpc_product(size_t a, size_t b, size_t *out)
{
  if (b && a > SIZE_MAX / b) return 0;
  *out = a*b; return 1;
}

static uint64_t soxr_lpc_gcd(uint64_t a, uint64_t b)
{
  while (b) { uint64_t t = a%b; a = b; b = t; }
  return a;
}

/* The target output sample rate determines the analysis window:
 *   output_window = floor_power_of_two(output_rate * window_ms / 1000)
 *   input_window   =  ceil(output_window * input_rate / output_rate)
 * The fit always uses input PCM frames (never raw DSD bits).
 * The padding budget is a separate parameter and may be raised by a
 * specified effective input-domain FIR support requirement. */
static int soxr_lpc_target_window(double out_hz, size_t *window)
{
  double desired = floor(out_hz * (double)SOXR_LPC_TARGET_WINDOW_MILLISECONDS / 1000.);
  size_t n = 1;
  if (!soxr_lpc_isfinite(desired) || desired < 1.) return 0;
  while (n <= SIZE_MAX/2 && (double)(n*2) <= desired) n*= 2;
  *window = n;
  return 1;
}

static int soxr_lpc_size_ceil(double n, size_t *out)
{
  if (!soxr_lpc_isfinite(n) || !(n >= 0.) || n >= (double)SIZE_MAX) return 0;
  *out = (size_t)ceil(n);
  return 1;
}

/* Compute padding in real input frames and its nearest output-frame count.
 * Prefer an exact integer-rate ratio period if its size is reasonable.
 * The memory budget limits padding and reports allocation errors. */
static int soxr_lpc_padding(double in_hz, double out_hz,
                            size_t target_window, size_t *pre, size_t *post)
{
  size_t want_out = target_window, want_in;
  uint64_t a, b, g, r1, r2, m;
  double ratio = in_hz/out_hz;
  double by_time = out_hz * (double)SOXR_LPC_PAD_MILLISECONDS/1000.;
  if (!soxr_lpc_isfinite(ratio) || ratio <= 0. || !soxr_lpc_isfinite(by_time)) return 0;
  if (!soxr_lpc_size_ceil(fmax((double)want_out, by_time), &want_out)) return 0;
  if (!soxr_lpc_size_ceil((double)want_out*ratio, &want_in)) return 0;
  if (!soxr_lpc_size_ceil((double)want_in/ratio, &want_out)) return 0;
  /* Exact integer-ratio padding avoids fractional initial trimming when
   * a complete ratio period can fit with <=25% over-padding. */
  if (in_hz >= 1. && out_hz >= 1. && in_hz <= UINT32_MAX && out_hz <= UINT32_MAX &&
     floor(in_hz) == in_hz && floor(out_hz) == out_hz) {
    a = (uint64_t)in_hz; b = (uint64_t)out_hz;
    g = soxr_lpc_gcd(a, b); r1 = a/g; r2 = b/g;
    m = (uint64_t)ceil(fmax((double)want_in/r1, (double)want_out/r2));
    if (m > 0 && m <= SIZE_MAX/r1 && m <= SIZE_MAX/r2 &&
       (double)r1*m <= 1.25*(double)want_in &&
       (double)r2*m <= 1.25*(double)want_out) {
      *pre = (size_t)(r1*m);*post = (size_t)(r2*m);return 1;
    }
  }
  /* When exact rate periods are too large, seek a small over-padding
   * whose converted duration is nearly an integer. This avoids phase
   * rounding at both boundaries without enormous rational-period pads. */
  {
    size_t best = want_in, k;
    double best_error = DBL_MAX;
    for (k = 0; k <= 256 && want_in <= SIZE_MAX-k; ++k) {
      size_t n = want_in+k;
      double scaled = (double)n/ratio;
      double error = fabs(scaled-floor(scaled+0.5));
      if (!soxr_lpc_isfinite(error))return 0;
      if (error < best_error) {best_error = error; best = n; }
      if (error < 0.001)break;
    }
    want_in = best;
  }
  *pre = want_in;
  if (!soxr_lpc_size_ceil(floor((double)want_in/ratio+0.5), post))return 0;
  if (!*pre || !*post)return 0;
  return 1;
}

static int soxr_lpc_memory_add(size_t frames, size_t stride, size_t *total)
{
  size_t n;
  if (!soxr_lpc_product(frames, stride, &n) || *total > SIZE_MAX-n) return 0;
  *total+= n;
  return 1;
}

static void soxr_lpc_free(struct soxr_lpc_state *s)
{
  if (!s) return;
  free(s->prime); free(s->tail); free(s->prefix);
  free(s->raw); free(s->work_mem);
  free(s);
}

static soxr_error_t soxr_lpc_new(soxr_t p, double inrate, double outrate)
{
  struct soxr_lpc_state *s;
  size_t n;
  soxr_error_t why = 0;
  if (!(s = (struct soxr_lpc_state*)calloc(1, sizeof (*s))))
    return "malloc failed: LPC state";
  if (!(SOXR_LPC_GAMMA > 0. && SOXR_LPC_GAMMA <= 1.) ||
      !soxr_lpc_isfinite((double)SOXR_LPC_GAMMA))
    {why = "invalid LPC gamma parameter";goto failed;}
  s->is_double_input = (p->io_spec.itype & 3) == SOXR_FLOAT64;
  s->nch = p->num_channels;
  if (!soxr_lpc_product(soxr_datatype_size(p->io_spec.itype), s->nch,
                       &s->input_bytes) ||
     !soxr_lpc_product(soxr_datatype_size(p->io_spec.otype), s->nch,
                       &s->output_bytes))
    {why = "LPC channel buffer size overflow";goto failed;}
  s->input_rate = inrate; s->output_rate = outrate;
  {
    size_t target_window, analysis, mem = 0;
    double fit;
    if (!soxr_lpc_target_window(outrate, &target_window))
      {why = "LPC target-rate analysis window overflow";goto failed;}
    fit = (double)target_window * (inrate/outrate);
    if (!soxr_lpc_size_ceil(fit, &analysis))
      {why = "LPC input-rate analysis window overflow";goto failed;}
    s->prime_n = analysis < 2*SOXR_LPC_ORDER+1 ? 2*SOXR_LPC_ORDER+1 : analysis;
    if (!soxr_lpc_padding(inrate, outrate, target_window, &s->pre_n, &s->post_n))
      {why = "LPC target-rate padding length overflow";goto failed;}
    s->drop_front = s->post_n;
    /* Exactly one input history: the primer is transferred to the tail
     * when all primer frames have entered the raw resampler. Prefix and
     * suffix also share one buffer: they are never used concurrently.
     * Prediction work needs only the analysis window (32-frame LPC
     * recurrence is kept on the stack), not the 50-ms extrapolation. */
    if (!soxr_lpc_memory_add(s->prime_n, s->input_bytes, &mem) ||
       !soxr_lpc_memory_add(s->pre_n, s->input_bytes, &mem) ||
       !soxr_lpc_memory_add(SOXR_LPC_DISCARD_FRAMES, s->output_bytes, &mem) ||
       !soxr_lpc_memory_add(s->prime_n, sizeof (double), &mem))
      {why = "LPC extrapolation memory size overflow";goto failed;}
    if ((size_t)SOXR_LPC_MEMORY_LIMIT_MIB > SIZE_MAX / (1024u * 1024u))
      {why = "LPC memory budget size overflow";goto failed;}
    s->memory_limit_bytes = (size_t)SOXR_LPC_MEMORY_LIMIT_MIB * (size_t)(1024u*1024u);
    if (mem > s->memory_limit_bytes)
      {why = "LPC extrapolation memory budget exceeded";goto failed;}
    s->allocated_bytes = mem;
  }
  if (!soxr_lpc_product(s->prime_n, s->input_bytes, &n))
    {why = "LPC primer size overflow";goto failed;}
  if (!(s->prime = malloc(n))) {why = "malloc failed: LPC primer";goto failed;}
  if (!soxr_lpc_product(s->pre_n, s->input_bytes, &n))
    {why = "LPC extrapolation size overflow";goto failed;}
  if (!(s->prefix = malloc(n))) {why = "malloc failed: LPC prefix / suffix";goto failed;}
  if (!soxr_lpc_product(SOXR_LPC_DISCARD_FRAMES, s->output_bytes, &n))
    {why = "LPC output scratch size overflow";goto failed;}
  if (!(s->raw = malloc(n))) {why = "malloc failed: LPC prefix discard scratch";goto failed;}
  if (!soxr_lpc_product(s->prime_n, sizeof (double), &n))
    {why = "LPC workspace size overflow";goto failed;}
  if (!(s->work_mem = malloc(n))) {why = "malloc failed: LPC prediction workspace";goto failed;}
  p->lpc = s;
  return 0;
failed:
  soxr_lpc_free(s);
  return why?why:SOXR_LPC_ERROR_ALLOC;
}

static double soxr_lpc_read(void const *src, size_t i, int f64)
{
  return f64 ? ((double const*)src)[i] : ((float const*)src)[i];
}
static void soxr_lpc_write(void *dst, size_t i, double v, int f64)
{
  if (f64) ((double*)dst)[i] = v;
  else ((float*)dst)[i] = (float)v;
}

/* Continue the boundary value if an LPC model cannot be computed safely.
 * Unlike symmetric reflection, this does not introduce a reversed transient.
 * Zero input extends as zero, and constant input extends as a constant. */
static void soxr_lpc_edge_hold(struct soxr_lpc_state *s,
    unsigned char const *src, size_t frames, unsigned char *dst,
    size_t added, int backwards, size_t channel)
{
    size_t i;
    double edge = 0.0;

    if (frames) {

        size_t index = backwards ? 0 : frames - 1;
        edge = soxr_lpc_read(src, index * s->nch + channel, s->is_double_input);

        if (!soxr_lpc_isfinite(edge)) {

            edge = 0.0;
        }
    }

    for (i = 0; i < added; ++i) {

        soxr_lpc_write(dst, i * s->nch + channel, edge, s->is_double_input);
    }
}

static void soxr_lpc_predict(struct soxr_lpc_state *s,
    unsigned char const *src, size_t frames, unsigned char *dst,
    size_t added, int backwards)
{
    double *work = (double *)s->work_mem;
    size_t ch, i, j;
    unsigned order;

    if (!frames) {

        memset(dst, 0, added * s->input_bytes);
        return;
    }

    order = frames > SOXR_LPC_ORDER ? SOXR_LPC_ORDER : (unsigned)frames;

    for (ch = 0; ch < s->nch; ++ch) {

        double autoc[SOXR_LPC_ORDER + 1] = { 0 };
        double coeff[SOXR_LPC_ORDER] = { 0 };
        double peak = 0.0;
        double error, epsilon, damp, ceiling;
        double history[SOXR_LPC_ORDER];
        unsigned used = order;
        int scale_exponent = 0;
        int fallback = 0;

        for (i = 0; i < frames; ++i) {

            double v = soxr_lpc_read(src, i * s->nch + ch, s->is_double_input);

            if (!soxr_lpc_isfinite(v)) {

                fallback = 1;
                v = 0.0;
            }

            work[i] = v;

            if (fabs(v) > peak) {

                peak = fabs(v);
            }
        }

        if (frames < 2 * SOXR_LPC_ORDER + 1 || peak == 0.0) {

            fallback = 1;
        }

        if (!fallback) {

            /* Normalize fitting data by a binary exponent. Actual source
             * samples and predicted output remain in their original scale. */
            frexp(peak, &scale_exponent);

            for (i = 0; i < frames; ++i) {

                double center = ((double)frames + 1.0) * 0.5;
                double k = ((double)i + 1.0 - center) / center;
                double window = 1.0 - k * k;   /* Welch window */

                work[i] = ldexp(work[i], -scale_exponent) * window;
            }

            for (i = 0; i <= order; ++i) {

                for (j = i; j < frames; ++j) {

                    autoc[i] += work[j] * work[j - i];
                }
            }

            error = autoc[0] * (1.0 + 1e-10);
            epsilon = 1e-9 * autoc[0];
            /* The original wrapper used:
             *   epsilon = 1e-9 * raw_autoc[0] + 1e-10;
             * Preserve the absolute 1e-10 energy floor after binary-exponent
             * normalization of the LPC analysis signal. Since
             * raw_autoc = autoc * 2^(2*scale_exponent), the equivalent
             * normalized-domain threshold is 1e-10 * 2^(-2*scale_exponent).
             * An infinite threshold for an extremely small nonzero input
             * correctly selects the constant-edge fallback without ever
             * performing unstable Levinson-Durbin divisions. */
            epsilon += ldexp(1e-10, -2 * scale_exponent);

            for (i = 0; i < order; ++i) {

                double r;

                if (error <= epsilon || !(error > 0.0) || !soxr_lpc_isfinite(error)) {

                    used = (unsigned)i;
                    break;
                }

                r = -autoc[i + 1];

                for (j = 0; j < i; ++j) {

                    r -= coeff[j] * autoc[i - j];
                }

                r /= error;

                if (!soxr_lpc_isfinite(r) || fabs(r) >= 1.0) {

                    used = (unsigned)i;
                    break;
                }

                coeff[i] = r;

                for (j = 0; j < i / 2; ++j) {

                    double tmp = coeff[j];
                    coeff[j] += r * coeff[i - 1 - j];
                    coeff[i - 1 - j] += r * tmp;
                }

                if (i & 1) {

                    coeff[j] += coeff[j] * r;
                }

                error *= 1.0 - r * r;
            }

            if (used == 0) {

                fallback = 1;
            }

            damp = SOXR_LPC_GAMMA;

            for (j = 0; j < used; ++j) {

                coeff[j] *= damp;
                damp *= SOXR_LPC_GAMMA;
            }
        }

        if (!fallback) {

            ceiling = peak > DBL_MAX / 2.5 ? DBL_MAX : peak * 2.5;

            /* Only the last LPC_ORDER samples enter the recurrence.
             * The order of the floating-point sums is unchanged from
             * the reference implementation; no long extrapolation scratch
             * array is required. */
            for (j = 0; j < used; ++j) {

                history[j] = backwards
                    ? soxr_lpc_read(src, j * s->nch + ch, s->is_double_input)
                    : soxr_lpc_read(src, (frames - used + j) * s->nch + ch, s->is_double_input);
            }

            for (i = 0; i < added; ++i) {

                double v = 0.0;

                for (j = 0; j < used; ++j) {

                    v -= history[backwards ? used - 1 - j : j] * coeff[used - 1 - j];
                }

                if (!soxr_lpc_isfinite(v) || fabs(v) > ceiling) {

                    fallback = 1;
                    break;
                }

                soxr_lpc_write(dst, (backwards ? added - 1 - i : i) * s->nch + ch,
                    v, s->is_double_input);

                if (backwards) {

                    for (j = used - 1; j > 0; --j) {
                        history[j] = history[j - 1];
                    }
                    history[0] = v;
                } else {

                    for (j = 1; j < used; ++j) {
                        history[j - 1] = history[j];
                    }
                    history[used - 1] = v;
                }
            }
        }

        if (fallback) {

            soxr_lpc_edge_hold(s, src, frames, dst, added, backwards, ch);
        }
    }
}

static void soxr_lpc_push_tail(struct soxr_lpc_state *s,
                           void const *src, size_t frames)
{
  size_t rem;
  if (frames >= s->prime_n) {
    memcpy(s->tail, (unsigned char const*)src +
      (frames-s->prime_n)*s->input_bytes, s->prime_n*s->input_bytes);
    s->tail_used = s->prime_n; return;
  }
  rem = s->tail_used+frames;
  if (rem > s->prime_n) {
    size_t lost = rem-s->prime_n;
    memmove(s->tail, s->tail+lost*s->input_bytes, (s->tail_used-lost)*s->input_bytes);
    s->tail_used-= lost;
  }
  memcpy(s->tail+s->tail_used*s->input_bytes, src, frames*s->input_bytes);
  s->tail_used+= frames;
}

/* No LPC output queue is necessary in the normal path: the SoXR rate engine
 * already retains un-emitted output internally. This avoids recopying the
 * entire converted stream. The fixed scratch buffer is used only to discard
 * the synthetic prefix and (after final duration is reached) the suffix.
 *
 * On undersized output buffers, limit input consumption instead of allowing
 * an unbounded internal backlog. No LPC heap allocations occur in process. */
static size_t soxr_lpc_allowed_frames(struct soxr_lpc_state const *s, size_t real_frames)
{
  double count = (double)real_frames * s->output_rate / s->input_rate;
  double integral = s->eof_requested ? floor(count + 0.5) : floor(count);
  if (!soxr_lpc_isfinite(integral) || integral <= 0.) return 0;
  return integral >= (double)SIZE_MAX ? SIZE_MAX : (size_t)integral;
}

static size_t soxr_lpc_allowed_output(struct soxr_lpc_state const *s)
{
  return soxr_lpc_allowed_frames(s, s->real_in);
}

static size_t soxr_lpc_min(size_t a, size_t b)
{
  return a < b ? a : b;
}

/* Supply source frames to the raw resampler. With a sufficiently large caller
 * buffer, no intermediate output copies or fixed-size processing chunks exist.
 * Prefix trimming and the final excess tail use a small fixed scratch area.
 * Passing NULL for raw idone makes it accept the complete selected source
 * segment rather than restricting input using soxr_i_for_o(). */
static soxr_error_t soxr_lpc_transfer(soxr_t p, void const *source, size_t frames,
  size_t permitted, unsigned char *out, size_t out_frames,
  size_t *used, size_t *written)
{
  struct soxr_lpc_state *s = p->lpc;
  unsigned char *destination;
  size_t capacity;
  size_t produced = 0;
  soxr_error_t err;

  if (s->drop_front) {
    destination = s->raw;
    capacity = soxr_lpc_min(s->drop_front, (size_t)SOXR_LPC_DISCARD_FRAMES);
  } else if (s->total_out >= permitted) {
    destination = s->raw;
    /* The synthetic suffix beyond the exact output duration may be discarded
     * only when the end-of-stream is known. Earlier output remains buffered
     * inside the underlying rate engine until more real input arrives. */
    capacity = s->eof_requested ? SOXR_LPC_DISCARD_FRAMES : 0;
  } else {
    destination = out;
    capacity = soxr_lpc_min(out_frames, permitted - s->total_out);
  }

  /* A non-null input pointer with a zero length drains buffered output
   * without setting the raw resampler's flushing state. */
  err = soxr_process_raw(p, source, frames, NULL, destination, capacity, &produced);
  if (err) return err;

  *used = frames;
  *written = 0;
  if (s->drop_front) {
    s->drop_front -= produced;
  } else if (s->total_out < permitted) {
    s->total_out += produced;
    *written = produced;
  }
  return 0;
}

static soxr_error_t soxr_lpc_process(soxr_t p, void const *in, size_t in_frames,
  size_t *in_done, void *out, size_t out_frames, size_t *out_done)
{
  struct soxr_lpc_state *s = p->lpc;
  size_t consumed = 0, emitted = 0, iterations = 0;
  int final = !in;
  soxr_error_t err = 0;
  unsigned char zero = 0;

  if (in && (ptrdiff_t)in_frames < 0) { final = 1; in_frames = ~in_frames; }
  if (!out && out_frames) return "null output buffer pointer";
  if (in_frames && in_frames > SIZE_MAX / s->input_bytes)
    return "LPC input buffer byte size overflow";
  if (out_frames && out_frames > SIZE_MAX / s->output_bytes)
    return "LPC output buffer byte size overflow";
  if (in && in_frames && !in_done)
    return "LPC streaming requires idone to prevent lost input samples";
  if (in_done) *in_done = 0;
  if (out_done) *out_done = 0;
  if (s->phase == 6) return in_frames ? "input after LPC stream ended" : 0;
  if (s->eof_requested && in_frames) return "input after LPC end-of-input";
  if (final) s->eof_requested = 1;

  while (++iterations < 1000000) {
    size_t n = 0, used = 0, created = 0, allowed;
    void const *source = &zero;
    int input_external = 0;

    if (s->phase == 0) {
      n = s->prime_n - s->prime_used;
      n = soxr_lpc_min(n, in_frames - consumed);
      if (n) {
        if (n > SIZE_MAX - s->real_in) { err = "LPC input frame count overflow"; break; }
        memcpy(s->prime + s->prime_used * s->input_bytes,
          (unsigned char const *)in + consumed * s->input_bytes,
          n * s->input_bytes);
        s->prime_used += n;
        consumed += n;
        s->real_in += n;
      }
      if (s->prime_used < s->prime_n && !s->eof_requested) break;
      if (!s->prime_used && s->eof_requested) { s->phase = 6; break; }
      soxr_lpc_predict(s, s->prime, s->prime_used, s->prefix, s->pre_n, 1);
      s->phase = 1;
      continue;
    }

    if (s->phase == 1) {
      n = s->pre_n - s->pre_pos;
      if (!n) { s->phase = 2; continue; }
      source = s->prefix + s->pre_pos * s->input_bytes;
    } else if (s->phase == 2) {
      n = s->prime_used - s->prime_pos;
      if (!n) {
        s->tail = s->prime;
        s->prime = NULL;
        s->tail_used = s->prime_used;
        s->phase = 3;
        continue;
      }
      source = s->prime + s->prime_pos * s->input_bytes;
    } else if (s->phase == 3) {
      n = in_frames - consumed;
      if (n) {
        source = (unsigned char const *)in + consumed * s->input_bytes;
        input_external = 1;
      } else if (s->eof_requested) { s->phase = 4; continue; }
      /* With zero input, drain any real output buffered inside SoXR. */
    } else if (s->phase == 4) {
      if (!s->suffix_ready) {
        soxr_lpc_predict(s, s->tail, s->tail_used, s->prefix, s->pre_n, 0);
        s->suffix_ready = 1;
      }
      n = s->pre_n - s->post_pos;
      if (!n) { s->phase = 5; continue; }
      source = s->prefix + s->post_pos * s->input_bytes;
    } else if (s->phase == 5) {
      /* Flush the resampler after all synthetic suffix frames were sent. */
      size_t cap = 0;
      unsigned char *dest = s->raw;
      allowed = soxr_lpc_allowed_output(s);
      if (s->drop_front) cap = soxr_lpc_min(s->drop_front, (size_t)SOXR_LPC_DISCARD_FRAMES);
      else if (s->total_out < allowed) {
        cap = soxr_lpc_min(out_frames - emitted, allowed - s->total_out);
        dest = out ? (unsigned char *)out + emitted * s->output_bytes : NULL;
      } else cap = SOXR_LPC_DISCARD_FRAMES;
      if (!cap) break;
      err = soxr_process_raw(p, NULL, 0, &used, dest, cap, &created);
      if (err) break;
      if (s->drop_front) s->drop_front -= created;
      else if (s->total_out < allowed) { s->total_out += created; emitted += created; }
      if (!created) {
        if (s->total_out < allowed) err = "LPC final output tail unexpectedly short";
        else s->phase = 6;
        break;
      }
      continue;
    } else break;

    if (s->phase == 3 && !n && !s->eof_requested &&
        (emitted == out_frames || s->total_out >= soxr_lpc_allowed_output(s))) break;

    if (input_external && n > SIZE_MAX - s->real_in) {
      err = "LPC input frame count overflow";
      break;
    }

    if (s->phase == 3 && input_external) {
      /* With a large enough output buffer, consume the entire input block.
       * For small output buffers, bound the added input to the available
       * output budget, preserving the ordinary partial-idone contract. */
      size_t free_out = out_frames - emitted;
      double ratio = s->input_rate / s->output_rate;
      double max_frames = (double)free_out * ratio;
      size_t budget = max_frames >= (double)SIZE_MAX ? SIZE_MAX :
        max_frames >= 0. ? (size_t)max_frames : 0;
      size_t needed = soxr_lpc_allowed_frames(s, s->real_in + n);
      if (needed > s->total_out + free_out) {
        if (!free_out) break;
        if (budget < 1) budget = 1;
        if (budget < n) n = budget;
      }
    }

    allowed = soxr_lpc_allowed_frames(s, s->real_in + (input_external ? n : 0));
    err = soxr_lpc_transfer(p, source, n, allowed,
      out ? (unsigned char *)out + emitted * s->output_bytes : NULL,
      out_frames - emitted, &used, &created);
    if (err) break;
    emitted += created;
    if (s->phase == 1) s->pre_pos += used;
    else if (s->phase == 2) s->prime_pos += used;
    else if (s->phase == 4) s->post_pos += used;
    else if (input_external && used) {
      soxr_lpc_push_tail(s, source, used);
      consumed += used;
      s->real_in += used;
    }
    if (!used && !created) break;
    if (s->phase == 3 && consumed == in_frames && !s->eof_requested) {
      /* A zero-input output pass is needed when input was buffered
       * internally rather than fully drained in the previous call. */
      continue;
    }
  }

  if (in_done) *in_done = consumed;
  if (out_done) *out_done = emitted;
  return err;
}
#endif
