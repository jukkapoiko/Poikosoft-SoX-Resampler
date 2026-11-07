# Poikosoft SoX Resampler v14

**A precision-focused, SIMD-optimized version of the SoX Resampler Library (`libsoxr`) for high-quality PCM sample-rate conversion.**

Developed and maintained by [Poikosoft](https://www.poikosoft.com), with substantial enhancements by Jukka Poikolainen. This project is based on the original open-source SoX Resampler; it is **not** a resampler written from scratch.

## What it does

Poikosoft SoX Resampler converts PCM audio between sample rates using the established SoX resampling architecture. It supports common integer and floating-point PCM formats, interleaved or planar channel buffers, and streaming conversion through the `soxr.h` C API.

The v14 branch concentrates on **numerical accuracy, filter quality, predictable memory use, and efficient single-threaded processing**. It preserves the original SoX linear-phase resampling filter design and multistage sample-rate conversion approach while extending the implementation around them.

**Versioning note:** `v14` is the Poikosoft release designation. The inherited `soxr_version()` API and `SOXR_THIS_VERSION` macros retain upstream libsoxr version `0.1.3` for compatibility.

## How v14 differs from the original SoX Resampler

### Higher numerical precision

- Improved floating-point calculations in filter design and the resampling pipeline
- More accurate FIR coefficient preparation and polyphase interpolation
- A high-precision sample clock for improved handling of non-integer conversion ratios
- Fast processing paths for both 32-bit and 64-bit floating-point audio
- An additional **EXTREME double-double precision** engine, with dedicated double-double arithmetic, FFT processing, and FIR operations
- Extended quality recipes with filter rejection *design targets* up to 64-bit quality (`SOXR_64_BITQ`)

**Important:** A 56-bit or 64-bit `BITQ` setting describes a filter-rejection design target. It does **not** promise 56 or 64 bits of accurate output audio, and it is distinct from IEEE floating-point word length.

### Optimized CPU processing

- New or substantially reworked SSE/SSE2, AVX and AVX2 processing paths
- Optimized FFT backends, FIR filtering and polyphase interpolation
- Runtime CPU/OS feature detection, with the best supported processing backend selected during initialization
- Scalar Ooura FFT paths and SIMD PFFFT paths for Fast Float32/Float64
- A dedicated double-double FFT implementation for EXTREME, including scalar and SIMD implementations
- AVX paths that do not require AVX2 or FMA3; AVX2 paths require FMA3 where used
- Single-threaded operation by design, without an OpenMP dependency

The Fast and EXTREME engines have independent SIMD configuration selectors in `soxr-config.h`: `USE_SIMD` and `EXTREME_USE_SIMD`. The default for both is automatic selection.

### More configuration choices

- Added quality/rejection targets: 16, 20, 24, 28, 32, 40, 56 and 64 `BITQ`
- Added steepness recipes covering 99%, 99.5%, 99.55%, 99.6%, 99.65%, 99.7%, 99.75% and 99.8%
- Configurable passband and rolloff behavior, with the original SoX linear-phase filter architecture retained
- A choice between the Fast engine and the separate EXTREME double-double engine
- The existing `soxr.h` streaming API and common SoXR data formats retained for integration

This branch is **linear-phase only**. Other phase-response modes from upstream libsoxr are not a supported feature of this modified release.

### Streaming, audio boundaries and robustness

- Optional automatic **LPC (linear predictive coding) extrapolation** at audio boundaries, intended to improve short-signal and start/end handling
- Preallocated per-instance working buffers and revised memory/error handling
- Multistage conversion preserved, with attention to accurate sample accounting and gapless operation
- No application-level threading required to use a resampler instance

The optional LPC feature is controlled by `SOXR_AUTO_LPC_EXTRAPOLATION` in `soxr-config.h` (enabled in the supplied configuration).

## Quality modes

| Mode | Purpose |
| --- | --- |
| **Fast Float32** | Efficient standard-precision processing |
| **Fast Float64** | Higher-accuracy floating-point processing, with selectable filter-rejection targets |
| **EXTREME** | Double-double internal arithmetic and FFT/FIR processing for maximum numerical precision |

These are internal precision/processing choices, not claims that audible quality will improve in every application. Actual quality and speed depend on the input/output rates, filter settings, CPU and signal.

## Basic integration

The public API is declared in [`soxr.h`](soxr.h). For example, create a stereo 96 kHz → 44.1 kHz resampler with a 64-bit filter-rejection target and the EXTREME engine:

```c
#include "soxr.h"

soxr_error_t error = 0;
soxr_io_spec_t io = soxr_io_spec(SOXR_FLOAT64_I, SOXR_FLOAT64_I);
soxr_quality_spec_t quality = soxr_quality_spec(
    SOXR_64_BITQ | SOXR_STEEP_997,
    SOXR_EXTREME);

soxr_t resampler = soxr_create(
    96000.0, 44100.0, 2, &error, &io, &quality, 0);

if (!resampler) {
    /* Handle the error string returned in error. */
}

/* Process audio with soxr_process(), then flush with a NULL input. */
/* Release resources with soxr_delete(resampler). */
```

When streaming, honor **both** `idone` and `odone` from `soxr_process()`; a call may consume only part of the supplied input or produce buffered output. At end of input, call `soxr_process()` with `in == NULL` and continue draining output as appropriate. See the header for the full API contract.

## Building

This repository includes a CMake project for the resampler library. For example:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON
cmake --build build --config Release
```

The supplied build compiles as C11 and keeps architecture-specific SIMD implementations in separate source files. On Windows, MSVC and compatible toolchains can build the project; the baseline library must not be compiled globally with AVX2-only instructions if it is expected to run on older CPUs. Supported compilation and runtime details are in [`CMakeLists.txt`](CMakeLists.txt) and [`soxr-config.h`](soxr-config.h).

This repository provides the resampling library source, not the full original SoX distribution or an audio file encoder/decoder.

## Original project, modifications and acknowledgements

**Original:** [SoX Resampler Library (`libsoxr`)](https://github.com/chirlu/soxr) — copyright © 2007–2018 `robs@users.sourceforge.net` and other upstream contributors.

**Poikosoft modifications:** © 2026 Jukka Poikolainen / [Poikosoft](https://www.poikosoft.com). These include the precision improvements, EXTREME double-double engine, new and optimized SIMD pathways, additional configuration options, and LPC boundary integration described above.

The code also contains third-party components with their own preserved notices, including **PFFFT** (Julien Pommier / FFTPACK lineage), **Ooura FFT** (Takuya Ooura), and LPC-derived material credited to **lvqcl**. Consult the copyright and license notices in the respective source files. The original authors retain ownership of their original contributions.

## License

The SoX Resampler-derived library is distributed under the **GNU Lesser General Public License, version 2.1 or (at your option) any later version** (`LGPL-2.1-or-later`). The license text is included in [`LICENSE`](LICENSE).

Poikosoft's modifications to this LGPL-covered library are distributed here under the same LGPL terms. Existing third-party copyright and permission notices remain applicable to their respective files. You may use, modify and redistribute this library subject to those terms; distributing a modified binary also carries the applicable LGPL source-code and relinking obligations.

This project is independently maintained by Poikosoft and is **not an official release of the upstream SoX Resampler project**.
