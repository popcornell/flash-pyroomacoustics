# pyroomacoustics, Rust core (fast)

This is a drop-in replacement for [pyroomacoustics](https://github.com/LCAV/pyroomacoustics):
same package name, same API, statistically equivalent output, and several times faster.
The C++/pybind11 extension `pyroomacoustics.libroom` is re-implemented in Rust (PyO3 + rayon).
The hot paths of the Python layer (ISM RIR synthesis, octave filterbank, ray-tracing tail
synthesis) run in compiled code too.

## Defaults (fast mode)

* **Image source method.** RIRs are the same as the original up to float32 rounding
  (relative L2 error ~1e-4). RT60, EDT, C50 and DRR are identical.
* **Ray tracing with scattering.** Every ray has its own random generator, so rays are traced
  in parallel. The output is statistically equivalent to the original, which is itself
  random. Over 4 seeds × 24 rooms, every acoustic parameter differs from the reference by
  less than the reference's own seed-to-seed variation (`bench/validate.py`).
* **Threads.** The compiled kernels and SciPy FFTs use at most `num_threads` threads. The
  default is **8**, capped by the CPUs the process may use (`taskset`/cgroups). Override it
  with `PRA_NUM_THREADS=N`, `OMP_NUM_THREADS=N` or `pra.constants.set("num_threads", N)`.
  The original's `rir_builder` started `cpu_count()` threads on every call.

### Exact mode

`pra.constants.set("exact_mode", True)` gives results bit-identical to the original C++
build. It restores the per-ray global RNG stream, the original ISM synthesis path and the
per-band filterbank. It is still faster than the original (see "How exactness is achieved").

## Install

Requirements: a Rust toolchain (`rustup`) and Python ≥ 3.9.

```bash
pip install maturin
maturin develop --release        # or: maturin build --release && pip install target/wheels/*.whl
```

## How exactness is achieved (exact mode)

Floating-point addition is not associative, so "same output" means reproducing the exact
order of every operation of the original compiled code (GCC 9, `-O3`, SSE2, no FMA, Eigen 5.0.1):

* **Eigen evaluation order** (`src/eig.rs`). Fixed-size norms and products are summed as
  binary trees, fixed-size dots sequentially, and dynamic-size reductions with Eigen's 4-wide
  packet traversal (two accumulators, `predux = (x0+x2)+(x1+x3)`, then a sequential tail).
  These orders were determined from the Eigen sources and verified with C++ probes.
* **`JacobiSVD`.** The wall basis and normal come from Eigen's `JacobiSVD` (column-pivoting
  Householder QR preconditioner + 2×2 Jacobi sweeps). It is ported operation by operation,
  including its GEMV kernel and its tie-breaking.
* **RNG.** `std::mt19937_64` and libstdc++'s `generate_canonical<float>` /
  `uniform_real_distribution<float>` are re-implemented bit-exactly, so
  `pra.random.seed(libroom=...)`, Lambertian scattering and `Room.contains()` consume the
  same random stream.
* **Threaded RIR builders.** `rir_builder` and `delay_sum` keep the original
  `num_threads`-block partition and summation order, because the float result depends on
  it. They use a persistent thread pool and only the sample ranges each block touches.
* **Ray tracing.** Histogram sums are accumulated in the original serial order. Rays never
  touch the RNG when no wall scatters, so those rays are traced in parallel and their
  events are merged in order. With scattering, the RNG-dependent trajectories are computed
  sequentially, while the expensive part (microphone visibility, energies, histogram events)
  runs in parallel and is merged in order.
* **`rt.py` hot loops.** `poisson_sequence` draws its uniforms in a batch, runs the
  recurrence in Rust, then rewinds the NumPy generator and advances it by exactly the number
  of draws used. `directional_sequence`'s accumulation loop moved to Rust with identical
  operations.
* **Math library.** Transcendental functions (`sinf`, `cosf`, `atan2f`, `sin`, ...) call the
  same system libm as the original.

### Verification

* `equiv/` is a differential harness: 86 scenarios and about 13.5k compared arrays. It covers
  ISM 2D/3D (all orders, octave bands, air absorption, min-phase, randomized ISM), hybrid and
  pure ray tracing with and without scattering, non-convex polygonal rooms, directivities,
  `simulate()`, low-level geometry fuzzing (thousands of random walls and segments), and the
  RIR builders for float32/float64 at several thread counts. Every output is compared
  bitwise against the original C++ build.
* The original test suite gives the same set of passing and failing tests (the failures are
  environmental, e.g. missing SOFA data files).

### Known, intentional differences

1. **2D ray tracing with scattering.** The original never initializes `Wall2D.basis` and
   samples scattered directions from that uninitialized memory. Its results are undefined:
   they change from run to run, and scattered rays all go along ±x. Here the 2D basis is the
   wall tangent, so 2D scattering is deterministic.
2. **Pybind11 internals.** These are not reproduced:
   * the auto-generated `TypeError` message text (the exception *type* is the same);
   * `get_wall(i)` returning the live object when a Python wrapper already exists;
   * arrays' `.base` pointing to the engine object.
3. **Undefined behaviour.** Where the original reads uninitialized memory (e.g. the point
   returned by `next_wall_hit` when nothing is hit) we return zeros. Where it would loop
   forever or crash on invalid input, we raise.

## Benchmarks

Both builds limited to 8 threads (`PRA_NUM_THREADS=8`), AMD EPYC 7742, best of 3,
`bench/bench.py`:

| workload | original (C++) | this | speed-up |
|---|---:|---:|---:|
| ShoeBox ISM, order 17, 1 band, 4 mics | 0.011 s | 0.007 s | 1.6× |
| ShoeBox ISM, order 17, octave bands, 4 mics | 0.067 s | 0.014 s | 4.8× |
| Hybrid ISM(3)+RT, 10k rays, no scattering | 0.482 s | 0.031 s | 15× |
| Hybrid ISM(3)+RT, 10k rays, scattering 0.1 | 4.312 s | 0.380 s | 11× |
| L-shaped room, hybrid with scattering | 0.197 s | 0.019 s | 10× |
| 20 random `inverse_sabine` rooms (dataset generation) | 4.742 s | 0.902 s | 5.3× |
| 10 rooms × 4 mics, RT60 0.2–1 s, full-order ISM (`bench/dataset_prof.py`) | 7.48 s | 1.74 s | 4.3× |
| 10 rooms × 4 mics, hybrid with scattering + air absorption | 5.96 s | 0.38 s | 16× |

## Aligned anechoic (direct-path) RIRs

For training targets (dereverberation, separation, ...) every room can produce the
direct-path RIR of each source/microphone pair, aligned at the sample level with the
reverberant RIR. Both the CPU (Rust) and the CUDA backend support it.

```python
room.compute_rir()
room.compute_anechoic_rir()        # room.anechoic_rir[m][s]

# or directly the signals: same shape (n_src, n_mics, n_samples), same time origin
premix, anechoic = room.simulate(return_premix=True, return_anechoic=True)
```

`anechoic_rir[m][s]` is the order-0 component of `rir[m][s]`: same time origin (sample
`n` is the same instant in both arrays), same fractional-delay filter, `1/d` attenuation,
air absorption, directivities, octave filter bank and high-pass filter. With the
randomized ISM it follows the displaced direct-path image. `simulate(return_anechoic=True)`
applies the source delays and, with `snr=...`, the same normalization as `premix`.
By default the free-field direct path is used even when it is occluded (non-convex rooms).
`compute_anechoic_rir(visible_only=True)` gives zeros there, as in `rir`.
`air_absorption=False` drops the air absorption.

`bench/anechoic_test.py` checks the alignment: with the high-pass filter off, `rir` and
`anechoic_rir` agree to ~1e-7 (single band) or ~1e-3 (octave bands) up to the first
reflection. The cross-correlation of `premix` and `anechoic` peaks at lag 0. The default
10 Hz high-pass filter is zero-phase, so it smears the low frequencies of the reflections
slightly before the direct path. The direct sound itself does not move.

### Direct path + early reflections target

```python
room.compute_anechoic_rir(early_ms=50)    # taper_ms=5 by default
premix, early = room.simulate(return_premix=True, return_anechoic=True, early_ms=50)
```

With `early_ms > 0` the target is the actual RIR `rir[m][s]`, kept up to `early_ms` after
the direct-path arrival at each microphone and faded out with a half cosine over
`taper_ms`. The time origin is the same, so `rir == early + late` sample by sample. This
target (≈50 ms) leaves the early reflections alone, which do not hurt intelligibility, and
removes only the late reverberation. The simulator is least realistic in exactly those
shoebox early reflections, so it is often the better target for training dereverberation
models. `early_ms=0` (the default) gives the direct path.

### Dropping the impulse at the direct sound (`rt_tail_t0_impulse`)

In hybrid simulations the original tail always starts with an impulse of random sign
exactly at the direct-path sample, so the reverberant RIR's direct sound gets a random
gain. In the cases checked here this was up to ±30–40% of its amplitude.
`pra.constants.set("rt_tail_t0_impulse", False)` (or `PRA_RT_TAIL_T0_IMPULSE=0`) drops it.
It works on CPU and GPU, with both tail types, and changes the RIR energy by <0.5%. It has
no effect in `exact_mode`. Some random tail energy still overlaps the direct sound: the
tail starts at the direct-path time and the octave filters spread it.

## CUDA backend (PyTorch + Triton)

```python
pra.constants.set("backend", "cuda")         # or PRA_BACKEND=cuda
pra.constants.set("cuda_device", "cuda:1")   # or PRA_CUDA_DEVICE=cuda:1
room.compute_rir()                            # or room.compute_rir(backend="cuda")

# dataset generation: ray tracing of all the rooms in a single GPU launch
pra.cuda.compute_rirs(rooms, anechoic=True)
```

Needs `torch` with CUDA (`pip install .[cuda]`; Triton ships with it). On the GPU:

* **Image source model of shoebox rooms.** Image generation, attenuations, air
  absorption, windowed-sinc synthesis and octave filter bank. It matches the CPU to
  float32 rounding (relative L2 error ≤ 6e-5). Below `cuda_min_ism_work` (image sources ×
  mics, default 40000, about order 15 with 4 mics) the CPU is faster and is used
  automatically.
* **Ray tracing of 3D shoebox rooms.** A Triton kernel with one GPU thread per ray. It
  covers specular reflections, absorption, Lambertian scattering with diffuse rain, the
  hybrid ISM/RT split and the energy/time thresholds, with the same rules as the CPU
  engine. Source directivities are supported.
* **Ray-tracing tail synthesis.** The random impulse sequence, the per-histogram-bin
  normalization, the envelope interpolation, the octave bands and the air absorption,
  batched over all source/mic pairs.
* **Anechoic (direct-path) RIRs** of any room.

The ray tracing output is statistically equivalent to the CPU (and the original), not
sample-identical:

* Without scattering, the histograms equal the CPU ones (float32 atomics).
* With scattering, the energy totals match the CPU within ±0.7% (noise level).
* Paired validation over 4 seeds × 24 rooms (`bench/validate.py`, `PRA_BACKEND=cuda`):
  every T20/EDT/C50/DRR/energy difference to the original is within 1.6 standard errors.
  The differences are far below the original's own seed-to-seed variation.

Falls back to the CPU, with a one-time warning:

* polygonal rooms;
* 2D ray tracing;
* directional microphones;
* directional sources (ISM part only);
* `exact_mode`.

With the CUDA backend, `source.images`/`source.damping` are not filled when the ISM runs
on the GPU.

Also on the GPU with the CUDA backend:

* **RIR high-pass filter.** The default 10 Hz filter is applied as `|H(f)|²` in the
  frequency domain instead of `sosfiltfilt`. It agrees to ≤1e-4 (relative L2) on RIRs
  of 0.2 s or more. Shorter RIRs differ by up to ~1%, and the short anechoic RIRs by ~1.6%,
  both from `sosfiltfilt`'s edge handling: low-order ISM RIRs end abruptly instead of
  decaying. The GPU version is a plain linear time-invariant filter, so the filtered
  anechoic RIR is exactly the direct-path part of the filtered full RIR.
* **`simulate()` convolutions**, including `return_anechoic`: float64 FFT convolution,
  equal to `fftconvolve` to 1e-15.
* **Tail synthesis is batched over all rooms** in `pra.cuda.compute_rirs`.

**Opt-in Gaussian tail:** `pra.constants.set("rt_tail", "gaussian")` or
`PRA_RT_TAIL=gaussian`.

* It replaces the random sequence of signed impulses with white Gaussian noise. The
  noise has the same expected local energy, computed in closed form from the impulse rate
  and the per-bin normalization; the impulse at t0 is kept.
* It removes the per-pair sequence generation and the very sparse, randomly missing
  tails of close mics.
* Over 4 seeds × 24 rooms, T20, EDT and C50 agree with the original within 1.6 SE. In the
  octave-band rooms, energy is +0.08 dB (2.6 SE) and DRR −0.21 dB (2.3 SE). Both are about a
  third of the original's own seed-to-seed variation. The likely cause is the original
  sequence's lagging rate update during the first ~45 ms.
* It saves only ~15% of the time, so the default stays `"poisson"`.

AMD EPYC 7742 limited to 8 threads vs one A100, `bench/gpu_bench.py`, 4 mics per room
(`cuda` = `pra.cuda.compute_rirs`, then `room.simulate`):

| workload | CPU (8 threads) | CUDA | CUDA, Gaussian tail |
|---|---:|---:|---:|
| 10 rooms, hybrid, default n_rays (~2.5k), RIRs only | 0.356 s | 0.053 s | 0.046 s |
| 10 rooms, hybrid, 100k rays, RIRs only | 3.05 s | 0.119 s | 0.099 s |
| 50 rooms, hybrid, default n_rays, RIRs only | 1.48 s | 0.237 s | 0.181 s |
| 10 rooms, full-order ISM, RIRs only | 1.76 s | 0.409 s | 0.381 s |
| 10 rooms, hybrid, RIRs + anechoic + `simulate` (10 s signal) | 0.760 s | 0.120 s | 0.103 s |
| 10 rooms, full-order ISM, RIRs + anechoic + `simulate` (10 s) | 2.25 s | 0.440 s | 0.439 s |

### Quirks of the original found along the way (kept as is)

* **Tail impulse at the direct sound.** The random tail of hybrid RIRs starts with an
  impulse of random sign exactly at the direct-path sample. The anechoic RIR does not
  contain it.
* **Missing tail for very close mics.** When a mic is very close to the source (here 8 cm),
  the tail can be missing or start very late. The Poisson rate is evaluated at `2·t0`, so
  the second impulse comes after 30 ms to 2 s.
* **2D shoebox ray tracing uses the wrong walls.** The engine stores the 2D walls in the
  order south, east, north, west. Its shoebox hit test indexes them as west, east, south,
  north, so three of the four sides use the wrong normal and coefficients. The rays leave
  the room: in a 2D room with absorption 0.01 the original's ray-tracing energy stops after
  24 ms, whereas a 3D room keeps it up to the 1 s time limit. This is why GPU ray tracing is
  3D only.
