# CUDA backend (PyTorch + Triton)

```python
pra.constants.set("backend", "cuda")         # or PRA_BACKEND=cuda
pra.constants.set("cuda_device", "cuda:1")   # or PRA_CUDA_DEVICE=cuda:1
room.compute_rir()                            # or room.compute_rir(backend="cuda")

# dataset generation: ray tracing of all the rooms in a single GPU launch
pra.cuda.compute_rirs(rooms, anechoic=True)
```

It needs `torch` with CUDA (`pip install "fast-pyroomacoustics[cuda]"`). Triton ships with
it. `pra.cuda.is_available()` tells whether the backend can be used.

## What runs on the GPU

| stage | module | scope |
|---|---|---|
| Image source model | `cuda.py` (`ism_rirs`) | shoebox rooms, 2D and 3D, omnidirectional sources and mics. Image generation, wall attenuations, air absorption, windowed-sinc synthesis, octave filter bank. |
| Ray tracing | `_rt_triton.py`, `cuda_rt.py` (`ray_tracing`) | 3D shoebox rooms, omnidirectional mics, any source directivity. One GPU thread per ray. |
| Tail synthesis | `cuda_rt.py` (`rt_tails`, `rt_tails_batch`) | Random impulse sequence, per-bin normalization, envelope interpolation, octave bands, air absorption, batched over all source/mic pairs (and rooms). |
| Anechoic RIRs | `cuda.py` (`direct_rirs`) | any room geometry, omnidirectional sources and mics. |
| High-pass filter | `cuda.py` (`highpass`) | all RIRs computed with the CUDA backend. |
| `simulate()` convolutions | `cuda.py` (`premix`) | float64 FFT convolution, including `return_anechoic`. Equal to `fftconvolve` to 1e-15. |

### Image source model

It matches the CPU to float32 rounding (relative L2 error ≤ 6e-5). Below
`cuda_min_ism_work` (image sources × mics, default 40000, about order 15 with 4 mics) the
CPU is faster, and the CPU is used automatically. With the CUDA backend
`source.images`/`source.damping` are not filled when the ISM runs on the GPU.

### Ray tracing

The Triton kernel follows the same rules as the CPU engine:

* specular reflections and absorption;
* Lambertian scattering with diffuse rain towards the microphones;
* the hybrid ISM/RT split: specular paths of order ≤ `max_order` are left to the ISM;
* the energy and time thresholds.

It processes `BLOCK` rays per program and accumulates the histograms with atomics.

The output is statistically equivalent to the CPU (and the original), not identical:

* **Without scattering**, the histograms equal the CPU ones up to float32 atomics.
* **With scattering**, the energy totals match the CPU within ±0.7%, which is the
  noise level.
* **Paired validation** over 4 seeds × 24 rooms, against the original: every
  T20/EDT/C50/DRR/energy difference is within 2.3 standard errors and far below the
  original's own seed-to-seed variation. See [validation.md](validation.md).

### High-pass filter

The default 10 Hz zero-phase high-pass is applied as `|H(f)|²` in the frequency domain,
instead of `scipy.signal.sosfiltfilt`.

* It agrees with `sosfiltfilt` to ≤ 1e-4 (relative L2) on RIRs of 0.2 s or more.
* Shorter RIRs differ by up to ~1%, and the short anechoic RIRs by ~1.6%. The difference
  comes from `sosfiltfilt`'s edge handling: low-order ISM RIRs end abruptly instead of
  decaying.
* The GPU version is a plain linear time-invariant filter, so the filtered anechoic RIR is
  exactly the direct-path part of the filtered full RIR.

## Fallbacks

These cases run on the CPU, with a one-time warning per reason:

| case | ISM | ray tracing |
|---|---|---|
| polygonal (non-shoebox) rooms | CPU | CPU |
| 2D rooms | GPU | CPU |
| directional microphones | CPU | CPU |
| directional sources | CPU (also the anechoic RIRs) | GPU |
| `exact_mode=True` | CPU | CPU |
| small ISM (< `cuda_min_ism_work`) | CPU, no warning | – |
| no PyTorch/CUDA/Triton | CPU | CPU |

GPU ray tracing is 3D only on purpose: the original's 2D shoebox ray tracing is broken (see
[quirks.md](quirks.md#2d-shoebox-ray-tracing-uses-the-wrong-walls)), so there is nothing
correct to match.

## Gaussian tail

```python
pra.constants.set("rt_tail", "gaussian")     # or PRA_RT_TAIL=gaussian
```

This is an opt-in approximation for the CUDA backend.

**What it does.** It replaces the random sequence of signed impulses with white Gaussian
noise of the same expected local energy. The energy is computed in closed form from the
impulse rate λ = min(4πc³(t0+t)²/V, 10⁴) (evaluated at t0+t, as in the original) and the per-bin normalization of the original. The
impulse at t0 is kept, unless `rt_tail_t0_impulse=False`.

**What it removes.** The per-pair sequence generation, and the very sparse, randomly
missing tails of close mics.

**Accuracy.** Over 4 seeds × 24 rooms:

* T20 and EDT agree with the original within 1.7 SE.
* Octave-band rooms: energy is +0.08 dB (2.6 SE) and DRR −0.21 dB (2.3 SE). Both are a
  third or less of the original's own seed-to-seed variation. The likely cause is that the
  original sequence updates its rate with a lag during the first ~45 ms.
* Single-band rooms: the C50 and DRR differences (−1.1 dB, −0.4 dB) come from one pair with
  the mic 8 cm from the source. There the original's tail is often missing (see
  [quirks.md](quirks.md)), while the Gaussian tail is always present.

**Speed.** It saves only about 15% of the time, so the default stays `"poisson"`.

## Benchmarks

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

Hybrid rooms: `inverse_sabine` RT60 0.2–1 s, ISM order 3, scattering 0.1, air absorption.
The first call of a process compiles the Triton kernel; the timings exclude it.
