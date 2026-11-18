# Architecture

```
Cargo.toml, pyproject.toml   maturin build: Rust crate -> pyroomacoustics.libroom
src/                         Rust extension module (PyO3)
python/pyroomacoustics/      Python package (upstream code + changes below)
bench/                       benchmarks and statistical validation
equiv/                       bitwise differential harness vs the original build
docs/                        this documentation
```

## Rust extension (`src/`)

`pyroomacoustics.libroom` replaces the original pybind11/C++ extension with the same
classes, functions and signatures.

| file | content |
|---|---|
| `lib.rs` | PyO3 bindings: `Wall`, `Wall2D`, `Room`, `Room2D`, `Microphone`, histograms, `rir_builder`, `delay_sum`, the geometry helpers, and the fast kernels and switches (`set_num_threads`, `set_parallel_rng`, ...) |
| `room.rs` | image source model and ray tracing (port of `room.{hpp,cpp}`) |
| `tracer.rs` | batch ray-tracing driver: parallel rays, ordered merge of histogram events |
| `wall.rs`, `geom.rs` | walls (polygon tests, intersections, reflections) and geometry routines |
| `mic.rs` | `Histogram2D` and `Microphone` |
| `rir.rs` | threaded `rir_builder` / `delay_sum` (original block partition) |
| `eig.rs` | bit-exact emulation of the Eigen operations used by the original (norms, dots, `JacobiSVD`) |
| `rng.rs` | bit-exact `std::mt19937_64` + libstdc++ `uniform_real_distribution<float>` |
| `fast.rs` | fused fast-mode kernels (multi-band ISM RIR synthesis) |
| `rt_kernels.rs` | kernels for the hot loops of `simulation/rt.py` (Poisson sequence, directional sequence) |
| `pool.rs` | bounded rayon thread pool shared by all kernels |

## Python package changes

Most of `python/pyroomacoustics/` is the upstream code. Changed or added:

| file | change |
|---|---|
| `parameters.py` | `num_threads` default and bound, `exact_mode`, `backend`, `cuda_device`, `cuda_min_ism_work`, `rt_tail`, `rt_tail_t0_impulse` |
| `room.py` | backend dispatch in `compute_rir`, `compute_anechoic_rir` (+ `early_ms`), `simulate(return_anechoic=..., early_ms=...)`, fast-mode ISM path |
| `simulation/ism.py`, `simulation/rt.py` | calls to the Rust kernels; optional removal of the t0 impulse |
| `cuda.py` | CUDA backend: shoebox ISM, direct-path RIRs, high-pass, convolutions, `compute_rirs` batching, fallback logic |
| `cuda_rt.py` | GPU ray tracing driver (job tables for the kernel) and tail synthesis (Poisson and Gaussian) |
| `_rt_triton.py` | Triton ray-tracing kernel |

## Data flow of `room.compute_rir()` (hybrid room)

```
image_source_model()            CPU (Rust) or GPU (cuda.ism_rirs, shoebox)
      │  ISM RIRs up to max_order
ray_tracing()                   CPU (tracer.rs) or GPU (_rt_triton.py)
      │  energy histograms per mic × source × band
tail synthesis                  CPU (rt.py + rt_kernels.rs) or GPU (cuda_rt.rt_tails)
      │  random impulse sequence × sqrt(histogram envelope), octave bands
ISM + tail, high-pass           CPU (sosfiltfilt) or GPU (|H(f)|²)
      ▼
room.rir[m][s]
```

`compute_anechoic_rir` runs the same ISM synthesis restricted to the order-0 image, with
the same post-processing, so the time origin is shared.

## Development notes

* Build with `maturin develop --release` and rerun `bench/anechoic_test.py`.
* **Changes that can affect exact mode:** run the `equiv/` harness against an
  environment with the original build. It must stay bitwise identical.
* **Changes to random outputs** (ray tracing, tails): run the paired validation in
  [validation.md](validation.md) with at least 4 seeds.
* **Shared machines:** keep `PRA_NUM_THREADS` ≤ 8, `pytest -n 4`, and pick an idle GPU
  with `CUDA_VISIBLE_DEVICES`.
