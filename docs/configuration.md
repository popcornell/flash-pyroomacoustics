# Configuration

Set constants with `pra.constants.set(name, value)` and read them with
`pra.constants.get(name)`. Some also have an environment variable, which gives the
default when the module is imported.

## Constants added by this package

| constant | env variable | default | meaning |
|---|---|---|---|
| `num_threads` | `PRA_NUM_THREADS` (then `OMP_NUM_THREADS`, `MKL_NUM_THREADS`) | `min(8, usable CPUs)` | Upper bound on the threads used by the compiled kernels and SciPy FFTs. Setting it resizes the pool. |
| `exact_mode` | – | `False` | `True`: bit-identical to the original C++ build (see [exact-mode.md](exact-mode.md)). Also forces the CPU backend. |
| `backend` | `PRA_BACKEND` | `"cpu"` | `"cpu"` or `"cuda"`. |
| `cuda_device` | `PRA_CUDA_DEVICE` | `"cuda"` | torch device of the CUDA backend, e.g. `"cuda:1"`. |
| `cuda_min_ism_work` | – | `40000` | Below this many image sources × microphones the ISM runs on the CPU, which is faster there. `0` forces the GPU. |
| `rt_tail` | `PRA_RT_TAIL` | `"poisson"` | Ray-tracing tail of the CUDA backend: `"poisson"` (as the original) or `"gaussian"` (opt-in, see [cuda.md](cuda.md#gaussian-tail)). |
| `rt_tail_t0_impulse` | `PRA_RT_TAIL_T0_IMPULSE` (`0`/`false` disables) | `True` | Keep the random-sign impulse that starts the tail at the direct-path sample (see [usage.md](usage.md#dropping-the-impulse-at-the-direct-sound)). Ignored in exact mode. |

"Usable CPUs" respects `taskset` and cgroup limits. The original's `rir_builder` started
`os.cpu_count()` threads on every call, which is a problem on shared servers.

## Upstream constants used by the new code paths

| constant | default | note |
|---|---|---|
| `rir_hpf_enable` | `True` | 10 Hz high-pass on every RIR. The CUDA backend applies it in the frequency domain (see [cuda.md](cuda.md#high-pass-filter)). |
| `rir_hpf_fc`, `rir_hpf_kwargs` | `10.0`, Butterworth order 2 | Same filter on both backends. |
| `frac_delay_length` | `81` | Windowed-sinc fractional delay length. The direct path is at `distance / c * fs + frac_delay_length // 2`. |
| `octave_bands_n_fft` | `512` | Octave filter bank length (multi-band materials and air absorption). |

## Low-level switches

`pyroomacoustics.libroom` has a few switches. They are normally set through the
constants above:

* `libroom.set_num_threads(n)`, used by `num_threads`;
* `libroom.set_parallel_rng(bool)`: one RNG per ray. `exact_mode=True` sets it to `False`.
