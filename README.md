# fast-pyroomacoustics

A faster, drop-in replacement for [pyroomacoustics](https://github.com/LCAV/pyroomacoustics).
The import name and API are unchanged (`import pyroomacoustics as pra`), and existing scripts
run as they are.

* **Rust core.** The C++/pybind11 extension `pyroomacoustics.libroom` is rewritten in Rust
  (PyO3 + rayon). The hot paths of the Python layer (ISM synthesis, octave filter bank,
  ray-tracing tail) are compiled too. It is 1.6–16× faster on 8 CPU threads.
* **CUDA backend (optional).** On a GPU (PyTorch + Triton) it runs the shoebox image source
  model, 3D ray tracing, tail synthesis, the high-pass filter and the `simulate()`
  convolutions. That is another 4–25× on dataset-style workloads.
* **Aligned training targets.** Direct-path RIRs, or direct path plus early reflections,
  aligned at the sample level with the reverberant RIRs. Built for dereverberation and
  separation datasets.
* **Exact mode.** `pra.constants.set("exact_mode", True)` gives results bit-identical to the
  original C++ build. The default fast mode is statistically equivalent: every acoustic
  parameter is within the original's own seed-to-seed variation.
* **Bounded threads.** At most 8 threads by default (`PRA_NUM_THREADS`), for shared
  machines.

## Install

From a release wheel:

```bash
pip install fast_pyroomacoustics-*.whl        # add torch with CUDA for the GPU backend
```

From source, which needs a Rust toolchain (`rustup`) and Python ≥ 3.9:

```bash
pip install maturin
maturin build --release && pip install target/wheels/*.whl
pip install torch                         # optional, for the CUDA backend
```

The package installs as `pyroomacoustics`, so uninstall the original first
(`pip uninstall pyroomacoustics`). See [docs/installation.md](docs/installation.md).

## Quick start

```python
import numpy as np
import pyroomacoustics as pra

room = pra.ShoeBox([6, 5, 3], fs=16000, materials=pra.Material(0.3, 0.1),
                   max_order=3, ray_tracing=True, air_absorption=True)
room.add_source([2, 3, 1.5], signal=np.random.randn(16000))
room.add_microphone_array(np.c_[[4, 2, 1.4], [4.1, 2, 1.4]])

# reverberant mixture + aligned direct-path target, same shape and time origin
premix, target = room.simulate(return_premix=True, return_anechoic=True)

# on a GPU, for a list of rooms at once (ray tracing in a single launch)
pra.constants.set("backend", "cuda")
pra.cuda.compute_rirs([room], anechoic=True)
premix, target = room.simulate(return_premix=True, return_anechoic=True)
```

## Performance

AMD EPYC 7742 limited to 8 threads, one A100, 4 mics per room:

| workload | original (C++) | this, CPU | this, CUDA |
|---|---:|---:|---:|
| 10 rooms, RT60 0.2–1 s, full-order ISM | 7.48 s | 1.74 s | 0.41 s |
| 10 rooms, hybrid ISM + ray tracing, scattering, air absorption | 5.96 s | 0.38 s | 0.05 s |
| 10 hybrid rooms, RIRs + anechoic targets + `simulate` (10 s signal) | – | 0.76 s | 0.12 s |

More benchmarks and the validation results are in [docs/validation.md](docs/validation.md).

## Documentation

| page | content |
|---|---|
| [Installation](docs/installation.md) | wheels, building from source, CUDA requirements |
| [Usage](docs/usage.md) | backends, aligned anechoic and early-reflection targets, dataset generation |
| [Configuration](docs/configuration.md) | all `pra.constants` and environment variables |
| [CUDA backend](docs/cuda.md) | what runs on the GPU, accuracy, fallbacks, Gaussian tail |
| [Exact mode](docs/exact-mode.md) | bit-exact reproduction of the C++ build, known differences |
| [Validation and benchmarks](docs/validation.md) | how equivalence is tested, how to rerun it |
| [Architecture](docs/architecture.md) | code layout, for contributors |
| [Quirks of the original](docs/quirks.md) | upstream behaviours found along the way |

For the pyroomacoustics API itself (rooms, materials, directivities, beamforming, DOA,
...), see the [upstream documentation](https://pyroomacoustics.readthedocs.io/). Everything
there applies unchanged.

## Citing

This package has no paper of its own. If you use it, please cite the original
pyroomacoustics paper, which is the reference for the simulation methods implemented
here. GitHub's "Cite this repository" button gives the same entry (from
[CITATION.cff](CITATION.cff)).

```bibtex
@inproceedings{scheibler2018pyroomacoustics,
  author    = {Scheibler, Robin and Bezzam, Eric and Dokmani{\'c}, Ivan},
  title     = {Pyroomacoustics: A Python package for audio room simulations and array processing algorithms},
  booktitle = {Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages     = {351--355},
  year      = {2018},
  doi       = {10.1109/ICASSP.2018.8461310}
}
```

If you want to point to this implementation as well (for the Rust/CUDA backends, the GPU
ray tracing or the aligned targets), link the repository:
<https://github.com/popcornell/fast-pyroomacoustics>.

## Credits and license

This is a reimplementation of [pyroomacoustics](https://github.com/LCAV/pyroomacoustics) by
Robin Scheibler, Eric Bezzam, Ivan Dokmanić and the LCAV/EPFL contributors. The Python
layer is theirs, with the changes described in the docs.

What this repository adds on top of the original:

* the Rust rewrite of `libroom`, with a bit-exact mode and a bounded thread pool;
* fused fast-mode kernels for ISM synthesis and the ray-tracing tail;
* the CUDA backend: GPU image source model, Triton ray tracing, batched tail synthesis,
  GPU high-pass and convolutions;
* aligned anechoic and early-reflection training targets, and the optional Gaussian tail
  and `rt_tail_t0_impulse` switch;
* the validation harnesses (`equiv/`, `bench/`).

MIT license, see [LICENSE](LICENSE). The original README is kept in
[docs/upstream-README.rst](docs/upstream-README.rst).
