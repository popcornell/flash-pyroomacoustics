# Installation

The distribution is called `fast-pyroomacoustics`, but it installs the module
`pyroomacoustics`. It replaces the original, so the two cannot be installed in the same
environment:

```bash
pip uninstall pyroomacoustics
```

## From a wheel

```bash
pip install fast_pyroomacoustics-<version>-cp311-cp311-manylinux_2_31_x86_64.whl
```

The wheel is about 126 MB, almost all of it the SOFA directivity files shipped with the
original package (`pyroomacoustics/data/sofa`).

## From source

Requirements:

* Python ≥ 3.9, NumPy, SciPy;
* a Rust toolchain (stable, via [rustup](https://rustup.rs));
* [maturin](https://www.maturin.rs) ≥ 1.7.

```bash
git clone https://github.com/popcornell/fast-pyroomacoustics
cd fast-pyroomacoustics
pip install maturin

# development install into the active environment
maturin develop --release

# or build a wheel
maturin build --release
pip install target/wheels/*.whl
```

Always build with `--release`. A debug build is one to two orders of magnitude slower.

## CUDA backend

The GPU backend is optional and pure Python. It needs PyTorch with CUDA. Triton ships with
the Linux PyTorch wheels, and the ray-tracing kernel is JIT-compiled by it, so no `nvcc`
is needed.

```bash
pip install "fast-pyroomacoustics[cuda]"   # or: pip install torch
python -c "import pyroomacoustics as pra; print(pra.cuda.is_available())"
```

The backend was developed and tested with torch 2.5.1 + CUDA 12.4 (Triton 3.1) on A100
GPUs. The first ray-tracing call of a process compiles the kernel, which takes a few
seconds. Triton caches it on disk.

## Checking the install

```bash
python bench/anechoic_test.py        # alignment checks, CPU vs CUDA, prints ALL OK
```

The upstream tests live in the original repository, next to its modules. To run them
against this build, copy them out, so that the installed package is imported and not the
upstream source tree:

```bash
git clone https://github.com/LCAV/pyroomacoustics upstream
mkdir tests && cd upstream/pyroomacoustics && find . -path '*tests*' -name '*.py' \
    -exec cp --parents {} ../../tests/ \; && cd ../../tests
pip install pytest pytest-xdist
PRA_NUM_THREADS=8 pytest -q -p no:cacheprovider -n 4 .
```

The test results are the same as for the original build. The failures are environmental,
e.g. missing optional SOFA files. `test_rt_multiband_energy` is a random test that
occasionally fails in both builds.
