# Fast mode and exact mode

## Fast mode (default)

* **Image source method.** RIRs are the same as the original up to float32 rounding
  (relative L2 error ~1e-4). RT60, EDT, C50 and DRR are identical. Distances, damping,
  air absorption and the fractional delays of all the bands are computed in one fused,
  parallel Rust kernel (`src/fast.rs`).
* **Ray tracing with scattering.** Every ray has its own random generator, so rays are
  traced in parallel. The original is itself random, and the output is statistically
  equivalent to it (see [validation.md](validation.md)).
* **Threads.** At most `num_threads` (default 8), see [configuration.md](configuration.md).

## Exact mode

```python
pra.constants.set("exact_mode", True)
```

The results are bit-identical to the original C++ build. Exact mode restores the
original's global random stream, its ISM synthesis path and its per-band filter bank. It is
still faster than the original, and it forces the CPU backend.

### How exactness is achieved

Floating-point addition is not associative, so "same output" means reproducing the exact
order of every operation of the original compiled code (GCC 9, `-O3`, SSE2, no FMA, Eigen
5.0.1).

* **Eigen evaluation order** (`src/eig.rs`). Fixed-size norms and products are summed as
  binary trees, fixed-size dots sequentially, and dynamic-size reductions with Eigen's 4-wide
  packet traversal (two accumulators, `predux = (x0+x2)+(x1+x3)`, then a sequential tail).
  These orders were determined from the Eigen sources and verified with C++ probes.
* **`JacobiSVD`.** The wall basis and normal come from Eigen's `JacobiSVD` (column-pivoting
  Householder QR preconditioner + 2×2 Jacobi sweeps). It is ported operation by operation,
  including its GEMV kernel and its tie-breaking.
* **RNG** (`src/rng.rs`). `std::mt19937_64` and libstdc++'s `generate_canonical<float>` /
  `uniform_real_distribution<float>` are re-implemented bit-exactly, so
  `pra.random.seed(libroom=...)`, Lambertian scattering and `Room.contains()` consume the
  same random stream.
* **Threaded RIR builders** (`src/rir.rs`). `rir_builder` and `delay_sum` keep the
  original's `num_threads`-block partition and summation order, because the float result
  depends on it. They use a persistent thread pool and only the sample ranges each block
  touches.
* **Ray tracing** (`src/tracer.rs`).
  * Histogram sums are accumulated in the original's serial order.
  * Rays never touch the RNG when no wall scatters, so those rays are traced in parallel
    and their events are merged in order.
  * With scattering, the RNG-dependent trajectories are computed sequentially. The
    expensive part (microphone visibility, energies, histogram events) runs in parallel and
    is merged in order.
* **`rt.py` hot loops** (`src/rt_kernels.rs`).
  * `poisson_sequence` draws its uniforms in a batch and runs the recurrence in Rust. It
    then rewinds the NumPy generator and advances it by exactly the number of draws used.
  * `directional_sequence`'s accumulation loop moved to Rust with identical operations.
* **Math library.** Transcendental functions (`sinf`, `cosf`, `atan2f`, `sin`, ...) call the
  same system libm as the original.

### Verification

`equiv/` is a differential harness: 86 scenarios and about 13.5k compared arrays. Every
output is compared bitwise against the original C++ build. It covers:

* ISM 2D/3D: all orders, octave bands, air absorption, min-phase, randomized ISM;
* hybrid and pure ray tracing, with and without scattering;
* non-convex polygonal rooms;
* directivities;
* `simulate()`;
* low-level geometry fuzzing (thousands of random walls and segments);
* the RIR builders for float32/float64 at several thread counts.

```bash
# in an environment with the original pyroomacoustics
python equiv/dump.py /tmp/ref
# in an environment with this package
(cd equiv && python exact_wrapper.py /tmp/fast)  # dump.py with exact_mode=True
python equiv/compare.py /tmp/ref /tmp/fast      # bitwise by default; --tol for tolerances
```

Run `dump.py` from a directory without a `pyroomacoustics/` folder, so that the installed
package is imported.

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
