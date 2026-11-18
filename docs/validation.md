# Validation and benchmarks

The goal is that a model trained on data from this package behaves as if trained on data
from the original. Three levels of checks:

1. **Bitwise** (`exact_mode`): the `equiv/` harness, see [exact-mode.md](exact-mode.md#verification).
2. **Deterministic outputs in fast mode** (ISM, GPU vs CPU, anechoic alignment): relative
   L2 error at float32 level, `bench/anechoic_test.py`.
3. **Random outputs** (ray tracing): paired statistical comparison of room-acoustic
   parameters against the original, `bench/validate.py` + `bench/compare_validate.py`.

## Statistical validation of the ray tracing

`bench/validate.py OUT.npz SEED` generates 12 random rooms × 2 mics for each room kind and
stores the RIRs and five parameters per RIR:

* T20, from the Schroeder decay;
* EDT;
* C50;
* DRR, with a ±2.5 ms window around the peak;
* total energy.

The room geometry depends only on the kind's position in the `KINDS` list. The random
seed of the simulation depends on `SEED`. The same rooms therefore get a different
realization of the random process in each build, and the comparison is paired.

```bash
export PRA_NUM_THREADS=8
# original pyroomacoustics environment
for s in 3 4 5 6; do KINDS=hybrid_scat,hybrid_octave_scat python bench/validate.py ref_s$s.npz $s; done
# this package (add PRA_BACKEND=cuda, PRA_RT_TAIL=gaussian, ... to test other modes)
for s in 3 4 5 6; do KINDS=hybrid_scat,hybrid_octave_scat python bench/validate.py new_s$s.npz $s; done
python bench/compare_validate.py "ref_s{}.npz" "new_s{}.npz" 3 4 5 6
```

`compare_validate.py` prints, for each kind and parameter, the mean paired difference, its
standard error, and the original's own seed-to-seed standard deviation, which is the
variation a model already sees from the original.

Results, 4 seeds × 24 RIRs per kind, as differences to the original (in SE):

| mode | kind | T20 | EDT | C50 | DRR | energy |
|---|---|---:|---:|---:|---:|---:|
| CPU, fast mode | hybrid, scattering 0.2 | −0.4 | +1.1 | −0.5 | +0.5 | +0.2 |
| CPU, fast mode | hybrid, octave materials, air | +0.9 | −0.2 | +0.5 | +0.4 | +0.8 |
| CUDA | hybrid, scattering 0.2 | +0.6 | +0.7 | −0.3 | −1.0 | +0.3 |
| CUDA | hybrid, octave materials, air | +2.3 | +0.2 | +0.1 | +0.2 | +0.4 |
| CUDA, Gaussian tail | hybrid, scattering 0.2 | +1.7 | +0.6 | −2.0* | −2.5* | +1.1 |
| CUDA, Gaussian tail | hybrid, octave materials, air | +1.3 | −1.6 | +0.5 | −2.3 | +2.6 |

\* These come from one mic 8 cm from the source. There the original's tail is often
missing, while the Gaussian tail is always there (see [quirks.md](quirks.md)).

In absolute terms the largest differences are 0.016 s of T20 (CUDA) and 0.21 dB of DRR /
0.08 dB of energy (Gaussian tail, octave rooms). The original's seed-to-seed standard
deviations are 0.046 s, 0.50 dB and 0.20 dB.

## Benchmarks

All on an AMD EPYC 7742 with both builds limited to 8 threads (`PRA_NUM_THREADS=8`), best
of 3.

### CPU: original vs this package (`bench/bench.py`, `bench/dataset_prof.py`)

| workload | original (C++) | this | speed-up |
|---|---:|---:|---:|
| ShoeBox ISM, order 17, 1 band, 4 mics | 0.011 s | 0.007 s | 1.6× |
| ShoeBox ISM, order 17, octave bands, 4 mics | 0.067 s | 0.014 s | 4.8× |
| Hybrid ISM(3)+RT, 10k rays, no scattering | 0.482 s | 0.031 s | 15× |
| Hybrid ISM(3)+RT, 10k rays, scattering 0.1 | 4.312 s | 0.380 s | 11× |
| L-shaped room, hybrid with scattering | 0.197 s | 0.019 s | 10× |
| 20 random `inverse_sabine` rooms (dataset generation) | 4.742 s | 0.902 s | 5.3× |
| 10 rooms × 4 mics, RT60 0.2–1 s, full-order ISM | 7.48 s | 1.74 s | 4.3× |
| 10 rooms × 4 mics, hybrid with scattering + air absorption | 5.96 s | 0.38 s | 16× |

### CPU vs CUDA (`bench/gpu_bench.py`, one A100)

See [cuda.md](cuda.md#benchmarks).

## Other scripts

| script | purpose |
|---|---|
| `bench/anechoic_test.py` | alignment of anechoic/early targets, CUDA vs CPU ISM, GPU high-pass, hybrid energy. Prints `ALL OK`. |
| `bench/bench.py` | CPU benchmarks of `room.simulate()` pipelines |
| `bench/dataset_prof.py` | dataset-style workloads (10 random rooms × 4 mics) |
| `bench/gpu_bench.py` | CPU vs CUDA vs CUDA + Gaussian tail |
| `bench/profile_ref.py`, `bench/rt_bench.py` | profiling helpers |
| `equiv/` | bitwise differential harness against the original build |
