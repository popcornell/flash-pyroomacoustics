# Usage

Everything in the [pyroomacoustics documentation](https://pyroomacoustics.readthedocs.io/)
applies unchanged. This page covers what this package adds.

## Choosing a backend

| backend | what | select with |
|---|---|---|
| `"cpu"` (default) | Rust core, at most `num_threads` threads | – |
| `"cuda"` | PyTorch + Triton on a GPU, with automatic CPU fallback | `pra.constants.set("backend", "cuda")` or `PRA_BACKEND=cuda` |

The backend can also be passed per call:

```python
room.compute_rir(backend="cuda")
room.compute_anechoic_rir(backend="cpu")
```

`room.simulate()` uses the global backend. See [cuda.md](cuda.md) for what runs on the GPU.

## Aligned anechoic (direct-path) RIRs

Dereverberation and separation models need a clean target aligned with the reverberant
input. Every room can produce the direct-path RIR of each source/microphone pair, on both
backends:

```python
room.compute_rir()
room.compute_anechoic_rir()        # room.anechoic_rir[m][s]

# or the signals directly: shape (n_src, n_mics, n_samples) for both
premix, anechoic = room.simulate(return_premix=True, return_anechoic=True)

# only the anechoic signals (the mixture is still in room.mic_array.signals)
anechoic = room.simulate(return_anechoic=True)
```

`anechoic_rir[m][s]` is the order-0 component of `rir[m][s]`, computed by the same code:

* **Time origin.** It is the same in both arrays (sample `n` is the same instant), so the
  direct sound sits on the same sample, with the same fractional-delay filter.
* **Same processing.** It has the same `1/d` attenuation, source and microphone
  directivities, octave filter bank and high-pass filter. Air absorption is included unless
  `air_absorption=False`.
* **Randomized ISM.** With `use_rand_ism=True` it follows the randomly displaced direct-path
  image.
* **In `simulate`.** The source delays are applied, and with `snr=...` it gets the same
  normalization as `premix`. The output length is the same as `premix`.
* **Occlusion.** In non-convex rooms the free-field direct path is used even when a wall
  blocks it. `compute_anechoic_rir(visible_only=True)` gives zeros there instead, as in
  `rir`.

The default 10 Hz high-pass filter is zero-phase, so it smears a little low-frequency
energy of the reflections to before the direct path. The direct sound itself does not
move. With `pra.constants.set("rir_hpf_enable", False)`, `rir` and `anechoic_rir` agree to
~1e-7 (single band) or ~1e-3 (octave bands, whose filters have long tails) up to the first
reflection. `bench/anechoic_test.py` checks this.

## Direct path + early reflections target

```python
room.compute_anechoic_rir(early_ms=50)          # taper_ms=5 by default
premix, early = room.simulate(return_premix=True, return_anechoic=True, early_ms=50)
```

With `early_ms > 0` the target is the actual RIR `rir[m][s]`, kept up to `early_ms` after
the direct-path arrival at each microphone and faded out with a half cosine over
`taper_ms`. The time origin is the same, so `rir == early + late` sample by sample.

This target (often 50 ms) leaves the early reflections alone, which do not hurt
intelligibility, and removes only the late reverberation. The model does not have to undo
the strong, very regular early reflections of shoebox rooms. That is where the simulator is
least realistic and where the model would overfit most easily. `early_ms=0` (the default)
gives the direct path.

## Dropping the impulse at the direct sound

In hybrid (ISM + ray tracing) simulations, the original's tail always starts with an
impulse of random sign exactly at the direct-path sample. This gives the direct sound of
the reverberant RIR a random gain, ±30–40% in the cases checked. To drop it:

```python
pra.constants.set("rt_tail_t0_impulse", False)     # or PRA_RT_TAIL_T0_IMPULSE=0
```

* It works on both backends and with both tail types (`rt_tail`), and changes the RIR
  energy by less than 0.5%.
* It has no effect in `exact_mode`.
* Some random tail energy still overlaps the direct sound, because the tail starts at the
  direct-path time and the octave filters spread it.

## Dataset generation recipe

A typical dereverberation data loop: random shoebox rooms, hybrid simulation, and an
aligned target. Batch the rooms for the GPU.

```python
import numpy as np
import pyroomacoustics as pra

pra.constants.set("backend", "cuda")             # falls back to the CPU if unavailable
pra.constants.set("rt_tail_t0_impulse", False)
rng = np.random.default_rng(0)

def make_room(speech, fs=16000):
    dim = rng.uniform([3, 3, 2.5], [10, 8, 4])
    rt60 = rng.uniform(0.2, 1.0)
    e_abs, _ = pra.inverse_sabine(rt60, dim)
    room = pra.ShoeBox(dim, fs=fs, materials=pra.Material(e_abs, 0.1), max_order=3,
                       ray_tracing=True, air_absorption=True)
    room.add_source(dim * rng.uniform(0.2, 0.8, 3), signal=speech)
    room.add_microphone_array(dim[:, None] * rng.uniform(0.2, 0.8, (3, 4)))
    return room

rooms = [make_room(s) for s in speech_batch]
pra.cuda.compute_rirs(rooms, anechoic=True)      # one ray-tracing launch for all rooms
for room in rooms:
    premix, target = room.simulate(return_premix=True, return_anechoic=True,
                                   early_ms=50)  # or early_ms=0 for the direct path
    mixture = premix.sum(axis=0)                 # (n_mics, n_samples)
```

Notes:

* `pra.cuda.compute_rirs` traces all the rooms in one GPU launch and synthesizes all the
  tails in one batch. It is much faster than calling `compute_rir` room by room when each
  room has few rays.
* Rooms the GPU does not support (polygonal rooms, 2D ray tracing, directional mics, ...)
  go to the CPU, with a one-time warning.
* Hybrid simulation (ISM up to order 3 plus a random ray-traced tail) is a better training
  distribution than full-order ISM. The tail is random and diffuse instead of a regular
  lattice of images, which is closer to real rooms and harder to overfit. It is also
  faster.
* On a shared machine, keep `PRA_NUM_THREADS` small. Select the GPU with
  `PRA_CUDA_DEVICE=cuda:N` or `CUDA_VISIBLE_DEVICES`.

## Reproducibility

`pra.random.seed(numpy=..., libroom=...)` seeds both NumPy and the compiled engine:

* **CPU, fast mode.** Every ray has its own random generator, seeded from one draw of the
  libroom engine. Results are reproducible for a given seed, independently of the number
  of threads.
* **CPU, `exact_mode`.** It reproduces the original's global random stream exactly, so the
  original's seeds give the original's results bit for bit.
* **CUDA.** The ray-tracing and tail seeds are drawn from the package's NumPy generator
  (`pra.random.seed(numpy=...)`), so a fixed seed gives the same rays. Float32 atomic
  histogram accumulation makes runs differ at the rounding level.
