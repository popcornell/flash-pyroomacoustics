# Quirks of the original

These upstream behaviours came up while matching the original. They are reproduced as
they are (in both modes unless noted), because changing them would change the training
data distribution. Some have opt-out switches.

## Tail impulse at the direct sound

The random tail of hybrid RIRs (`simulation/rt.py`, `poisson_sequence`) starts with an
impulse of random sign exactly at the direct-path time t0. After the octave filtering, the
direct sound of the reverberant RIR gets a random gain (±30–40% in the cases checked).
The anechoic RIR does not contain it.

**Opt out:** `pra.constants.set("rt_tail_t0_impulse", False)`, in fast mode only.

## Missing tail for very close mics

The Poisson rate of the tail is evaluated at `t0 + t`, while the sequence itself starts at
t0. For a mic very close to the source (here 8 cm, t0 ≈ 0.2 ms), the rate right after t0
is tiny, so the second impulse can come 30 ms to 2 s later. The tail is then missing or
starts very late.

The Gaussian tail of the CUDA backend (`rt_tail="gaussian"`) always has a tail there. This
is the main source of its C50/DRR differences in [validation.md](validation.md).

## Interpolation of the tail envelope

`interp_hist` leaves the last padding samples of the envelope at zero, so the tail ends a
few samples early. It is reproduced on both backends.

## 2D shoebox ray tracing uses the wrong walls

The engine stores the 2D walls in the order south, east, north, west. Its shoebox hit test
indexes them as west, east, south, north, so three of the four sides use the wrong normal
and coefficients. The rays leave the room. In a 2D room with absorption 0.01, the
original's ray-tracing energy stops after 24 ms, whereas a 3D room keeps it up to the 1 s
time limit (verified on the original build).

For this reason the GPU ray tracing is 3D only. The CPU reproduces the original.

## 2D scattering reads uninitialized memory

`Wall2D.basis` is never initialized, so scattered directions in 2D are drawn from garbage
memory. They change from run to run and all go along ±x. Here the 2D basis is the wall
tangent. This difference is intentional and also applies in exact mode.

## `rir_builder` used every CPU

The original started `os.cpu_count()` threads on every call. Here the thread count is
bounded by `num_threads` (default 8).
