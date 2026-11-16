"""
Registry of deterministic scenarios for differential testing of pyroomacoustics.

Each scenario is a function taking no argument and returning a flat dict
``str -> numpy array | python scalar | str``.  Every scenario seeds the package
RNG (``pra.random.seed(numpy=..., libroom=...)``) and uses a local
``np.random.default_rng(seed)`` for its own inputs, so the outputs only depend
on the pyroomacoustics implementation that is importable.

Only ``pyroomacoustics``, ``numpy`` and ``scipy`` are imported so the file runs
unchanged against the reference C++ build and the re-implementation.

Conventions for keys
--------------------
* ``<key>__layout`` : a string ``"<dtype>|<C/F/CF/->|<w/ro>"`` describing the
  memory layout and writeability of an array returned *directly* by libroom
  (not a copy made by Python code).  Used to detect API differences
  (C vs Fortran order, read-only views, ...).
* ``<key>__type`` : python type name of a value returned by libroom.
* ``err_<case>__type`` / ``err_<case>__msg`` : exception type / message for an
  error-behaviour case (``"<no exception>"`` if none was raised).
* ``__error__`` : the whole scenario raised (see ``run_scenario``).
"""

import re
import warnings

import numpy as np

import pyroomacoustics as pra
from pyroomacoustics import libroom

SCENARIOS = {}  # name -> (category, function)

# A fixed thread count for the threaded C++ routines, so that the outputs do not
# depend on os.cpu_count() (the summation order in rir_builder/delay_sum depends
# on the number of threads).
DEFAULT_NUM_THREADS = 4


def scenario(category, name=None):
    def deco(fn):
        SCENARIOS[name or fn.__name__] = (category, fn)
        return fn

    return deco


def _reset_globals():
    """Put package-global state back to known defaults before every scenario."""
    libroom.set_eps(1e-5)
    pra.constants.set("num_threads", DEFAULT_NUM_THREADS)
    for k in ["rir_hpf_enable", "frac_delay_length", "sinc_lut_granularity"]:
        pra.constants.set(k, pra.parameters._constants_default[k])


def run_scenario(name):
    category, fn = SCENARIOS[name]
    _reset_globals()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = fn()
    except Exception as e:  # noqa
        out = {"__error__": _err_str(e)}
    finally:
        _reset_globals()
    return out


# ---------------------------------------------------------------------------
# Recording helpers
# ---------------------------------------------------------------------------

_ADDR_RE = re.compile(r"0x[0-9a-fA-F]+")


def _err_str(e):
    return f"{type(e).__name__}: {_ADDR_RE.sub('0x?', str(e))}"


def layout(a):
    a = np.asarray(a)
    order = ("C" if a.flags.c_contiguous else "") + ("F" if a.flags.f_contiguous else "")
    return f"{a.dtype}|{order or '-'}|{'w' if a.flags.writeable else 'ro'}"


def put(out, key, val, lay=False):
    """Store val under key. If lay, also store its layout (for libroom returns)."""
    if key in out:
        raise KeyError(f"duplicate key {key}")
    if val is None:
        out[key] = "__None__"
        return
    if isinstance(val, np.ndarray):
        if lay:
            out[key + "__layout"] = layout(val)
        out[key] = np.array(val, copy=True)
    else:
        out[key] = val


def put_type(out, key, val):
    out[key + "__type"] = type(val).__name__


def record_error(out, case, fn):
    """Run fn(); record exception type/message, or the return value."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ret = fn()
        out[f"err_{case}__type"] = "<no exception>"
        out[f"err_{case}__msg"] = ""
        return ret
    except Exception as e:  # noqa
        out[f"err_{case}__type"] = type(e).__name__
        out[f"err_{case}__msg"] = _ADDR_RE.sub("0x?", str(e))
        return None


def capture_wall(out, w, prefix):
    for attr in ["normal", "origin", "corners", "absorption", "scatter"]:
        put(out, f"{prefix}_{attr}", getattr(w, attr), lay=True)
    if w.dim == 3:
        put(out, f"{prefix}_basis", w.basis, lay=True)
        put(out, f"{prefix}_flat_corners", w.flat_corners, lay=True)
    else:
        # Wall2D.basis is never initialized by the C++ code: its *content* is
        # uninitialized memory.  Only shape/dtype are recorded.
        b = w.basis
        put(out, f"{prefix}_basis_shape", np.array(b.shape))
        out[f"{prefix}_basis__layout"] = layout(b)
        put(out, f"{prefix}_flat_corners", w.flat_corners, lay=True)
    put(out, f"{prefix}_name", w.name)
    a = w.area()
    put_type(out, f"{prefix}_area", a)
    put(out, f"{prefix}_area", a)
    put(out, f"{prefix}_dim", w.dim)


def capture_engine(out, eng, prefix="eng"):
    for attr in [
        "sources",
        "orders",
        "orders_xyz",
        "gen_walls",
        "attenuations",
        "visible_mics",
        "source_directions",
    ]:
        put(out, f"{prefix}_{attr}", getattr(eng, attr), lay=True)
    md = eng.max_dist
    put_type(out, f"{prefix}_max_dist", md)
    put(out, f"{prefix}_max_dist", md)
    put(out, f"{prefix}_obstructing_walls", np.array(eng.obstructing_walls, dtype=np.int64))
    put(out, f"{prefix}_is_hybrid_sim", bool(eng.is_hybrid_sim))
    put(out, f"{prefix}_dim", eng.dim)
    walls = eng.walls
    put(out, f"{prefix}_n_walls", len(walls))
    for i, w in enumerate(walls):
        capture_wall(out, w, f"{prefix}_wall{i}")
    mics = eng.microphones
    put(out, f"{prefix}_n_mics", len(mics))
    for i, m in enumerate(mics):
        put(out, f"{prefix}_mic{i}_loc", m.loc, lay=True)
        hs = m.histograms
        put(out, f"{prefix}_mic{i}_n_hist", len(hs))
        if len(hs) > 0:
            h = hs[0].get_hist()
            put(out, f"{prefix}_mic{i}_hist0", h, lay=True)
            put(out, f"{prefix}_mic{i}_counts0", hs[0].get_counts(), lay=True)


def capture_room(out, room, prefix="", engine=True):
    """Capture everything observable after compute_rir()/simulate()."""
    p = prefix
    if room.rir is not None:
        for m, row in enumerate(room.rir):
            for s, h in enumerate(row):
                put(out, f"{p}rir_m{m}_s{s}", h)
    if room.mic_array is not None and room.mic_array.signals is not None:
        put(out, f"{p}signals", room.mic_array.signals)
    for s, src in enumerate(room.sources):
        for attr in ["images", "damping", "orders", "orders_xyz", "walls", "directions"]:
            v = getattr(src, attr, None)
            if v is not None:
                put(out, f"{p}src{s}_{attr}", np.asarray(v))
    if room.visibility is not None:
        for s, v in enumerate(room.visibility):
            put(out, f"{p}visibility_s{s}", np.asarray(v))
    rth = getattr(room, "rt_histograms", None)
    if rth is not None:
        for m, row in enumerate(rth):
            for s, hl in enumerate(row):
                for d, h in enumerate(hl):
                    put(out, f"{p}rt_hist_m{m}_s{s}_d{d}", h, lay=(m == 0 and s == 0 and d == 0))
    if engine:
        capture_engine(out, room.room_engine, prefix=f"{p}eng")


# ---------------------------------------------------------------------------
# Common room builders
# ---------------------------------------------------------------------------

MICS3D = np.c_[[2.1, 1.7, 1.2], [2.6, 3.1, 1.6], [4.3, 2.2, 0.9]]
SRCS3D = [[1.2, 1.1, 1.5], [3.7, 3.4, 2.1]]
DIM3D = [5.3, 4.4, 2.9]

MICS2D = np.c_[[2.1, 1.7], [2.6, 3.1], [4.3, 2.2]]
SRCS2D = [[1.2, 1.1], [3.7, 3.4]]
DIM2D = [5.3, 4.4]


def _signals(rng, fs, n_src, dur=0.25):
    return [rng.standard_normal(int(dur * fs)) for _ in range(n_src)]


def _populate(room, rng, srcs, mics, fs, with_signals=True, src_dirs=None, mic_dirs=None):
    sigs = _signals(rng, fs, len(srcs)) if with_signals else [None] * len(srcs)
    for i, (s, x) in enumerate(zip(srcs, sigs)):
        d = None if src_dirs is None else src_dirs[i]
        room.add_source(s, signal=x, delay=0.01 * i, directivity=d)
    room.add_microphone_array(mics, directivity=mic_dirs)
    return room


def _shoebox_run(
    seed,
    dim=DIM3D,
    fs=16000,
    max_order=3,
    materials=None,
    srcs=SRCS3D,
    mics=MICS3D,
    simulate=True,
    rt=None,
    **kwargs,
):
    pra.random.seed(numpy=seed, libroom=seed + 1000)
    rng = np.random.default_rng(seed)
    if materials is None:
        materials = pra.Material(0.25)
    room = pra.ShoeBox(dim, fs=fs, max_order=max_order, materials=materials, **kwargs)
    if rt is not None:
        room.set_ray_tracing(**rt)
    _populate(room, rng, srcs, mics, fs)
    out = {}
    if simulate:
        room.simulate()
    else:
        room.compute_rir()
    capture_room(out, room)
    return out, room


def _octave_materials():
    return pra.make_materials(
        ceiling="hard_surface",
        floor="carpet_cotton",
        east="brickwork",
        west="brickwork",
        north="plasterboard",
        south="wooden_lining",
    )


# ---------------------------------------------------------------------------
# ShoeBox 3D ISM
# ---------------------------------------------------------------------------

for _order in [0, 1, 3, 10, 17]:

    def _mk(order):
        def fn():
            out, _ = _shoebox_run(100 + order, max_order=order, simulate=(order <= 10))
            return out

        return fn

    SCENARIOS[f"ism3d_single_band_o{_order}"] = ("ism3d", _mk(_order))


@scenario("ism3d")
def ism3d_octave_brickwork():
    out, _ = _shoebox_run(201, materials=pra.Material("brickwork"), max_order=4)
    return out


@scenario("ism3d")
def ism3d_octave_per_wall():
    m = _octave_materials()
    out, _ = _shoebox_run(202, materials=m, max_order=5)
    return out


@scenario("ism3d")
def ism3d_octave_scatter_ism_only():
    # scattering coefficients given but no ray tracing
    out, _ = _shoebox_run(203, materials=pra.Material("brickwork", "rpg_skyline"), max_order=3)
    return out


@scenario("ism3d")
def ism3d_fs8k():
    out, _ = _shoebox_run(204, fs=8000, max_order=6, materials=pra.Material(0.4))
    return out


@scenario("ism3d")
def ism3d_fs44k_octave():
    out, _ = _shoebox_run(205, fs=44100, max_order=3, materials=_octave_materials())
    return out


@scenario("ism3d")
def ism3d_air_absorption():
    out, _ = _shoebox_run(206, max_order=4, air_absorption=True, materials=_octave_materials())
    return out


@scenario("ism3d")
def ism3d_air_absorption_single_band():
    out, _ = _shoebox_run(207, max_order=4, air_absorption=True, temperature=25.0, humidity=60.0)
    return out


@scenario("ism3d")
def ism3d_min_phase():
    out, _ = _shoebox_run(208, max_order=3, min_phase=True, materials=_octave_materials())
    return out


@scenario("ism3d")
def ism3d_min_phase_air():
    out, _ = _shoebox_run(209, max_order=3, min_phase=True, air_absorption=True, materials=pra.Material("brickwork"))
    return out


@scenario("ism3d")
def ism3d_rand_ism():
    out, _ = _shoebox_run(210, max_order=5, use_rand_ism=True)
    return out


@scenario("ism3d")
def ism3d_rand_ism_disp():
    out, _ = _shoebox_run(211, max_order=4, use_rand_ism=True, max_rand_disp=0.03, materials=_octave_materials())
    return out


@scenario("ism3d")
def ism3d_hpf_disabled():
    pra.constants.set("rir_hpf_enable", False)
    out, _ = _shoebox_run(212, max_order=3)
    return out


@scenario("ism3d")
def ism3d_fdl61_threads1():
    pra.constants.set("frac_delay_length", 61)
    pra.constants.set("num_threads", 1)
    out, _ = _shoebox_run(213, max_order=4)
    return out


@scenario("ism3d")
def ism3d_threads64():
    pra.constants.set("num_threads", 64)
    out, _ = _shoebox_run(214, max_order=6, materials=_octave_materials())
    return out


@scenario("ism3d")
def ism3d_one_mic_one_src():
    out, _ = _shoebox_run(215, max_order=2, srcs=SRCS3D[:1], mics=MICS3D[:, :1])
    return out


@scenario("ism3d")
def ism3d_sigma2_awgn():
    # (t0 != 0 raises NotImplementedError in pra; see errors scenario)
    out, _ = _shoebox_run(216, max_order=2, sigma2_awgn=1e-4)
    return out


@scenario("ism3d")
def ism3d_inverse_sabine():
    rt60 = 0.45
    room_dim = [6.0, 5.0, 3.1]
    e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)
    out, room = _shoebox_run(
        217, dim=room_dim, max_order=max_order, materials=pra.Material(e_absorption)
    )
    put(out, "inv_sabine_e_absorption", float(e_absorption))
    put(out, "inv_sabine_max_order", int(max_order))
    put(out, "rt60_theory_sabine", float(room.rt60_theory("sabine")))
    put(out, "rt60_theory_eyring", float(room.rt60_theory("eyring")))
    put(out, "measured_rt60", np.asarray(room.measure_rt60()))
    return out


@scenario("ism3d")
def ism3d_inverse_sabine_hybrid():
    rt60 = 0.3
    room_dim = [4.0, 3.5, 2.7]
    e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)
    out, room = _shoebox_run(
        218,
        dim=room_dim,
        max_order=3,
        materials=pra.Material(e_absorption),
        ray_tracing=True,
        air_absorption=True,
        mics=np.c_[[2.0, 1.5, 1.2], [1.0, 2.5, 1.7]],
        srcs=[[3.0, 1.0, 1.3]],
        rt=dict(n_rays=5000, receiver_radius=0.5, energy_thres=1e-5, time_thres=0.6, hist_bin_size=0.004),
    )
    put(out, "inv_sabine_e_absorption", float(e_absorption))
    put(out, "inv_sabine_max_order", int(max_order))
    put(out, "measured_rt60", np.asarray(room.measure_rt60()))
    put(out, "measured_rt60_db30", np.asarray(room.measure_rt60(decay_db=30)))
    return out


@scenario("simulate")
def sim_snr():
    pra.random.seed(numpy=301, libroom=302)
    rng = np.random.default_rng(301)
    room = pra.ShoeBox(DIM3D, fs=16000, max_order=3, materials=pra.Material(0.3))
    _populate(room, rng, SRCS3D, MICS3D, 16000)
    room.simulate(snr=10.0, reference_mic=1)
    out = {}
    capture_room(out, room, engine=False)
    put(out, "sigma2_awgn", float(room.sigma2_awgn))
    return out


@scenario("simulate")
def sim_return_premix():
    pra.random.seed(numpy=303, libroom=304)
    rng = np.random.default_rng(303)
    room = pra.ShoeBox(DIM3D, fs=16000, max_order=3, materials=_octave_materials())
    _populate(room, rng, SRCS3D, MICS3D, 16000)
    premix = room.simulate(return_premix=True)
    out = {}
    put(out, "premix", premix)
    capture_room(out, room, engine=False)
    premix2 = room.simulate(snr=5.0, return_premix=True)
    put(out, "premix_snr", premix2)
    put(out, "signals_snr", room.mic_array.signals)
    return out


@scenario("simulate")
def sim_compute_rir_twice():
    pra.random.seed(numpy=305, libroom=306)
    rng = np.random.default_rng(305)
    room = pra.ShoeBox(DIM3D, fs=16000, max_order=4, materials=pra.Material(0.2, 0.15), ray_tracing=True, use_rand_ism=True)
    room.set_ray_tracing(n_rays=3000, receiver_radius=0.4, energy_thres=1e-5, time_thres=0.5)
    _populate(room, rng, SRCS3D, MICS3D, 16000)
    out = {}
    room.compute_rir()
    capture_room(out, room, prefix="a_", engine=False)
    room.compute_rir()
    capture_room(out, room, prefix="b_")
    room.simulate()
    put(out, "signals_after", room.mic_array.signals)
    return out


@scenario("simulate")
def sim_measure_rt60():
    out, room = _shoebox_run(307, max_order=12, materials=pra.Material(0.15), simulate=False)
    put(out, "room_rt60", np.asarray(room.measure_rt60()))
    put(out, "room_rt60_db20", np.asarray(room.measure_rt60(decay_db=20)))
    put(out, "room_rt60_lin", np.asarray(room.measure_rt60(linear_domain_fit=True)))
    h = room.rir[0][0]
    put(out, "exp_rt60", float(pra.experimental.measure_rt60(h, fs=room.fs)))
    put(out, "exp_rt60_db30", float(pra.experimental.measure_rt60(h, fs=room.fs, decay_db=30)))
    put(out, "exp_rt60_thres", float(pra.experimental.measure_rt60(h, fs=room.fs, decay_db=20, energy_thres=0.95)))
    put(out, "rt60_theory_sabine", float(room.rt60_theory("sabine")))
    put(out, "rt60_theory_eyring", float(room.rt60_theory("eyring")))
    put(out, "volume", float(room.get_volume()))
    return out


@scenario("simulate")
def sim_add_soundsource_objects():
    pra.random.seed(numpy=309, libroom=310)
    rng = np.random.default_rng(309)
    fs = 16000
    sources = [pra.SoundSource(s, signal=rng.standard_normal(3000)) for s in SRCS3D]
    mics = pra.MicrophoneArray(MICS3D, fs)
    room = pra.ShoeBox(DIM3D, fs=fs, max_order=2, materials=pra.Material(0.3), sources=sources, mics=mics)
    room.simulate()
    out = {}
    capture_room(out, room, engine=False)
    put(out, "direct_snr", float(room.direct_snr(MICS3D[:, 0])))
    put(out, "is_inside", np.array([room.is_inside(p) for p in rng.uniform(-1, 7, size=(50, 3))]))
    put(out, "bbox", np.asarray(room.get_bbox()))
    return out


# ---------------------------------------------------------------------------
# ShoeBox 2D ISM
# ---------------------------------------------------------------------------


@scenario("ism2d")
def ism2d_single_band_o1():
    out, _ = _shoebox_run(401, dim=DIM2D, srcs=SRCS2D, mics=MICS2D, max_order=1)
    return out


@scenario("ism2d")
def ism2d_single_band_o8():
    out, _ = _shoebox_run(402, dim=DIM2D, srcs=SRCS2D, mics=MICS2D, max_order=8, materials=pra.Material(0.35))
    return out


@scenario("ism2d")
def ism2d_octave():
    mats = pra.make_materials(east="brickwork", west="plasterboard", north="hard_surface", south="carpet_cotton")
    out, _ = _shoebox_run(403, dim=DIM2D, srcs=SRCS2D, mics=MICS2D, max_order=5, materials=mats, air_absorption=True)
    return out


@scenario("ism2d")
def ism2d_rand_ism_fs8k():
    out, _ = _shoebox_run(404, dim=DIM2D, srcs=SRCS2D, mics=MICS2D, fs=8000, max_order=4, use_rand_ism=True)
    return out


# ---------------------------------------------------------------------------
# Hybrid ISM + ray tracing
# ---------------------------------------------------------------------------

RT_SMALL = dict(n_rays=10000, receiver_radius=0.5, energy_thres=1e-6, time_thres=0.8, hist_bin_size=0.004)


@scenario("hybrid")
def hybrid3d_single_band():
    out, _ = _shoebox_run(501, max_order=3, materials=pra.Material(0.2), ray_tracing=True, rt=RT_SMALL)
    return out


@scenario("hybrid")
def hybrid3d_single_band_scatter():
    out, _ = _shoebox_run(502, max_order=3, materials=pra.Material(0.2, 0.3), ray_tracing=True, rt=RT_SMALL)
    return out


@scenario("hybrid")
def hybrid3d_octave():
    out, _ = _shoebox_run(503, max_order=2, materials=_octave_materials(), ray_tracing=True, rt=RT_SMALL)
    return out


@scenario("hybrid")
def hybrid3d_octave_scatter():
    out, _ = _shoebox_run(
        504,
        max_order=3,
        materials=pra.Material("brickwork", "rpg_skyline"),
        ray_tracing=True,
        air_absorption=True,
        rt=dict(n_rays=20000, receiver_radius=0.3, energy_thres=1e-7, time_thres=1.0, hist_bin_size=0.002),
    )
    return out


@scenario("hybrid")
def hybrid3d_mixed_scatter_per_wall():
    mats = {
        "ceiling": pra.Material("hard_surface", "rpg_qrd"),
        "floor": pra.Material("carpet_cotton", "classroom_tables"),
        "east": pra.Material("brickwork"),
        "west": pra.Material("brickwork", "rpg_skyline"),
        "north": pra.Material("plasterboard", "amphitheatre_steps"),
        "south": pra.Material("wooden_lining"),
    }
    out, _ = _shoebox_run(505, max_order=2, materials=mats, ray_tracing=True, rt=RT_SMALL, fs=8000)
    return out


@scenario("hybrid")
def hybrid3d_default_rt_params():
    # set_ray_tracing() without n_rays: n_rays from Vorlaender formula
    pra.random.seed(numpy=506, libroom=1506)
    rng = np.random.default_rng(506)
    room = pra.ShoeBox([3.0, 2.5, 2.2], fs=8000, max_order=2, materials=pra.Material(0.4, 0.1), ray_tracing=True)
    _populate(room, rng, [[1.0, 1.0, 1.0]], np.c_[[2.0, 1.5, 1.2]], 8000)
    room.compute_rir()
    out = {}
    put(out, "n_rays", int(room.rt_args["n_rays"]))
    capture_room(out, room)
    return out


@scenario("hybrid")
def hybrid3d_receiver_radius_hist():
    out, _ = _shoebox_run(
        507,
        max_order=1,
        materials=pra.Material(0.1, 0.05),
        ray_tracing=True,
        fs=44100,
        rt=dict(n_rays=2000, receiver_radius=0.8, energy_thres=1e-4, time_thres=0.4, hist_bin_size=0.01),
    )
    return out


@scenario("hybrid")
def hybrid2d_single_band():
    out, _ = _shoebox_run(511, dim=DIM2D, srcs=SRCS2D, mics=MICS2D, max_order=3, materials=pra.Material(0.2), ray_tracing=True, rt=RT_SMALL)
    return out


@scenario("hybrid")
def hybrid2d_scatter():
    # NOTE: 2D scattering uses Wall2D.basis, which is uninitialized in the C++
    # reference (see compare.py KNOWN_NONDETERMINISTIC).
    out, _ = _shoebox_run(512, dim=DIM2D, srcs=SRCS2D, mics=MICS2D, max_order=2, materials=pra.Material(0.2, 0.25), ray_tracing=True, rt=RT_SMALL)
    return out


@scenario("hybrid")
def hybrid2d_octave():
    mats = pra.make_materials(east="brickwork", west="plasterboard", north="hard_surface", south="carpet_cotton")
    out, _ = _shoebox_run(513, dim=DIM2D, srcs=SRCS2D, mics=MICS2D, max_order=2, materials=mats, ray_tracing=True, rt=RT_SMALL)
    return out


@scenario("hybrid")
def rt_only3d():
    out, _ = _shoebox_run(521, max_order=-1, materials=pra.Material(0.25, 0.1), ray_tracing=True, rt=dict(RT_SMALL, n_rays=8000))
    return out


@scenario("hybrid")
def rt_only3d_octave():
    out, _ = _shoebox_run(522, max_order=-1, materials=_octave_materials(), ray_tracing=True, air_absorption=True, rt=RT_SMALL)
    return out


@scenario("hybrid")
def rt_only2d():
    out, _ = _shoebox_run(523, dim=DIM2D, srcs=SRCS2D, mics=MICS2D, max_order=-1, materials=pra.Material(0.25), ray_tracing=True, rt=RT_SMALL)
    return out


@scenario("hybrid")
def hybrid_unset_ray_tracing():
    pra.random.seed(numpy=524, libroom=1524)
    rng = np.random.default_rng(524)
    room = pra.ShoeBox(DIM3D, fs=16000, max_order=2, materials=pra.Material(0.3, 0.1), ray_tracing=True)
    room.set_ray_tracing(n_rays=1000)
    room.unset_ray_tracing()
    _populate(room, rng, SRCS3D, MICS3D, 16000)
    room.simulate()
    out = {}
    capture_room(out, room)
    return out


# ---------------------------------------------------------------------------
# Polygonal rooms
# ---------------------------------------------------------------------------

PENTAGON = np.array([[0.0, 4.0, 5.5, 3.0, -0.5], [0.0, 0.0, 3.0, 5.0, 3.0]])
LSHAPE = np.array([[0.0, 6.0, 6.0, 2.5, 2.5, 0.0], [0.0, 0.0, 2.5, 2.5, 5.0, 5.0]])
LSHAPE_MICS2D = np.c_[[1.2, 4.0], [5.0, 1.0], [1.0, 1.2]]
LSHAPE_SRCS2D = [[4.5, 1.5], [1.2, 3.8]]
PENT_MICS2D = np.c_[[2.0, 1.0], [3.5, 2.5], [1.0, 2.5]]
PENT_SRCS2D = [[1.5, 1.5], [3.0, 3.5]]


def _poly_run(seed, corners, srcs, mics, max_order, height=None, mats2d=None, mats3d=None, rt=None, fs=16000, **kw):
    pra.random.seed(numpy=seed, libroom=seed + 1000)
    rng = np.random.default_rng(seed)
    room = pra.Room.from_corners(corners, fs=fs, max_order=max_order, materials=mats2d, ray_tracing=rt is not None, **kw)
    if height is not None:
        room.extrude(height, materials=mats3d)
        mics = np.r_[mics, 1.3 + 0.2 * np.arange(mics.shape[1])[None, :]]
        srcs = [list(s) + [1.1 + 0.3 * i] for i, s in enumerate(srcs)]
    if rt is not None:
        room.set_ray_tracing(**rt)
    _populate(room, rng, srcs, mics, fs)
    room.simulate()
    out = {}
    capture_room(out, room)
    put(out, "volume", float(room.get_volume()))
    pts = rng.uniform(-1, 7, size=(60, room.dim))
    put(out, "is_inside", np.array([room.is_inside(p) for p in pts]))
    put(out, "engine_contains", np.array([room.room_engine.contains(p.astype(np.float32)) for p in pts]))
    return out


for _o in [1, 3, 6]:

    def _mk_pc(o):
        def fn():
            return _poly_run(600 + o, PENTAGON, PENT_SRCS2D, PENT_MICS2D, o, mats2d=pra.Material(0.2))

        return fn

    SCENARIOS[f"poly2d_convex_o{_o}"] = ("poly", _mk_pc(_o))


@scenario("poly")
def poly2d_lshape_o4():
    return _poly_run(611, LSHAPE, LSHAPE_SRCS2D, LSHAPE_MICS2D, 4, mats2d=pra.Material(0.15))


@scenario("poly")
def poly2d_lshape_octave_o3():
    mats = [pra.Material(m) for m in ["brickwork", "plasterboard", "hard_surface", "carpet_cotton", "brickwork", "wooden_lining"]]
    return _poly_run(612, LSHAPE, LSHAPE_SRCS2D, LSHAPE_MICS2D, 3, mats2d=mats)


@scenario("poly")
def poly2d_lshape_hybrid():
    return _poly_run(613, LSHAPE, LSHAPE_SRCS2D, LSHAPE_MICS2D, 2, mats2d=pra.Material(0.2), rt=RT_SMALL)


@scenario("poly")
def poly2d_lshape_hybrid_scatter():
    # 2D scattering -> uninitialized Wall2D.basis in the reference.
    return _poly_run(614, LSHAPE, LSHAPE_SRCS2D, LSHAPE_MICS2D, 2, mats2d=pra.Material(0.2, 0.2), rt=RT_SMALL)


@scenario("poly")
def poly3d_convex_o3():
    return _poly_run(621, PENTAGON, PENT_SRCS2D, PENT_MICS2D, 3, height=3.0, mats2d=pra.Material(0.2))


@scenario("poly")
def poly3d_convex_o5_mats():
    mats = {"floor": pra.Material("carpet_cotton"), "ceiling": pra.Material("hard_surface")}
    walls = [pra.Material("brickwork")] * 5
    return _poly_run(622, PENTAGON, PENT_SRCS2D, PENT_MICS2D, 5, height=2.7, mats2d=walls, mats3d=mats)


@scenario("poly")
def poly3d_lshape_o1():
    return _poly_run(623, LSHAPE, LSHAPE_SRCS2D, LSHAPE_MICS2D, 1, height=2.8, mats2d=pra.Material(0.25))


@scenario("poly")
def poly3d_lshape_o4():
    return _poly_run(624, LSHAPE, LSHAPE_SRCS2D, LSHAPE_MICS2D, 4, height=2.8, mats2d=pra.Material(0.25))


@scenario("poly")
def poly3d_lshape_hybrid():
    return _poly_run(625, LSHAPE, LSHAPE_SRCS2D, LSHAPE_MICS2D, 2, height=2.8, mats2d=pra.Material(0.25), rt=RT_SMALL)


@scenario("poly")
def poly3d_lshape_hybrid_scatter_octave():
    mats = {"floor": pra.Material("carpet_cotton", "classroom_tables"), "ceiling": pra.Material("hard_surface", "rpg_qrd")}
    walls = [pra.Material("brickwork", "rpg_skyline")] * 6
    return _poly_run(626, LSHAPE, LSHAPE_SRCS2D, LSHAPE_MICS2D, 3, height=2.8, mats2d=walls, mats3d=mats, rt=RT_SMALL, air_absorption=True)


@scenario("poly")
def poly3d_lshape_rt_only():
    return _poly_run(627, LSHAPE, LSHAPE_SRCS2D, LSHAPE_MICS2D, -1, height=2.8, mats2d=pra.Material(0.25, 0.15), rt=RT_SMALL)


@scenario("poly")
def poly3d_extrude_tilted():
    pra.random.seed(numpy=628, libroom=1628)
    rng = np.random.default_rng(628)
    room = pra.Room.from_corners(PENTAGON, fs=16000, max_order=3, materials=pra.Material(0.3))
    room.extrude(3.0, v_vec=np.array([0.2, 0.1, 1.0]))
    _populate(room, rng, [[1.6, 1.6, 1.0], [2.8, 3.0, 2.0]], np.c_[[2.2, 1.2, 1.4], [3.2, 2.5, 1.7]], 16000)
    room.compute_rir()
    out = {}
    capture_room(out, room)
    return out


# ---------------------------------------------------------------------------
# Directivities
# ---------------------------------------------------------------------------

from pyroomacoustics.directivities import (  # noqa: E402
    Cardioid,
    CardioidFamily,
    DirectionVector,
    FigureEight,
    HyperCardioid,
    SubCardioid,
)


def _dirs():
    src_dirs = [
        Cardioid(DirectionVector(azimuth=30, colatitude=80, degrees=True)),
        CardioidFamily(DirectionVector(azimuth=200, colatitude=100, degrees=True), p=0.3, gain=0.8),
    ]
    mic_dirs = [
        HyperCardioid(DirectionVector(azimuth=-45, colatitude=90, degrees=True)),
        SubCardioid(DirectionVector(azimuth=120, colatitude=60, degrees=True)),
        FigureEight(DirectionVector(azimuth=10, colatitude=95, degrees=True), gain=1.2),
    ]
    return src_dirs, mic_dirs


def _dir_run(seed, rt=None, mats=None, max_order=3, src_only=False, mic_only=False, **kw):
    pra.random.seed(numpy=seed, libroom=seed + 1000)
    rng = np.random.default_rng(seed)
    src_dirs, mic_dirs = _dirs()
    if src_only:
        mic_dirs = None
    if mic_only:
        src_dirs = None
    # pra refuses directive mics in add_microphone_array() when ray tracing is
    # already on, so ray tracing is enabled after adding them (still supported
    # by Room.ray_tracing()).
    room = pra.ShoeBox(DIM3D, fs=16000, max_order=max_order, materials=mats or pra.Material(0.25), **kw)
    _populate(room, rng, SRCS3D, MICS3D, 16000, src_dirs=src_dirs, mic_dirs=mic_dirs)
    if rt is not None:
        room.set_ray_tracing(**rt)
    room.simulate()
    out = {}
    capture_room(out, room)
    return out


@scenario("directivity")
def dir_ism_src_mic():
    return _dir_run(701)


@scenario("directivity")
def dir_ism_src_only_octave():
    return _dir_run(702, mats=_octave_materials(), src_only=True)


@scenario("directivity")
def dir_ism_mic_only_minphase():
    return _dir_run(703, mic_only=True, min_phase=True)


@scenario("directivity")
def dir_hybrid_src_mic():
    return _dir_run(704, rt=dict(RT_SMALL, receiver_n_directions=50), mats=pra.Material(0.2, 0.1), max_order=2)


@scenario("directivity")
def dir_hybrid_octave_scatter():
    return _dir_run(
        705,
        rt=dict(RT_SMALL, n_rays=3000, receiver_n_directions=30),
        mats=pra.Material("brickwork", "rpg_skyline"),
        max_order=2,
        air_absorption=True,
    )


@scenario("directivity")
def dir_rt_only_mic():
    return _dir_run(706, rt=dict(RT_SMALL, receiver_n_directions=40), mats=pra.Material(0.3), max_order=-1, mic_only=True)


@scenario("directivity")
def dir_patterns():
    """Directivity objects themselves (pure Python but used by the room code)."""
    pra.random.seed(numpy=707, libroom=1707)
    rng = np.random.default_rng(707)
    src_dirs, mic_dirs = _dirs()
    out = {}
    az = rng.uniform(0, 360, 200)
    col = rng.uniform(0, 180, 200)
    vecs = rng.standard_normal((3, 200))
    vecs /= np.linalg.norm(vecs, axis=0)
    for i, d in enumerate(src_dirs + mic_dirs):
        put(out, f"d{i}_resp", d.get_response(az, col, degrees=True))
        put(out, f"d{i}_resp_cart", d.get_response_cartesian(vecs.T))
        rays, en = d.sample_rays(500)
        put(out, f"d{i}_rays", rays)
        put(out, f"d{i}_energies", np.asarray(en))
    return out


def _measured_directivity(rng, fs, n_grid=48, n_taps=64):
    grid = pra.doa.GridSphere(n_points=n_grid)
    irs = np.zeros((n_grid, n_taps))
    irs[:, 0] = 1.0
    irs[:, 1:] = 0.1 * rng.standard_normal((n_grid, n_taps - 1)) * np.exp(-np.arange(1, n_taps) / 8.0)
    cart = grid.cartesian
    irs[:, 0] *= 0.6 + 0.4 * cart[0]  # some direction dependent gain
    return pra.MeasuredDirectivity(
        orientation=pra.Rotation3D([30.0, 10.0], "zy", degrees=True), grid=grid, impulse_responses=irs, fs=fs
    )


@scenario("directivity")
def dir_measured_synthetic_ism():
    pra.random.seed(numpy=708, libroom=1708)
    rng = np.random.default_rng(708)
    md = _measured_directivity(rng, 16000)
    room = pra.ShoeBox(DIM3D, fs=16000, max_order=3, materials=pra.Material(0.3))
    _populate(room, rng, SRCS3D[:1], MICS3D[:, :2], 16000, mic_dirs=[md, None])
    room.simulate()
    out = {}
    capture_room(out, room)
    return out


@scenario("directivity")
def dir_measured_synthetic_hybrid():
    pra.random.seed(numpy=709, libroom=1709)
    rng = np.random.default_rng(709)
    md = _measured_directivity(rng, 16000)
    room = pra.ShoeBox(DIM3D, fs=16000, max_order=2, materials=pra.Material(0.3, 0.1))
    _populate(room, rng, SRCS3D[:1], MICS3D[:, :2], 16000, mic_dirs=[md, None])
    room.set_ray_tracing(**dict(RT_SMALL, n_rays=3000, receiver_n_directions=40))
    room.simulate()
    out = {}
    capture_room(out, room)
    return out


@scenario("directivity")
def dir_sofa_measured():
    """Uses a bundled SOFA file; skipped (recorded) if the 'sofa' reader is unavailable."""
    out = {}
    try:
        import sofa  # noqa: F401
    except ImportError:
        out["__skipped__"] = "python package 'sofa' not installed"
        return out
    pra.random.seed(numpy=710, libroom=1710)
    rng = np.random.default_rng(710)
    db = pra.datasets.SOFADatabase()
    mdf = pra.MeasuredDirectivityFile("AKG_c480_c414_CUBE", fs=16000)
    md = mdf.get_mic_directivity("AKG_c414K", orientation=pra.Rotation3D([0, 0], "yz"))
    room = pra.ShoeBox(DIM3D, fs=16000, max_order=2, materials=pra.Material(0.3))
    _populate(room, rng, SRCS3D[:1], MICS3D[:, :1], 16000, mic_dirs=[md])
    room.set_ray_tracing(**dict(RT_SMALL, n_rays=2000))
    room.simulate()
    capture_room(out, room)
    del db
    return out


# ---------------------------------------------------------------------------
# Low-level libroom: walls
# ---------------------------------------------------------------------------


def _rand_rot(rng):
    q, r = np.linalg.qr(rng.standard_normal((3, 3)))
    q = q * np.sign(np.diag(r))[None, :]
    return q


def _rand_poly2d(rng, n, convex=True):
    ang = np.sort(rng.uniform(0, 2 * np.pi, n))
    rad = np.ones(n) if convex else rng.uniform(0.35, 1.0, n)
    s = rng.uniform(0.5, 4.0)
    return s * np.c_[rad * np.cos(ang), rad * np.sin(ang)].T


def _make_walls3d(rng, n_rand=400):
    """Return a list of (corners float32 (3,n), absorption, scatter)."""
    specs = []
    for i in range(n_rand):
        n = int(rng.integers(3, 9))
        p2 = _rand_poly2d(rng, n, convex=bool(i % 2 == 0))
        R = _rand_rot(rng)
        t = rng.uniform(-5, 5, size=(3, 1))
        c = R @ np.r_[p2, np.zeros((1, n))] + t
        if i % 3 == 1:
            c = c[:, ::-1]
        specs.append(c.astype(np.float32))
    # axis-aligned rectangles (shoebox-like walls), both orderings
    for i in range(24):
        L = rng.uniform(1, 6, size=3)
        ax = i % 3
        o = [a for a in range(3) if a != ax]
        c = np.zeros((3, 4))
        c[ax, :] = rng.choice([0.0, L[ax]])
        c[o[0], :] = [0, L[o[0]], L[o[0]], 0]
        c[o[1], :] = [0, 0, L[o[1]], L[o[1]]]
        if i % 2:
            c = c[:, ::-1]
        specs.append(c.astype(np.float32))
    # exact integer rectangles
    specs.append(np.array([[0, 4, 4, 0], [0, 0, 4, 4], [0, 0, 0, 0]], dtype=np.float32))
    specs.append(np.array([[0, 0, 0, 0], [3, 0, 0, 3], [0, 0, 4, 4]], dtype=np.float32))
    return specs


def _query_points3d(rng, c, w, n=24):
    """Random query segments around a wall, incl. degenerate ones."""
    ncor = c.shape[1]
    center = c.mean(axis=1)
    scale = float(np.max(np.abs(c - center[:, None]))) + 0.5
    normal = np.asarray(w.normal, dtype=np.float64)
    pts1, pts2 = [], []
    for k in range(n):
        kind = k % 8
        if kind == 0:  # generic crossing
            p = center + rng.uniform(-1, 1, 3) * scale
            a = p + normal * rng.uniform(0.2, 2)
            b = p - normal * rng.uniform(0.2, 2)
        elif kind == 1:  # random segment
            a = center + rng.uniform(-2, 2, 3) * scale
            b = center + rng.uniform(-2, 2, 3) * scale
        elif kind == 2:  # endpoint on a corner
            j = int(rng.integers(ncor))
            a = c[:, j].astype(np.float64)
            b = a + normal * rng.uniform(-2, 2)
        elif kind == 3:  # endpoint at an edge midpoint (boundary)
            j = int(rng.integers(ncor))
            a = 0.5 * (c[:, j] + c[:, (j + 1) % ncor]).astype(np.float64)
            b = a + rng.standard_normal(3)
        elif kind == 4:  # crossing exactly through an edge midpoint
            j = int(rng.integers(ncor))
            m = 0.5 * (c[:, j] + c[:, (j + 1) % ncor]).astype(np.float64)
            a, b = m + normal, m - normal
        elif kind == 5:  # segment lying in the plane
            a = c[:, 0].astype(np.float64)
            b = c[:, 1 % ncor].astype(np.float64) * 0.7 + 0.3 * center
        elif kind == 6:  # endpoint on the plane interior (centroid)
            a = center
            b = center + normal * rng.uniform(0.5, 1.5)
        else:  # tiny offsets around eps
            p = center + rng.uniform(-0.5, 0.5, 3)
            off = rng.choice([1e-6, 5e-6, 1e-5, 2e-5, 1e-4])
            a = p + normal * off
            b = p - normal * rng.uniform(0.5, 1.5)
        pts1.append(a)
        pts2.append(b)
    return np.array(pts1, dtype=np.float32), np.array(pts2, dtype=np.float32)


def _wall_fuzz(out, walls, specs, rng, prefix, dim):
    """Fuzz the methods of a list of walls; records stacked arrays."""
    n_q = 24
    rets_isect, pts_isect, rets_intersects = [], [], []
    rets_side, rets_refl, pts_refl = [], [], []
    nr1, nr2 = [], []
    types = set()
    for w, c in zip(walls, specs):
        if dim == 3:
            p1, p2 = _query_points3d(rng, c, w, n_q)
        else:
            p1, p2 = _query_points2d(rng, c, w, n_q)
        ri, pi, rs, rr, pr, a1, a2, rint = [], [], [], [], [], [], [], []
        for a, b in zip(p1, p2):
            loc = np.full(dim, np.nan, dtype=np.float32)
            r = w.intersection(a, b, loc)
            types.add(("intersection", type(r).__name__))
            ri.append(int(r))
            pi.append(loc)
            rint.append(int(w.intersects(a, b)))
            s = w.side(a)
            types.add(("side", type(s).__name__))
            rs.append(int(s))
            loc2 = np.full(dim, np.nan, dtype=np.float32)
            r2 = w.reflect(a, loc2)
            rr.append(int(r2))
            pr.append(loc2)
            length = float(np.float32(np.linalg.norm(b - a)))
            v1 = w.normal_reflect(a, b, length)
            v2 = w.normal_reflect(b - a)
            types.add(("normal_reflect", layout(v1)))
            a1.append(v1)
            a2.append(v2)
        rets_isect.append(ri)
        pts_isect.append(pi)
        rets_intersects.append(rint)
        rets_side.append(rs)
        rets_refl.append(rr)
        pts_refl.append(pr)
        nr1.append(a1)
        nr2.append(a2)
    put(out, f"{prefix}_intersection_ret", np.array(rets_isect, dtype=np.int64))
    put(out, f"{prefix}_intersection_pt", np.array(pts_isect, dtype=np.float32))
    put(out, f"{prefix}_intersects_ret", np.array(rets_intersects, dtype=np.int64))
    put(out, f"{prefix}_side_ret", np.array(rets_side, dtype=np.int64))
    put(out, f"{prefix}_reflect_ret", np.array(rets_refl, dtype=np.int64))
    put(out, f"{prefix}_reflect_pt", np.array(pts_refl, dtype=np.float32))
    put(out, f"{prefix}_normal_reflect3", np.array(nr1))
    put(out, f"{prefix}_normal_reflect1", np.array(nr2))
    put(out, f"{prefix}_return_types", np.array(sorted("%s:%s" % t for t in types)))


def _stack_wall_props(out, walls, prefix, dim, maxc=8):
    n = len(walls)
    normals = np.array([w.normal for w in walls])
    origins = np.array([w.origin for w in walls])
    areas = np.array([w.area() for w in walls], dtype=np.float64)
    ncor = np.array([w.corners.shape[1] for w in walls])
    corners = np.full((n, dim, maxc), np.nan, dtype=np.float32)
    for i, w in enumerate(walls):
        corners[i, :, : w.corners.shape[1]] = w.corners
    put(out, f"{prefix}_normal", normals)
    put(out, f"{prefix}_origin", origins)
    put(out, f"{prefix}_area", areas)
    put(out, f"{prefix}_ncorners", ncor)
    put(out, f"{prefix}_corners", corners)
    if dim == 3:
        put(out, f"{prefix}_basis", np.array([w.basis for w in walls]))
        flat = np.full((n, 2, maxc), np.nan, dtype=np.float32)
        for i, w in enumerate(walls):
            flat[i, :, : w.flat_corners.shape[1]] = w.flat_corners
        put(out, f"{prefix}_flat_corners", flat)
    else:
        put(out, f"{prefix}_flat_corners_shape", np.array([w.flat_corners.shape for w in walls]))
    w0 = walls[0]
    for attr in ["normal", "origin", "corners", "basis", "flat_corners", "absorption", "scatter"]:
        out[f"{prefix}_w0_{attr}__layout"] = layout(getattr(w0, attr))
    put_type(out, f"{prefix}_area", w0.area())


@scenario("lowlevel")
def ll_wall3d_fuzz():
    pra.random.seed(numpy=801, libroom=1801)
    rng = np.random.default_rng(801)
    specs = _make_walls3d(rng)
    walls = []
    for i, c in enumerate(specs):
        nb = [1, 1, 7][i % 3]
        ab = rng.uniform(0, 1, nb).astype(np.float32)
        sc = rng.uniform(0, 1, nb).astype(np.float32) * (i % 2)
        walls.append(libroom.Wall(c, ab, sc, f"w{i}"))
    out = {}
    _stack_wall_props(out, walls, "wall", 3)
    put(out, "wall_names", np.array([w.name for w in walls]))
    put(out, "wall_absorption_b0", np.array([w.absorption[0] for w in walls]))
    _wall_fuzz(out, walls, specs, rng, "wall", 3)
    # same_as against itself, a copy, the reversed wall, a shifted wall
    sa = []
    for c, w in zip(specs[:40], walls[:40]):
        w_copy = libroom.Wall(c.copy())
        w_rev = libroom.Wall(np.ascontiguousarray(c[:, ::-1]))
        w_shift = libroom.Wall(c + np.float32(1e-3))
        w_roll = libroom.Wall(np.ascontiguousarray(np.roll(c, 1, axis=1)))
        sa.append([w.same_as(w), w.same_as(w_copy), w.same_as(w_rev), w.same_as(w_shift), w.same_as(w_roll)])
    put(out, "same_as", np.array(sa))
    # default args
    wd = libroom.Wall(specs[-1])
    put(out, "default_absorption", wd.absorption, lay=True)
    put(out, "default_scatter", wd.scatter, lay=True)
    put(out, "default_name", wd.name)
    put(out, "wall_dim", libroom.Wall.dim)
    put(out, "wall_dim_inst", wd.dim)
    # readwrite properties
    wd.absorption = [0.1, 0.2, 0.3]
    wd.scatter = np.array([0.5, 0.5, 0.25])
    wd.name = "renamed"
    put(out, "rw_absorption", wd.absorption, lay=True)
    put(out, "rw_scatter", wd.scatter, lay=True)
    put(out, "rw_name", wd.name)
    # corners passed as float64 / python lists
    wl = libroom.Wall([[0, 1, 1, 0], [0, 0, 1, 1], [2, 2, 2, 2]], [0.3], [0.1], "list")
    capture_wall(out, wl, "from_list")
    put(out, "isect_enum", np.array([int(libroom.Wall.Isect.NONE), int(libroom.Wall.Isect.VALID), int(libroom.Wall.Isect.ENDPT), int(libroom.Wall.Isect.BNDRY)]))
    put(out, "isect_consts", np.array([libroom.WALL_ISECT_NONE, libroom.WALL_ISECT_VALID, libroom.WALL_ISECT_VALID_ENDPT, libroom.WALL_ISECT_VALID_BNDRY]))
    put(out, "isect_enum_names", np.array([str(libroom.Wall.Isect.VALID), repr(libroom.Wall.Isect.BNDRY)]))
    return out


def _make_walls2d(rng, n=500):
    specs = []
    for i in range(n):
        if i % 5 == 0:  # axis aligned
            a = rng.uniform(0, 5, 2).round(1)
            b = a.copy()
            b[i % 2] += np.round(rng.uniform(0.5, 4), 1)
            c = np.c_[a, b]
        else:
            c = rng.uniform(-5, 5, size=(2, 2))
        if i % 3 == 1:
            c = c[:, ::-1]
        specs.append(np.ascontiguousarray(c, dtype=np.float32))
    return specs


def _query_points2d(rng, c, w, n=24):
    normal = np.asarray(w.normal, dtype=np.float64)
    c64 = c.astype(np.float64)
    pts1, pts2 = [], []
    for k in range(n):
        kind = k % 7
        t = rng.uniform(0, 1)
        p = c64[:, 0] * (1 - t) + c64[:, 1] * t
        if kind == 0:
            a, b = p + normal * rng.uniform(0.1, 2), p - normal * rng.uniform(0.1, 2)
        elif kind == 1:
            a, b = rng.uniform(-6, 6, 2), rng.uniform(-6, 6, 2)
        elif kind == 2:  # endpoint on a wall endpoint
            a = c64[:, int(rng.integers(2))]
            b = a + rng.standard_normal(2)
        elif kind == 3:  # endpoint on the wall
            a, b = p, p + normal * rng.uniform(-2, 2)
        elif kind == 4:  # through the wall endpoint
            e = c64[:, int(rng.integers(2))]
            a, b = e + normal, e - normal
        elif kind == 5:  # collinear
            a, b = c64[:, 0] * 1.2 - 0.2 * c64[:, 1], c64[:, 1] * 0.5 + 0.5 * c64[:, 0]
        else:
            off = rng.choice([1e-6, 5e-6, 1e-5, 2e-5, 1e-4])
            a, b = p + normal * off, p - normal
        pts1.append(a)
        pts2.append(b)
    return np.array(pts1, dtype=np.float32), np.array(pts2, dtype=np.float32)


@scenario("lowlevel")
def ll_wall2d_fuzz():
    pra.random.seed(numpy=802, libroom=1802)
    rng = np.random.default_rng(802)
    specs = _make_walls2d(rng)
    walls = [libroom.Wall2D(c, np.array([0.2], np.float32), np.array([0.1], np.float32), f"w{i}") for i, c in enumerate(specs)]
    out = {}
    _stack_wall_props(out, walls, "wall2d", 2, maxc=2)
    _wall_fuzz(out, walls, specs, rng, "wall2d", 2)
    sa = []
    for c, w in zip(specs[:40], walls[:40]):
        sa.append(
            [
                w.same_as(w),
                w.same_as(libroom.Wall2D(c.copy())),
                w.same_as(libroom.Wall2D(np.ascontiguousarray(c[:, ::-1]))),
                w.same_as(libroom.Wall2D(c + np.float32(1e-3))),
            ]
        )
    put(out, "same_as", np.array(sa))
    put(out, "wall2d_dim", libroom.Wall2D.dim)
    return out


@scenario("lowlevel")
def ll_lambertian():
    out = {}
    rng = np.random.default_rng(803)
    specs = _make_walls3d(rng, n_rand=6)
    for i, c in enumerate(specs[:6] + specs[-2:]):
        w = libroom.Wall(c)
        libroom.set_rng_seed(1234 + i)
        s = np.array([w.sample_lambertian_reflection() for _ in range(500)])
        put(out, f"lambert3d_w{i}", s)
    libroom.set_rng_seed(0)
    v = w.sample_lambertian_reflection()
    put(out, "lambert3d_single", v, lay=True)
    # interleaving with seeds via pra.random.seed
    pra.random.seed(numpy=1, libroom=2**63 + 5)
    put(out, "lambert3d_bigseed", np.array([w.sample_lambertian_reflection() for _ in range(50)]))
    pra.random.seed(numpy=1234)  # libroom seed derived from numpy rng
    put(out, "lambert3d_derived_seed", np.array([w.sample_lambertian_reflection() for _ in range(50)]))
    # 2D: uses the uninitialized Wall2D.basis -> only the shape is stored
    w2 = libroom.Wall2D(np.array([[0, 1], [0, 2]], dtype=np.float32))
    libroom.set_rng_seed(7)
    v2 = w2.sample_lambertian_reflection()
    put(out, "lambert2d_shape", np.array(v2.shape))
    out["lambert2d__layout"] = layout(v2)
    return out


# ---------------------------------------------------------------------------
# Low-level geometry routines
# ---------------------------------------------------------------------------


@scenario("lowlevel")
def ll_geometry_fuzz():
    rng = np.random.default_rng(810)
    out = {}
    N = 10000
    # ccw3p: random + exactly collinear on an integer grid
    P = rng.uniform(-5, 5, size=(N, 3, 2)).astype(np.float32)
    G = rng.integers(-3, 4, size=(N, 3, 2)).astype(np.float32)
    G[::3, 2] = 2 * G[::3, 1] - G[::3, 0]  # collinear
    G[1::7, 2] = G[1::7, 0]  # coincident
    put(out, "ccw3p_rand", np.array([libroom.ccw3p(*p) for p in P]))
    put(out, "ccw3p_grid", np.array([libroom.ccw3p(*p) for p in G]))
    put_type(out, "ccw3p", libroom.ccw3p(*P[0]))
    # 2D segments
    S = rng.uniform(-3, 3, size=(N, 4, 2)).astype(np.float32)
    SG = rng.integers(-2, 3, size=(N, 4, 2)).astype(np.float32)
    SG[::4, 2] = SG[::4, 0]  # shared endpoint
    SG[1::4, 2] = 0.5 * (SG[1::4, 0] + SG[1::4, 1])  # endpoint on other segment
    for nm, arr in [("rand", S), ("grid", SG)]:
        put(out, f"check_isect2d_{nm}", np.array([libroom.check_intersection_2d_segments(*s) for s in arr]))
        rets, locs = [], []
        for s in arr:
            loc = np.full(2, np.nan, dtype=np.float32)
            rets.append(libroom.intersection_2d_segments(*s, loc))
            locs.append(loc)
        put(out, f"isect2d_{nm}_ret", np.array(rets))
        put(out, f"isect2d_{nm}_loc", np.array(locs))
    # 3D segment / plane
    rets, locs = [], []
    for k in range(N):
        a1, a2, p = rng.uniform(-3, 3, size=(3, 3)).astype(np.float32)
        nrm = rng.standard_normal(3).astype(np.float32)
        if k % 5 == 1:
            a2 = p.copy()  # endpoint on plane
        if k % 5 == 2:
            nrm = np.cross(a2 - a1, rng.standard_normal(3)).astype(np.float32)  # parallel
        if k % 5 == 3:
            nrm = np.array([0, 0, 1], np.float32)
            a1[2] = p[2]
            a2[2] = p[2]  # in plane
        loc = np.full(3, np.nan, dtype=np.float32)
        rets.append(libroom.intersection_3d_segment_plane(a1, a2, p, nrm, loc))
        locs.append(loc)
    put(out, "isect3d_ret", np.array(rets))
    put(out, "isect3d_loc", np.array(locs))
    # cross
    V = rng.standard_normal((N, 2, 3)).astype(np.float32)
    cr = [libroom.cross(*v) for v in V]
    put(out, "cross", np.array(cr))
    out["cross__layout"] = layout(cr[0])
    # is_inside_2d_polygon
    rets = []
    poly_store = []
    for k in range(400):
        n = int(rng.integers(3, 9))
        if k % 4 == 0:
            pol = rng.integers(0, 5, size=(2, n)).astype(np.float32)
            pol = np.array([[0, 4, 4, 2, 0], [0, 0, 4, 2, 4]], np.float32) if k % 8 == 0 else np.array([[0, 4, 4, 0], [0, 0, 4, 4]], np.float32)
        else:
            pol = _rand_poly2d(rng, n, convex=(k % 2 == 0)).astype(np.float32)
        nc = pol.shape[1]
        row = []
        for j in range(12):
            if j < 4:
                pt = rng.uniform(-4, 4, 2)
            elif j < 6:
                pt = pol[:, int(rng.integers(nc))]  # vertex
            elif j < 8:
                i0 = int(rng.integers(nc))
                pt = 0.5 * (pol[:, i0] + pol[:, (i0 + 1) % nc])  # edge midpoint
            elif j < 10:
                pt = pol.mean(axis=1)
            else:
                pt = np.array([pol[0].min() - 1, pol[1, int(rng.integers(nc))]])  # ray through vertex
            row.append(libroom.is_inside_2d_polygon(np.asarray(pt, np.float32), pol))
        rets.append(row)
        poly_store.append(float(libroom.area_2d_polygon(pol)))
    put(out, "is_inside_2d", np.array(rets))
    put(out, "area_2d", np.array(poly_store))
    put(out, "area_2d_list", float(libroom.area_2d_polygon([[0, 4, 4, 0], [0, 0, 4, 4]])))
    put(out, "area_2d_cw", float(libroom.area_2d_polygon([[0, 0, 4, 4], [0, 4, 4, 0]])))
    put_type(out, "area_2d", libroom.area_2d_polygon([[0, 4, 4, 0], [0, 0, 4, 4]]))
    # cos_angle_between and dist_line_point
    ca2 = [libroom.cos_angle_between(*rng.standard_normal((2, 2)).astype(np.float32)) for _ in range(N)]
    ca3 = [libroom.cos_angle_between(*rng.standard_normal((2, 3)).astype(np.float32)) for _ in range(N)]
    put(out, "cos_angle_2d", np.array(ca2))
    put(out, "cos_angle_3d", np.array(ca3))
    put(out, "cos_angle_par", np.array([libroom.cos_angle_between([1.0, 1.0, 0.0], [-1.0, -1.0, 0.0]), libroom.cos_angle_between([2.0, 0.0], [3.0, 0.0])]))
    dl2 = [libroom.dist_line_point(*rng.standard_normal((3, 2)).astype(np.float32)) for _ in range(N)]
    dl3 = [libroom.dist_line_point(*rng.standard_normal((3, 3)).astype(np.float32)) for _ in range(N)]
    put(out, "dist_line_point_2d", np.array(dl2))
    put(out, "dist_line_point_3d", np.array(dl3))
    put(out, "dist_line_point_on", float(libroom.dist_line_point([0.0, 0, 0], [1.0, 1, 1], [2.0, 2, 2])))
    put_type(out, "cos_angle", ca2[0])
    put_type(out, "dist_line_point", dl2[0])
    return out


@scenario("lowlevel")
def ll_eps():
    out = {}
    put(out, "eps_default", libroom.get_eps())
    put_type(out, "eps", libroom.get_eps())
    rng = np.random.default_rng(811)
    w = libroom.Wall(np.array([[0, 4, 4, 0], [0, 0, 4, 4], [0, 0, 0, 0]], np.float32))
    w2 = libroom.Wall2D(np.array([[0, 4], [0, 0]], np.float32))
    offs = np.array([0, 1e-7, 1e-6, 5e-6, 1e-5, 2e-5, 1e-4, 1e-3, 1e-2], np.float32)
    for eps in [1e-5, 1e-3, 1e-7, 0.0, 0.1, 1e-5 + 1e-9]:
        libroom.set_eps(eps)
        tag = repr(float(eps))
        put(out, f"get_eps_{tag}", libroom.get_eps())
        rows = []
        for o in offs:
            p = np.array([2, 2, o], np.float32)
            loc = np.full(3, np.nan, np.float32)
            r = w.intersection([2.0, 2.0, 1.0], p, loc)
            loc2 = np.full(3, np.nan, np.float32)
            rows.append([w.side(p), w.side(-p), r, w.reflect(p, loc2), w2.side([1.0, o]), w2.side([4.0 + o, 0.0])])
            # boundary detection
            loc3 = np.full(3, np.nan, np.float32)
            rows[-1].append(w.intersection([o, 2.0, 1.0], [o, 2.0, -1.0], loc3))
            rows[-1].append(libroom.is_inside_2d_polygon(np.array([o, 2.0], np.float32), np.array([[0, 4, 4, 0], [0, 0, 4, 4]], np.float32)))
            loc4 = np.full(2, np.nan, np.float32)
            rows[-1].append(libroom.intersection_2d_segments([o, -1.0], [o, 1.0], [0.0, 0.0], [2.0, 0.0], loc4))
        put(out, f"eps_rows_{tag}", np.array(rows))
        # random near-degenerate
        pts = rng.uniform(-1e-3, 1e-3, size=(200, 3)).astype(np.float32) + np.array([2, 2, 0], np.float32)
        put(out, f"eps_side_rand_{tag}", np.array([w.side(p) for p in pts]))
        # wall constructor planarity check depends on eps
        c = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1e-4]], np.float32)
        record_error(out, f"nonplanar_eps_{tag}", lambda: libroom.Wall(c).normal)
    libroom.set_eps(1e-5)
    return out


# ---------------------------------------------------------------------------
# Low-level RIR builders
# ---------------------------------------------------------------------------


@scenario("rir_builder")
def ll_rir_builder():
    rng = np.random.default_rng(820)
    out = {}
    fs = 16000
    for dtype in [np.float32, np.float64]:
        dn = np.dtype(dtype).name
        for fdl in [81, 61, 17]:
            for lut in [20, 7]:
                n = 3000
                t = rng.uniform((fdl + 1) / fs, 0.25, n).astype(dtype)
                t[:20] = (np.arange(20) + fdl) / fs  # integer sample positions
                a = rng.standard_normal(n).astype(dtype)
                L = int(np.ceil(0.25 * fs)) + fdl + 1
                for nt in [1, 3, 8, 64]:
                    if lut == 7 and nt not in (1, 8):
                        continue
                    rir = np.zeros(L, dtype=dtype)
                    libroom.rir_builder(rir, t, a, fs, fdl, lut, nt)
                    put(out, f"rir_{dn}_fdl{fdl}_lut{lut}_nt{nt}", rir)
    # mixed dtypes: rir float64, time/alpha float32 (forcecast)
    t = rng.uniform(0.01, 0.1, 500).astype(np.float32)
    a = rng.standard_normal(500).astype(np.float32)
    rir = np.zeros(int(0.1 * fs) + 100)
    libroom.rir_builder(rir, t, a, fs, 81, 20, 2)
    put(out, "rir_mixed_f64_f32", rir)
    rir = np.zeros(int(0.1 * fs) + 100, dtype=np.float32)
    libroom.rir_builder(rir, t.astype(np.float64), a.astype(np.float64), fs, 81, 20, 2)
    put(out, "rir_mixed_f32_f64", rir)
    # accumulates into a non-zero rir
    rir = np.ones(int(0.1 * fs) + 100)
    libroom.rir_builder(rir, t.astype(np.float64), a.astype(np.float64), fs, 81, 20, 3)
    put(out, "rir_accumulate", rir)
    # more threads than times
    rir = np.zeros(400)
    libroom.rir_builder(rir, np.array([0.01, 0.012]), np.array([1.0, -0.5]), fs, 81, 20, 8)
    put(out, "rir_more_threads_than_times", rir)
    # other fs
    for fs2 in [8000, 44100]:
        t = rng.uniform(0.005, 0.05, 800)
        a = rng.standard_normal(800)
        rir = np.zeros(int(0.05 * fs2) + 100)
        libroom.rir_builder(rir, t, a, fs2, 81, 20, 4)
        put(out, f"rir_fs{fs2}", rir)
    # silently ignored cases (wrong dtype / non contiguous output)
    rir_i = np.zeros(400, dtype=np.int32)
    record_error(out, "rir_int_dtype", lambda: libroom.rir_builder(rir_i, np.array([0.01]), np.array([1.0]), fs, 81, 20, 1))
    put(out, "rir_int_dtype_out", rir_i)
    rir_nc = np.zeros((400, 2))[:, 0]
    record_error(out, "rir_noncontig", lambda: libroom.rir_builder(rir_nc, np.array([0.01]), np.array([1.0]), fs, 81, 20, 1))
    put(out, "rir_noncontig_out", rir_nc)
    rir_32 = np.zeros(400, dtype=np.float32)
    record_error(out, "rir_list_time", lambda: libroom.rir_builder(rir_32, [0.01, 0.015], [1.0, 0.5], fs, 81, 20, 1))
    put(out, "rir_list_time_out", rir_32)
    return out


@scenario("rir_builder")
def ll_fractional_delay():
    rng = np.random.default_rng(821)
    out = {}
    for dtype in [np.float32, np.float64]:
        dn = np.dtype(dtype).name
        for fdl in [81, 61, 80]:  # 80 -> even: last column untouched
            for lut in [20, 11]:
                n = 2500
                t = rng.uniform(0, 1, n).astype(dtype)
                t[:5] = [0.0, 0.5, 1.0 - 1e-7, 0.25, 0.999]
                for nt in [1, 3, 8, 64]:
                    if lut == 11 and nt != 3:
                        continue
                    o = np.full((n, fdl), -7.0, dtype=dtype)
                    libroom.fractional_delay(o, t, lut, nt)
                    put(out, f"fd_{dn}_fdl{fdl}_lut{lut}_nt{nt}", o)
    # compare with pra helper (python)
    put(out, "pra_fractional_delay", np.asarray(pra.fractional_delay(0.3)))
    put(out, "pra_fractional_delay_arr", np.asarray(pra.fractional_delay(np.array([0.1, 0.7]))))
    return out


@scenario("rir_builder")
def ll_delay_sum():
    rng = np.random.default_rng(822)
    out = {}
    for dtype in [np.float32, np.float64]:
        dn = np.dtype(dtype).name
        for taps in [81, 61]:
            n = 3000
            irs = rng.standard_normal((n, taps)).astype(dtype)
            d = rng.integers(0, 4000, n).astype(np.int32)
            d[:10] = 0
            L = int(d.max()) + taps + 5
            for nt in [1, 3, 8, 64]:
                o = np.zeros(L, dtype=dtype)
                libroom.delay_sum(irs, d, o, nt)
                put(out, f"ds_{dn}_taps{taps}_nt{nt}", o)
    # int64 delays (forcecast), accumulation into non-zero output
    irs = rng.standard_normal((100, 21))
    d = rng.integers(0, 300, 100)
    o = np.full(330, 0.5)
    libroom.delay_sum(irs, d, o, 2)
    put(out, "ds_int64_delays_accum", o)
    # f32 irs into f64 output -> forcecast copy (output not modified?)
    o2 = np.zeros(330)
    record_error(out, "ds_mixed", lambda: libroom.delay_sum(irs.astype(np.float32), d.astype(np.int32), o2, 2))
    put(out, "ds_mixed_out", o2)
    return out


# ---------------------------------------------------------------------------
# Low-level microphones / histograms
# ---------------------------------------------------------------------------


def _mic_fuzz(out, MicCls, dim, rng, prefix):
    n_bands = 3
    loc = rng.uniform(1, 3, dim)
    mic = MicCls(loc, n_bands, 0.02, 2.0)
    put(out, f"{prefix}_omni_nhist", len(mic.histograms))
    put(out, f"{prefix}_omni_hist0_init", mic.histograms[0].get_hist(), lay=True)
    put(out, f"{prefix}_loc", mic.loc, lay=True)
    for k in range(400):
        dist = float(rng.uniform(0, 5.0))  # beyond max_dist_init -> growth
        en = rng.uniform(0, 1, n_bands).astype(np.float32)
        origin = rng.uniform(-2, 5, dim)
        mic.log_histogram(dist, en, origin)
    put(out, f"{prefix}_omni_hist", mic.histograms[0].get_hist())
    put(out, f"{prefix}_omni_counts", mic.histograms[0].get_counts())
    # directional
    nd = 25
    dirs = rng.standard_normal((nd, dim))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    mic.set_directions(dirs.astype(np.float32))
    put(out, f"{prefix}_dir_nhist", len(mic.histograms))
    origins = loc[None, :] + rng.standard_normal((600, dim))
    # exact ties: origin direction equidistant between two direction vectors
    for j in range(10):
        bis = dirs[j] + dirs[j + 1]
        origins[j] = loc + bis / np.linalg.norm(bis)
        origins[10 + j] = loc + dirs[j]  # exactly a direction
    for k, o in enumerate(origins):
        dist = float(rng.uniform(0, 3.0))
        en = rng.uniform(0, 1, n_bands).astype(np.float32)
        mic.log_histogram(dist, en, o)
    # each direction's histogram grows independently -> ragged, one key per direction
    for j, h in enumerate(mic.histograms):
        put(out, f"{prefix}_dir_hist_d{j}", h.get_hist())
        put(out, f"{prefix}_dir_counts_d{j}", h.get_counts())
    # per-origin direction bin, recovered by logging into a fresh histogram one by one
    bins = []
    mic2 = MicCls(loc, 1, 0.5, 1.0)
    mic2.set_directions(dirs.astype(np.float32))
    for o in origins[:80]:
        for h in mic2.histograms:
            h.reset()
        mic2.log_histogram(0.1, np.ones(1, np.float32), o)
        bins.append(int(np.argmax([h.get_counts().sum() for h in mic2.histograms])))
    put(out, f"{prefix}_dir_bins", np.array(bins))
    mic.make_omnidirectional()
    put(out, f"{prefix}_after_omni_nhist", len(mic.histograms))
    put(out, f"{prefix}_after_omni_hist", mic.histograms[0].get_hist())


@scenario("lowlevel")
def ll_microphone():
    rng = np.random.default_rng(830)
    out = {}
    _mic_fuzz(out, libroom.Microphone, 3, rng, "mic3d")
    _mic_fuzz(out, libroom.Microphone2D, 2, rng, "mic2d")
    return out


@scenario("lowlevel")
def ll_histogram2d():
    rng = np.random.default_rng(831)
    out = {}
    h = libroom.Histogram2D(3, 10)
    put(out, "init_hist", h.get_hist(), lay=True)
    put(out, "init_counts", h.get_counts(), lay=True)
    for k in range(2000):
        r = int(rng.integers(0, 7))  # rows beyond 3 -> doubling growth
        c = int(rng.integers(0, 45))
        h.log(r, c, float(rng.standard_normal()))
    hist = h.get_hist()
    cnt = h.get_counts()
    put(out, "hist", hist)
    put(out, "counts", cnt)
    put(out, "bins", np.array([[h.bin(r, c) for c in range(hist.shape[1])] for r in range(hist.shape[0])]))
    put_type(out, "bin", h.bin(0, 0))
    h.log(20, 100, 1.5)
    put(out, "grown_shape", np.array(h.get_hist().shape))
    h.reset()
    put(out, "reset_hist", h.get_hist())
    put(out, "reset_counts", h.get_counts())
    h.log(0, 0, 2.5)
    put(out, "after_reset_bin", h.bin(0, 0))
    # get_hist returns a copy?
    g = h.get_hist()
    g[0, 0] = 99.0
    put(out, "get_hist_is_copy", float(h.get_hist()[0, 0]))
    # hit objects
    hit = libroom.Hit(4)
    put(out, "hit_distance", hit.distance)
    put(out, "hit_transmitted", hit.transmitted, lay=True)
    hit2 = libroom.Hit(1.5, np.array([0.1, 0.2], np.float32))
    put(out, "hit2_distance", hit2.distance)
    put(out, "hit2_transmitted", hit2.transmitted, lay=True)
    return out


# ---------------------------------------------------------------------------
# Low-level Room engine
# ---------------------------------------------------------------------------


def _ll_room3d(n_bands=1, scatter=0.0, lshape=True, hybrid=True, ism_order=3):
    if lshape:
        floor = LSHAPE
    else:
        floor = PENTAGON
    room = pra.Room.from_corners(floor, materials=pra.Material(0.2))
    room.extrude(2.8)
    walls = []
    for i, w in enumerate(room.walls):
        ab = np.linspace(0.1, 0.4, n_bands).astype(np.float32) + 0.01 * i
        sc = np.full(n_bands, scatter, np.float32)
        walls.append(libroom.Wall(w.corners, ab, sc, w.name))
    obst = pra.room.find_non_convex_walls(walls)
    eng = libroom.Room(walls, obst, [], 343.0, ism_order, 1e-7, 1.0, 0.3, 0.004, hybrid)
    return eng, walls


@scenario("lowlevel")
def ll_room3d_engine():
    pra.random.seed(numpy=840, libroom=1840)
    rng = np.random.default_rng(840)
    out = {}
    eng, walls = _ll_room3d(n_bands=1, scatter=0.0)
    put(out, "obstructing", np.array(eng.obstructing_walls))
    put(out, "max_distance", eng.get_max_distance())
    put_type(out, "max_distance", eng.get_max_distance())
    # contains: random points (consumes libroom rng)
    pts = rng.uniform(-1, 7, size=(300, 3)).astype(np.float32)
    pts[:20, 2] = 1.0
    # NOTE: points lying exactly on a wall make the reference Room::contains()
    # loop forever (the ambiguous-intersection retry never terminates), so
    # they are avoided here.
    pts[20:30] = np.c_[[2.5 + 1e-3] * 10, np.linspace(0.1, 4.9, 10), [1.4] * 10]  # next to the reentrant wall
    put(out, "contains", np.array([eng.contains(p) for p in pts]))
    put_type(out, "contains", eng.contains(pts[0]))
    # next_wall_hit
    res_p, res_w, res_d = [], [], []
    for k in range(300):
        s = rng.uniform(0, 5, 3).astype(np.float32)
        s[2] = rng.uniform(0, 2.8)
        e = s + rng.standard_normal(3).astype(np.float32) * 8
        for scat in (False, True):
            p, wi, d = eng.next_wall_hit(s, e, scat)
            res_p.append(p)
            res_w.append(wi)
            res_d.append(d)
    # Room::next_wall_hit returns an *uninitialized* point when no wall is hit
    # (index -1), so the point is masked with NaN in that case.
    res_p = np.array(res_p)
    res_p[np.array(res_w) < 0] = np.nan
    put(out, "nwh_point", res_p)
    put(out, "nwh_wall", np.array(res_w))
    put(out, "nwh_dist", np.array(res_d))
    out["nwh_point__layout"] = layout(res_p[0])
    put_type(out, "nwh_dist", res_d[0])
    # image source model with mics
    for m in MICS_LS3D.T:
        eng.add_mic(m.astype(np.float32))
    for s, src in enumerate(SRC_LS3D):
        n = eng.image_source_model(np.array(src, np.float32))
        put(out, f"ism_n_s{s}", n)
        for attr in ["sources", "orders", "orders_xyz", "gen_walls", "attenuations", "visible_mics", "source_directions"]:
            put(out, f"ism_s{s}_{attr}", getattr(eng, attr), lay=(s == 0))
    # simul_ray both overloads
    src = np.array(SRC_LS3D[0], np.float32)
    for k in range(40):
        eng.simul_ray(float(rng.uniform(0, 2 * np.pi)), float(rng.uniform(0, np.pi)), src, np.ones(1, np.float32) * 1000)
    d = rng.standard_normal((40, 3)).astype(np.float32)
    for v in d:
        eng.simul_ray(v / np.linalg.norm(v), src, np.ones(1, np.float32) * 500)
    for i, mic in enumerate(eng.microphones):
        put(out, f"simul_ray_hist_m{i}", mic.histograms[0].get_hist())
        put(out, f"simul_ray_counts_m{i}", mic.histograms[0].get_counts())
    eng.reset_mics()
    put(out, "after_reset_hist_m0", eng.microphones[0].histograms[0].get_hist())
    # scat_ray
    sr = []
    for k in range(50):
        wi = int(rng.integers(len(walls)))
        hp = walls[wi].corners.mean(axis=1).astype(np.float32)
        prev = rng.uniform(0.5, 4.0, 3).astype(np.float32)
        sr.append(eng.scat_ray(np.ones(1, np.float32) * 100.0, eng.get_wall(wi), prev, hp, float(rng.uniform(0, 5))))
    put(out, "scat_ray_ret", np.array(sr))
    put(out, "scat_ray_hist_m0", eng.microphones[0].histograms[0].get_hist())
    # broadcast bands
    eng.reset_mics()
    eng.broadcast_bands_to(7)
    put(out, "bcast_absorption_w0", eng.walls[0].absorption, lay=True)
    put(out, "bcast_scatter_w0", eng.walls[0].scatter)
    put(out, "bcast_hist_shape", np.array(eng.microphones[0].histograms[0].get_hist().shape))
    eng.broadcast_bands_to(1)
    put(out, "bcast1_absorption_w0", eng.walls[0].absorption)
    put(out, "bcast1_hist_shape", np.array(eng.microphones[0].histograms[0].get_hist().shape))
    # set_params / is_hybrid_sim property
    eng.set_params(340.0, 2, 1e-6, 0.5, 0.4, 0.002, False)
    put(out, "is_hybrid_after_set_params", eng.is_hybrid_sim)
    eng.is_hybrid_sim = True
    put(out, "is_hybrid_after_set", eng.is_hybrid_sim)
    n = eng.image_source_model(src)
    put(out, "ism_after_set_params_n", n)
    put(out, "ism_after_set_params_att", eng.attenuations)
    # live references: walls returned by .walls/.get_wall mutate the engine
    eng.walls[0].absorption = np.array([0.77], np.float32)
    put(out, "live_walls_absorption", eng.walls[0].absorption)
    # get_wall() returns by value (pybind 'copy' policy) UNLESS a python wrapper
    # for that wall already exists (e.g. a held reference from .walls), in which
    # case pybind returns the existing (live) instance.
    eng.get_wall(1).name = "renamed"
    put(out, "get_wall_no_wrapper_name", eng.walls[1].name)
    held = eng.walls
    eng.get_wall(1).name = "renamed"
    put(out, "get_wall_wrapper_held_name", eng.walls[1].name)
    put(out, "get_wall_is_held", eng.get_wall(1) is held[1])
    del held
    # python Room objects: ShoeBox.walls are the engine's walls, from_corners' are separate copies
    sb = pra.ShoeBox([4, 3, 2.5])
    put(out, "shoebox_walls_are_engine_walls", sb.walls[0] is sb.room_engine.walls[0])
    sb.walls[0].absorption = np.array([0.42], np.float32)
    put(out, "shoebox_walls_mutation_seen", sb.room_engine.walls[0].absorption)
    pc = pra.Room.from_corners(np.array([[0, 4, 4, 0], [0, 0, 3, 3]]))
    put(out, "from_corners_walls_are_engine_walls", pc.walls[0] is pc.room_engine.walls[0])
    pc.walls[0].absorption = np.array([0.42], np.float32)
    put(out, "from_corners_walls_mutation_seen", pc.room_engine.walls[0].absorption)
    put(out, "walls_list_identity", eng.walls[0] is eng.walls[0])
    put(out, "mics_list_identity", eng.microphones[0] is eng.microphones[0])
    put(out, "sources_base_is_engine", eng.sources.base is eng)
    put(out, "room_dim", libroom.Room.dim)
    put(out, "room2d_dim", libroom.Room2D.dim)
    return out


MICS_LS3D = np.c_[[1.2, 4.0, 1.3], [5.0, 1.0, 1.5], [1.0, 1.2, 2.0]]
SRC_LS3D = [[4.5, 1.5, 1.1], [1.2, 3.8, 1.9]]


@scenario("lowlevel")
def ll_room3d_ray_tracing_overloads():
    pra.random.seed(numpy=841, libroom=1841)
    rng = np.random.default_rng(841)
    out = {}
    for nb, sc in [(1, 0.0), (1, 0.2), (3, 0.1)]:
        eng, walls = _ll_room3d(n_bands=nb, scatter=sc)
        eng.set_params(343.0, 2, 1e-6, 0.6, 0.4, 0.004, True)
        for m in MICS_LS3D.T:
            eng.add_mic(m.astype(np.float32))
        src = np.array(SRC_LS3D[0], np.float32)
        tag = f"nb{nb}_sc{sc}"

        def grab(name):
            for i, mic in enumerate(eng.microphones):
                put(out, f"{tag}_{name}_m{i}", mic.histograms[0].get_hist())
                put(out, f"{tag}_{name}_cnt_m{i}", mic.histograms[0].get_counts())
            eng.reset_mics()

        libroom.set_rng_seed(11)
        angles = np.c_[rng.uniform(0, 2 * np.pi, 800), rng.uniform(0, np.pi, 800)].T.astype(np.float32)
        eng.ray_tracing(angles, src)
        grab("angles")
        libroom.set_rng_seed(12)
        vecs = rng.standard_normal((800, 3)).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        en = rng.uniform(0.5, 1.5, (800, nb)).astype(np.float32)
        eng.ray_tracing(vecs, en, src)
        grab("vectors")
        libroom.set_rng_seed(13)
        eng.ray_tracing(30, 25, src)
        grab("phis_thetas")
        libroom.set_rng_seed(14)
        eng.ray_tracing(900, src)
        grab("nrays")
        # directional receivers
        dirs = rng.standard_normal((12, 3)).astype(np.float32)
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        eng.microphones[0].set_directions(dirs)
        libroom.set_rng_seed(15)
        eng.ray_tracing(700, src)
        for j, h in enumerate(eng.microphones[0].histograms):
            put(out, f"{tag}_directional_m0_d{j}", h.get_hist())
        eng.microphones[0].make_omnidirectional()
        eng.reset_mics()
    return out


def _ll_room2d(n_bands=1, scatter=0.0):
    walls = []
    nc = LSHAPE.shape[1]
    for i in range(nc):
        c = np.c_[LSHAPE[:, i], LSHAPE[:, (i + 1) % nc]].astype(np.float32)
        ab = np.linspace(0.1, 0.4, n_bands).astype(np.float32) + 0.01 * i
        walls.append(libroom.Wall2D(c, ab, np.full(n_bands, scatter, np.float32), f"w{i}"))
    obst = pra.room.find_non_convex_walls(walls)
    eng = libroom.Room2D(walls, obst, [], 343.0, 3, 1e-7, 1.0, 0.3, 0.004, True)
    return eng, walls


@scenario("lowlevel")
def ll_room2d_engine():
    pra.random.seed(numpy=842, libroom=1842)
    rng = np.random.default_rng(842)
    out = {}
    eng, walls = _ll_room2d()
    put(out, "obstructing", np.array(eng.obstructing_walls))
    put(out, "max_distance", eng.get_max_distance())
    pts = rng.uniform(-1, 7, size=(300, 2)).astype(np.float32)
    pts[:10] = np.c_[[2.5 - 1e-3] * 10, np.linspace(0.1, 4.9, 10)]  # next to (not on) the reentrant wall
    put(out, "contains", np.array([eng.contains(p) for p in pts]))
    res_p, res_w, res_d = [], [], []
    for k in range(300):
        s = rng.uniform(0, 5, 2).astype(np.float32)
        e = s + rng.standard_normal(2).astype(np.float32) * 8
        p, wi, d = eng.next_wall_hit(s, e, bool(k % 2))
        res_p.append(p)
        res_w.append(wi)
        res_d.append(d)
    # Room::next_wall_hit returns an *uninitialized* point when no wall is hit
    # (index -1), so the point is masked with NaN in that case.
    res_p = np.array(res_p)
    res_p[np.array(res_w) < 0] = np.nan
    put(out, "nwh_point", res_p)
    put(out, "nwh_wall", np.array(res_w))
    put(out, "nwh_dist", np.array(res_d))
    for m in LSHAPE_MICS2D.T:
        eng.add_mic(m.astype(np.float32))
    for s, src in enumerate(LSHAPE_SRCS2D):
        n = eng.image_source_model(np.array(src, np.float32))
        put(out, f"ism_n_s{s}", n)
        for attr in ["sources", "orders", "orders_xyz", "gen_walls", "attenuations", "visible_mics", "source_directions"]:
            put(out, f"ism_s{s}_{attr}", getattr(eng, attr), lay=(s == 0))
    src = np.array(LSHAPE_SRCS2D[0], np.float32)
    eng.set_params(343.0, 2, 1e-6, 0.6, 0.4, 0.004, True)
    eng.reset_mics()
    eng.ray_tracing(np.array([rng.uniform(0, 2 * np.pi, 500)], np.float32), src)
    put(out, "rt_angles_m0", eng.microphones[0].histograms[0].get_hist())
    eng.reset_mics()
    v = rng.standard_normal((500, 2)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    eng.ray_tracing(v, np.ones((500, 1), np.float32), src)
    put(out, "rt_vectors_m0", eng.microphones[0].histograms[0].get_hist())
    eng.reset_mics()
    eng.ray_tracing(64, 5, src)
    put(out, "rt_phis_m0", eng.microphones[0].histograms[0].get_hist())
    eng.reset_mics()
    eng.ray_tracing(600, src)
    put(out, "rt_nrays_m0", eng.microphones[0].histograms[0].get_hist())
    eng.reset_mics()
    eng.simul_ray(0.3, 1.0, src, np.ones(1, np.float32))
    eng.simul_ray(np.array([0.6, 0.8], np.float32), src, np.ones(1, np.float32))
    put(out, "simul_ray_m1", eng.microphones[1].histograms[0].get_hist())
    eng.broadcast_bands_to(4)
    put(out, "bcast_abs_w0", eng.walls[0].absorption)
    return out


@scenario("lowlevel")
def ll_room_shoebox_ctor():
    pra.random.seed(numpy=843, libroom=1843)
    rng = np.random.default_rng(843)
    out = {}
    for nb in [1, 7]:
        ab = rng.uniform(0.05, 0.6, (nb, 6)).astype(np.float32)
        sc = rng.uniform(0.0, 0.3, (nb, 6)).astype(np.float32)
        eng = libroom.Room(np.array(DIM3D, np.float32), ab, sc, [], 343.0, 4, 1e-7, 1.0, 0.3, 0.004, True)
        for m in MICS3D.T:
            eng.add_mic(m.astype(np.float32))
        capture_engine(out, eng, prefix=f"sb{nb}_pre")
        n = eng.image_source_model(np.array(SRCS3D[0], np.float32))
        put(out, f"sb{nb}_n", n)
        capture_engine(out, eng, prefix=f"sb{nb}")
        put(out, f"sb{nb}_contains", np.array([eng.contains(p) for p in rng.uniform(-1, 6, (100, 3)).astype(np.float32)]))
        eng.set_params(343.0, 4, 1e-7, 0.5, 0.3, 0.004, True)
        eng.reset_mics()
        eng.ray_tracing(1500, np.array(SRCS3D[1], np.float32))
        put(out, f"sb{nb}_rt_m0", eng.microphones[0].histograms[0].get_hist())
    ab2 = rng.uniform(0.05, 0.6, (1, 4)).astype(np.float32)
    eng2 = libroom.Room2D(np.array(DIM2D, np.float32), ab2, np.zeros((1, 4), np.float32), [], 343.0, 5, 1e-7, 1.0, 0.3, 0.004, False)
    for m in MICS2D.T:
        eng2.add_mic(m.astype(np.float32))
    n = eng2.image_source_model(np.array(SRCS2D[0], np.float32))
    put(out, "sb2d_n", n)
    capture_engine(out, eng2, prefix="sb2d")
    return out


# ---------------------------------------------------------------------------
# Error behaviour
# ---------------------------------------------------------------------------


@scenario("errors")
def errors():
    pra.random.seed(numpy=900, libroom=1900)
    out = {}
    sq = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]], np.float32)
    record_error(out, "wall_nonplanar", lambda: libroom.Wall(np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1]], np.float32)))
    record_error(out, "wall_band_mismatch", lambda: libroom.Wall(sq, np.array([0.1, 0.2], np.float32), np.array([0.1], np.float32)))
    record_error(out, "wall2d_band_mismatch", lambda: libroom.Wall2D(np.array([[0, 1], [0, 0]], np.float32), np.array([0.1, 0.2], np.float32), np.array([0.1, 0.1, 0.2], np.float32)))
    record_error(out, "wall_wrong_dim", lambda: libroom.Wall(np.zeros((2, 4), np.float32)))
    record_error(out, "wall_intersection_f64_out", lambda: libroom.Wall(sq).intersection([0.5, 0.5, 1.0], [0.5, 0.5, -1.0], np.zeros(3)))
    loc = np.zeros(3, np.float32)
    r = record_error(out, "wall_intersection_list_out", lambda: libroom.Wall(sq).intersection([0.5, 0.5, 1.0], [0.5, 0.5, -1.0], [0.0, 0.0, 0.0]))
    out["wall_intersection_list_out_ret"] = "__None__" if r is None else int(r)
    record_error(out, "ccw3p_3d", lambda: libroom.ccw3p([0, 0, 0], [1, 0, 0], [0, 1, 0]))

    def bcast():
        ab = np.full((2, 6), 0.2, np.float32)
        eng = libroom.Room(np.array([4, 3, 2.5], np.float32), ab, 0 * ab, [], 343.0, 1, 1e-7, 1.0, 0.3, 0.004, False)
        eng.broadcast_bands_to(3)

    record_error(out, "room_too_few_walls", lambda: libroom.Room([libroom.Wall(sq)], [], [], 343.0, 1, 1e-7, 1.0, 0.3, 0.004, False))

    record_error(out, "broadcast_incompatible", bcast)

    def mixed_bands_room():
        room = pra.ShoeBox([4, 3, 2.5], materials={"ceiling": pra.Material("hard_surface"), "floor": pra.Material(0.2), "east": pra.Material(0.2), "west": pra.Material(0.2), "north": pra.Material(0.2), "south": pra.Material(0.2)})
        room.add_source([1, 1, 1])
        room.add_microphone([2, 2, 1])
        room.compute_rir()
        return room.rir[0][0]

    rir = record_error(out, "mixed_bands_shoebox", mixed_bands_room)
    if rir is not None:
        put(out, "mixed_bands_shoebox_rir", rir)
    fs = 16000
    record_error(out, "rir_too_small", lambda: libroom.rir_builder(np.zeros(100, np.float32), np.array([0.1], np.float32), np.array([1.0], np.float32), fs, 81, 20, 2))
    record_error(out, "rir_negative_time", lambda: libroom.rir_builder(np.zeros(100, np.float32), np.array([0.0], np.float32), np.array([1.0], np.float32), fs, 81, 20, 2))
    record_error(out, "rir_even_fdl", lambda: libroom.rir_builder(np.zeros(1000), np.array([0.02]), np.array([1.0]), fs, 80, 20, 2))
    record_error(out, "rir_size_mismatch", lambda: libroom.rir_builder(np.zeros(1000), np.array([0.02, 0.03]), np.array([1.0]), fs, 81, 20, 2))
    record_error(out, "ds_too_small", lambda: libroom.delay_sum(np.ones((3, 10)), np.array([0, 5, 95], np.int32), np.zeros(100), 2))
    record_error(out, "ds_negative", lambda: libroom.delay_sum(np.ones((3, 10)), np.array([0, -5, 9], np.int32), np.zeros(100), 2))
    record_error(out, "ds_mismatch", lambda: libroom.delay_sum(np.ones((3, 10)), np.array([0, 5], np.int32), np.zeros(100), 2))
    record_error(out, "fd_not_2d", lambda: libroom.fractional_delay(np.zeros(81), np.array([0.5]), 20, 1))
    record_error(out, "fd_mismatch", lambda: libroom.fractional_delay(np.zeros((3, 81)), np.array([0.5, 0.1]), 20, 1))
    record_error(out, "material_bad_name", lambda: pra.Material("not_a_material"))

    def no_source():
        room = pra.ShoeBox([4, 3, 2.5])
        room.add_microphone([2, 2, 1])
        room.simulate()

    record_error(out, "simulate_no_source", no_source)

    def no_mic():
        room = pra.ShoeBox([4, 3, 2.5])
        room.add_source([1, 1, 1])
        room.compute_rir()

    record_error(out, "compute_rir_no_mic", no_mic)

    def src_outside():
        room = pra.ShoeBox([4, 3, 2.5])
        room.add_source([5, 1, 1])

    record_error(out, "source_outside", src_outside)

    def rt_2d_directive():
        room = pra.ShoeBox([4, 3], ray_tracing=True)
        room.set_ray_tracing(n_rays=100)
        room.add_source([1, 1], directivity=Cardioid(DirectionVector(0, 90)))
        room.add_microphone([2, 2])
        room.compute_rir()

    record_error(out, "rt_2d_directive", rt_2d_directive)

    def t0_nonzero():
        room = pra.ShoeBox([4, 3, 2.5], t0=0.01)
        room.add_source([1, 1, 1])
        room.add_microphone([2, 2, 1])
        room.compute_rir()

    record_error(out, "t0_nonzero", t0_nonzero)

    def mic_dir_with_rt():
        room = pra.ShoeBox([4, 3, 2.5], ray_tracing=True)
        room.add_microphone([2, 2, 1], directivity=Cardioid(DirectionVector(0, 90)))

    record_error(out, "mic_directivity_with_rt", mic_dir_with_rt)
    record_error(out, "room_ctor_bad_args", lambda: libroom.Room([], [], [], 343.0))
    record_error(out, "histogram_bad_args", lambda: libroom.Histogram2D(1.5, "a"))

    def mic_outside():
        room = pra.ShoeBox([4, 3, 2.5], max_order=2)
        room.add_source([1, 1, 1])
        room.add_microphone([5, 2, 1])
        room.compute_rir()
        return room

    room = record_error(out, "mic_outside", mic_outside)
    if room is not None:
        put(out, "mic_outside_rir", room.rir[0][0])
        put(out, "mic_outside_vis", room.visibility[0])
    return out


def list_scenarios():
    return [(n, c) for n, (c, _) in SCENARIOS.items()]
