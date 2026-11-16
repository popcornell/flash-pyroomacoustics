"""
GPU (CUDA) backend based on PyTorch.

It computes the image source part of the RIRs of shoebox rooms, and the
direct-path (anechoic) RIRs of any room, on the GPU. The arithmetic follows
the CPU implementation (same image sources, attenuations, air absorption,
windowed-sinc fractional delays, octave filter bank and time origin), so the
results agree with the CPU backend up to float32 rounding.

Use it by setting the backend globally::

    pra.constants.set("backend", "cuda")      # or PRA_BACKEND=cuda
    pra.constants.set("cuda_device", "cuda:1")  # or PRA_CUDA_DEVICE=cuda:1

or per call: ``room.compute_rir(backend="cuda")``.

Ray tracing of 3D shoebox rooms and the synthesis of the ray tracing tail
also run on the GPU (see :py:mod:`pyroomacoustics.cuda_rt`), and
:py:func:`compute_rirs` traces many rooms in a single launch. Rooms (or parts)
that the GPU path does not support (polygonal rooms, 2D ray tracing,
directional microphones, directional sources for the ISM, exact mode) fall
back to the CPU backend with a warning. Low-order ISM (below the
``cuda_min_ism_work`` constant) runs on the CPU, where it is faster.
"""

import math
import warnings

import numpy as np

from .parameters import constants

_tables = {}
_windows = {}
_warned = set()


def _torch():
    import torch

    return torch


def is_available():
    """True if PyTorch is installed and a CUDA device is visible."""
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


def _device():
    torch = _torch()
    return torch.device(constants.get("cuda_device"))


def _warn_once(msg):
    if msg not in _warned:
        _warned.add(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=3)


def unsupported_reason(room, shoebox_only=True):
    """None if the GPU path supports ``room``, else the reason."""
    if not is_available():
        return "PyTorch with CUDA is not available"
    if constants.get("exact_mode"):
        return "exact_mode is on"
    if shoebox_only and not hasattr(room, "shoebox_dim"):
        return "only shoebox rooms are supported"
    if any(d is not None for d in room.mic_array.directivity):
        return "directional microphones are not supported"
    if any(s.directivity is not None for s in room.sources):
        return "directional sources are not supported"
    return None


def ism_work(room):
    """Number of (image source, microphone) pairs of the shoebox ISM."""
    o = int(room.max_order)
    if o < 0:
        return 0
    n = 1 + 2 * o * (o + 1) if room.dim == 2 else 1 + 2 * o * (2 * o * o + 3 * o + 4) // 3
    return n * room.mic_array.M * len(room.sources)


def _sinc_tables(fdl, lut_gran, device):
    """Window-premultiplied interpolation tables (same as the Rust kernel)."""
    key = (fdl, lut_gran, str(device))
    if key not in _tables:
        torch = _torch()
        fdl2 = (fdl - 1) // 2
        lut_size = (fdl + 2) * lut_gran
        delta = np.float32(1.0 / lut_gran)
        n = np.float32(-fdl2 - 1)
        lut = np.zeros(lut_size, dtype=np.float32)
        pi = np.float32(np.pi)
        for i in range(lut_size):
            lut[i] = 1.0 if n == 0 else np.sin(pi * n) / (pi * n)
            n = np.float32(n + delta)
        i = np.arange(fdl, dtype=np.float32)
        win = (0.5 - 0.5 * np.cos(pi * 2 * i / np.float32(fdl - 1))).astype(np.float32)
        g = np.arange(lut_gran + 1)[:, None]
        k = np.arange(fdl)[None, :]
        a = lut[g + k * lut_gran]
        b = lut[g + 1 + k * lut_gran]
        tw = torch.from_numpy((win * a).astype(np.float32)).to(device)
        td = torch.from_numpy((win * (b - a)).astype(np.float32)).to(device)
        _tables[key] = (tw, td)
    return _tables[key]


def _squared_windows(octave_bands, n_fft, device):
    key = (id(octave_bands), n_fft, str(device))
    if key not in _windows:
        torch = _torch()
        _windows[key] = torch.from_numpy(octave_bands._squared_windows(n_fft)).to(
            device, torch.float32
        )
    return _windows[key]


_lattice_cache = {}


def _lattice(order, dim, device):
    """All integer reflection indices with |px|+|py|(+|pz|) <= order, (n, dim) int32 on device."""
    torch = _torch()
    key = (order, dim, str(device))
    if key not in _lattice_cache:
        if len(_lattice_cache) > 4:
            _lattice_cache.clear()
        r = torch.arange(-order, order + 1, device=device, dtype=torch.int32)
        if dim == 2:
            py, px = torch.meshgrid(r, r, indexing="ij")
            keep = px.abs() + py.abs() <= order
            pts = torch.stack([px[keep], py[keep]], dim=1)
        else:
            py, px = torch.meshgrid(r, r, indexing="ij")
            l1 = px.abs() + py.abs()
            blocks = []
            # z-slabs keep the temporary cube small
            zs = max(1, (1 << 24) // max(1, r.numel() ** 2))
            for z0 in range(-order, order + 1, zs):
                z = torch.arange(z0, min(z0 + zs, order + 1), device=device, dtype=torch.int32)
                keep = l1[None] + z.abs()[:, None, None] <= order
                zz = z[:, None, None].expand_as(keep)
                blocks.append(
                    torch.stack(
                        [px[None].expand_as(keep)[keep], py[None].expand_as(keep)[keep], zz[keep]],
                        dim=1,
                    )
                )
            pts = torch.cat(blocks)
        _lattice_cache[key] = pts
    return _lattice_cache[key]


def _lattice_chunks(order, dim, target, device):
    if order < 0:
        return
    pts = _lattice(order, dim, device)
    for i in range(0, pts.shape[0], target):
        yield pts[i : i + target]


def _reflection_coeffs(room):
    """(n_bands, n_walls) pressure reflection coefficients as in libroom."""
    # libroom's shoebox ISM indexes the walls as 2*axis (low) / 2*axis+1 (high)
    # in ``wall_names`` order (west, east, south, north, floor, ceiling); in 2D
    # the engine's wall list is ordered differently, so look them up by name.
    by_name = {w.name: w for w in room.walls}
    walls = [by_name[n] for n in room.wall_names]
    absorption = np.array([w.absorption for w in walls], dtype=np.float32).T
    scattering = np.array([w.scatter for w in walls], dtype=np.float32).T
    r = np.sqrt(np.float32(1.0) - absorption)
    hybrid = room.simulator_state["ism_needed"] and room.simulator_state["rt_needed"]
    if hybrid and room.max_order > 0 and scattering.max() > 0:
        r = r * np.sqrt(np.float32(1.0) - scattering)
    return r


class _Accumulator:
    """Adds windowed-sinc fractional delays of point sources to (M, N, n_bands)."""

    def __init__(self, room, mics, n_bands, device):
        torch = _torch()
        self.device = device
        self.fs = float(room.fs)
        self.c = float(room.c)
        self.fdl = int(constants.get("frac_delay_length"))
        if self.fdl % 2 == 0:
            raise ValueError("the fractional filter length should be odd")
        self.fdl2 = self.fdl // 2
        self.lut_gran = int(constants.get("sinc_lut_granularity"))
        self.tw, self.td = _sinc_tables(self.fdl, self.lut_gran, device)
        self.mics = torch.as_tensor(mics.T, dtype=torch.float64, device=device)  # (M, D)
        self.n_bands = n_bands
        air = room.air_absorption
        self.air = None
        if air is not None:
            self.air = torch.as_tensor(np.asarray(air, np.float64), device=device)
            if n_bands != self.air.shape[0]:
                if n_bands != 1:
                    raise ValueError("inconsistent number of bands")
                self.n_bands = self.air.shape[0]

    def times(self, images):
        """Arrival times (n, M) including the fractional delay offset."""
        dist = (images[:, None, :] - self.mics[None, :, :]).norm(dim=-1)
        return dist, dist / self.c + self.fdl2 / self.fs

    def n_samples(self, t_max):
        return int(math.ceil(t_max * self.fs + self.fdl2 + 1)) + 1

    def add(self, out, images, att, mic_mask=None):
        """out: (M * N, n_bands) float32; images (n, D) f64; att (n, nb) f32."""
        torch = _torch()
        M = self.mics.shape[0]
        N = out.shape[0] // M
        dist, t = self.times(images)
        amp = att[:, None, :].to(torch.float64) / dist[..., None]
        if self.air is not None:
            amp = amp * torch.exp(-0.5 * self.air * dist[..., None])
        amp = amp.to(torch.float32)
        if amp.shape[-1] != self.n_bands:
            amp = amp.expand(-1, -1, self.n_bands)
        if mic_mask is not None:
            amp = amp * mic_mask[None, :, None]
        sf = t * self.fs
        ip = torch.floor(sf)
        fp = (sf - ip).to(torch.float32)
        xof = (1.0 - fp) * self.lut_gran
        off = torch.floor(xof)
        xo = xof - off
        g = off.to(torch.long).clamp_(0, self.lut_gran)
        w = self.tw[g] + xo[..., None] * self.td[g]  # (n, M, fdl)
        k = torch.arange(self.fdl, device=self.device)
        idx = ip.to(torch.long)[..., None] - self.fdl2 + k  # (n, M, fdl)
        # by construction 0 <= idx < N (times include the fdl2 offset and N is
        # computed from the largest time); clamp only as a safety net
        idx.clamp_(0, N - 1)
        row = idx + (torch.arange(M, device=self.device) * N)[None, :, None]
        vals = w[..., None] * amp[:, :, None, :]  # (n, M, fdl, nb)
        out.index_add_(0, row.reshape(-1), vals.reshape(-1, vals.shape[-1]))


def _to_rirs(room, out, Ns, n_bands, device):
    """(M, N, nb) band signals -> list of per-mic numpy RIRs (filter bank sum)."""
    torch = _torch()
    rirs = []
    for m, Nm in enumerate(Ns):
        x = out[m, :Nm, :]
        if n_bands == 1:
            rirs.append(x[:, 0])
            continue
        n_fft = 2 ** math.ceil(math.log2(2 * Nm))
        H = _squared_windows(room.octave_bands, n_fft, device)
        X = torch.fft.rfft(x.T, n=n_fft, dim=-1)
        y = torch.fft.irfft((X * H).sum(0), n=n_fft)[:Nm]
        rirs.append(y)
    return [r.cpu().numpy().astype(np.float32) for r in rirs]


def _chunk_size(M, fdl, nb):
    # keep the (n, M, fdl, nb) tap tensor around 64M elements
    return max(256, (1 << 26) // max(1, M * fdl * nb))


def ism_rirs(room):
    """
    Image source RIRs of a shoebox room on the GPU.

    Returns ``(rirs, direct_positions)`` where ``rirs[m][s]`` is a float32
    numpy array, and ``direct_positions[s]`` is the position of the direct
    path image (differs from the source position with the randomized ISM).
    """
    torch = _torch()
    device = _device()
    R = room.mic_array.R
    M = R.shape[1]
    dim = room.dim
    order = int(room.max_order)
    L = torch.as_tensor(np.asarray(room.shoebox_dim, np.float64), device=device)
    r = torch.as_tensor(_reflection_coeffs(room), device=device)  # (nb, 2D)
    nb_walls = r.shape[0]
    acc = _Accumulator(room, R, nb_walls, device)
    nb = acc.n_bands
    inside = torch.as_tensor(
        [float(room.is_inside(R[:, m])) for m in range(M)],
        dtype=torch.float32,
        device=device,
    )
    target = _chunk_size(M, acc.fdl, nb)
    rand = room.simulator_state["random_ism_needed"]
    max_disp = float(room.max_rand_disp)

    from . import random as pra_random

    rirs = [[] for _ in range(M)]
    direct = []
    for src in room.sources:
        s = torch.as_tensor(np.asarray(src.position, np.float64), device=device)
        seed = int(pra_random.get_rng().integers(2**62)) if rand else 0
        disp0 = None
        if rand:
            gen = torch.Generator(device=device).manual_seed(seed)
            disp0 = (torch.rand(dim, generator=gen, device=device, dtype=torch.float64) * 2 - 1) * max_disp

        def images_of(ci, p):
            p = p.to(torch.int64)
            pf = p.to(torch.float64)
            odd = (p.abs() % 2) == 1
            loc = pf * L + torch.where(odd, L - s, s)
            # reflections on the "low" (2d) and "high" (2d+1) walls
            pos = p > 0
            neg = p < 0
            pa = p.abs()
            p1 = torch.where(pos, pa // 2, torch.where(neg, (pa + 1) // 2, torch.zeros_like(p)))
            p2 = torch.where(pos, (pa + 1) // 2, torch.where(neg, pa // 2, torch.zeros_like(p)))
            att = torch.ones(p.shape[0], nb_walls, device=device)
            for d in range(dim):
                att = att * torch.pow(r[:, 2 * d][None, :], p1[:, d, None].to(torch.float32))
                att = att * torch.pow(r[:, 2 * d + 1][None, :], p2[:, d, None].to(torch.float32))
            if rand:
                gen = torch.Generator(device=device).manual_seed(seed + 1 + ci)
                disp = (torch.rand(loc.shape, generator=gen, device=device, dtype=torch.float64) * 2 - 1) * max_disp
                is0 = (p == 0).all(dim=1)
                disp[is0] = disp0
                loc = loc + disp
            return loc, att

        # pass 1: per-microphone length
        t_max = torch.zeros(M, dtype=torch.float64, device=device)
        for ci, p in enumerate(_lattice_chunks(order, dim, target, device)):
            loc, _ = images_of(ci, p)
            _, t = acc.times(loc)
            t_max = torch.maximum(t_max, t.max(dim=0).values)
        t_max = t_max.cpu().numpy()
        Ns = [
            acc.n_samples(t_max[m]) if inside[m] > 0 and order >= 0 else acc.n_samples(0.0)
            for m in range(M)
        ]
        N = max(Ns)
        out = torch.zeros(M * N, nb, dtype=torch.float32, device=device)
        # pass 2: synthesis
        for ci, p in enumerate(_lattice_chunks(order, dim, target, device)):
            loc, att = images_of(ci, p)
            acc.add(out, loc, att, mic_mask=inside)
        for m, h in enumerate(_to_rirs(room, out.view(M, N, nb), Ns, nb, device)):
            rirs[m].append(h)
        pos0 = s + disp0 if rand else s
        direct.append(pos0.cpu().numpy())
    return rirs, direct


def direct_rirs(room, positions, air_absorption=True, visible=None):
    """
    Direct-path (anechoic) RIRs on the GPU, ``rirs[m][s]``, aligned with the
    image source RIRs (same delay convention, filter and filter bank).
    ``visible[s][m]`` (optional) zeroes occluded direct paths.
    """
    torch = _torch()
    device = _device()
    R = room.mic_array.R
    M = R.shape[1]
    nb_walls = max(1, len(room.walls[0].absorption)) if len(room.walls) else 1
    saved = room.air_absorption
    if not air_absorption:
        room.air_absorption = None
    try:
        acc = _Accumulator(room, R, nb_walls, device)
    finally:
        room.air_absorption = saved
    nb = acc.n_bands
    rirs = [[] for _ in range(M)]
    for si, pos in enumerate(positions):
        img = torch.as_tensor(np.asarray(pos, np.float64), device=device)[None, :]
        _, t = acc.times(img)
        Ns = [acc.n_samples(float(t[0, m])) for m in range(M)]
        N = max(Ns)
        out = torch.zeros(M * N, nb, dtype=torch.float32, device=device)
        mask = None
        if visible is not None:
            mask = torch.as_tensor(np.asarray(visible[si], np.float32), device=device)
        acc.add(out, img, torch.ones(1, nb_walls, device=device), mic_mask=mask)
        for m, h in enumerate(_to_rirs(room, out.view(M, N, nb), Ns, nb, device)):
            rirs[m].append(h)
    return rirs


def compute_rirs(rooms, anechoic=False):
    """
    Compute the RIRs of many rooms with the GPU backend.

    Ray tracing of all the (3D shoebox) rooms that need it is done in a single
    GPU launch, which keeps the GPU busy even with few rays per room. Then
    ``room.compute_rir(backend="cuda")`` (and ``compute_anechoic_rir`` if
    ``anechoic=True``) is called for every room. Unsupported rooms use the
    CPU backend, as in :py:meth:`Room.compute_rir`.
    """
    from . import cuda_rt

    batch = [
        r
        for r in rooms
        if r.simulator_state["rt_needed"]
        and not r.simulator_state["rt_done"]
        and cuda_rt.unsupported_reason(r) is None
    ]
    if batch:
        cuda_rt.ray_tracing(batch)
    # tails of all the rooms traced on the GPU, in one batch
    tail_rooms = [
        r
        for r in rooms
        if r.simulator_state["rt_needed"]
        and r.simulator_state["rt_done"]
        and cuda_rt.unsupported_reason(r) is None
    ]
    if tail_rooms:
        for r, t in zip(tail_rooms, cuda_rt.rt_tails_batch(tail_rooms)):
            r._pending_gpu_tails = t
    for r in rooms:
        r.compute_rir(backend="cuda")
        if anechoic:
            r.compute_anechoic_rir(backend="cuda")
    return rooms


_hpf_cache = {}


def highpass(rirs, fs):
    """
    The RIR high-pass filter of ``compute_rir`` (``sosfiltfilt`` of the
    ``rir_hpf_*`` design) for a list of 1D arrays, on the GPU. It applies the
    same zero-phase magnitude response ``|H(f)|^2`` in the frequency domain;
    on RIRs the result agrees with ``sosfiltfilt`` to ~1e-5 (relative L2).
    """
    import scipy.signal

    from .utilities import design_highpass_filter_sos

    torch = _torch()
    device = _device()
    if len(rirs) == 0:
        return []
    L = max(len(h) for h in rirs)
    pad = int(0.5 * fs)  # the zero-phase response has decayed after 0.5 s
    n_fft = 2 ** math.ceil(math.log2(L + 2 * pad))
    key = (fs, n_fft, constants.get("rir_hpf_fc"), str(constants.get("rir_hpf_kwargs")), str(device))
    if key not in _hpf_cache:
        if len(_hpf_cache) > 16:
            _hpf_cache.clear()
        sos = design_highpass_filter_sos(
            fs, constants.get("rir_hpf_fc"), **constants.get("rir_hpf_kwargs")
        )
        _, H = scipy.signal.sosfreqz(sos, worN=np.fft.rfftfreq(n_fft, 1.0 / fs), fs=fs)
        _hpf_cache[key] = torch.as_tensor(np.abs(H) ** 2, device=device)
    H2 = _hpf_cache[key]
    x = np.zeros((len(rirs), L))
    for i, h in enumerate(rirs):
        x[i, : len(h)] = h
    x = torch.as_tensor(x, device=device)
    y = torch.fft.irfft(torch.fft.rfft(x, n=n_fft) * H2, n=n_fft)[:, :L].cpu().numpy()
    return [np.ascontiguousarray(y[i, : len(h)]) for i, h in enumerate(rirs)]


def premix(room, rirs, L):
    """
    ``premix[s, m]``: source ``s`` convolved with ``rirs[m][s]`` and delayed,
    in an ``(n_sources, n_mics, L)`` array, as in ``Room.simulate`` (float64
    FFT convolution on the GPU).
    """
    import scipy.fft

    torch = _torch()
    device = _device()
    M, S = room.mic_array.M, len(room.sources)
    out = torch.zeros(S, M, L, dtype=torch.float64, device=device)
    for s, src in enumerate(room.sources):
        sig = src.signal
        if sig is None:
            continue
        sig = np.asarray(sig, dtype=np.float64)
        d = int(np.floor(src.delay * room.fs))
        hs = [np.asarray(rirs[m][s], dtype=np.float64) for m in range(M)]
        Lh = max(len(h) for h in hs)
        n = len(sig) + Lh - 1
        n_fft = scipy.fft.next_fast_len(n, real=True)
        hb = np.zeros((M, Lh))
        for m, h in enumerate(hs):
            hb[m, : len(h)] = h
        X = torch.fft.rfft(torch.as_tensor(sig, device=device), n=n_fft)
        Hf = torch.fft.rfft(torch.as_tensor(hb, device=device), n=n_fft)
        y = torch.fft.irfft(Hf * X[None, :], n=n_fft)[:, :n]
        lens = torch.as_tensor([len(sig) + len(h) - 1 for h in hs], device=device)
        y = y * (torch.arange(n, device=device)[None, :] < lens[:, None])
        out[s, :, d : d + n] += y
    return out.cpu().numpy()


def use_gpu_post():
    """True if the GPU can be used for the backend-independent steps."""
    return is_available() and not constants.get("exact_mode")
