"""
GPU ray tracing for 3D shoebox rooms (Triton kernel) and GPU synthesis of the
ray tracing tail of the RIR.

The kernel follows ``Geometry::simul_ray``/``scat_ray`` of the CPU engine (in
its default fast mode, where every ray has its own random generator):
specular reflections, energy absorption, Lambertian scattering with the
diffuse-rain contribution to every receiver, hybrid ISM/RT skipping of the
first ``max_order`` specular reflections, and the energy/time thresholds.
One GPU thread traces one ray; histogram bins are accumulated with atomics.
Many rooms (and sources) are traced in a single launch.

Results are statistically equivalent to the CPU (and original) ray tracer,
not sample-identical: random draws differ and float32 atomics change the
summation order.
"""

import math

import numpy as np

from .parameters import constants

_kernel = None


def _get_kernel():
    global _kernel
    if _kernel is None:
        from ._rt_triton import _rt_kernel

        _kernel = _rt_kernel
    return _kernel


def unsupported_reason(room):
    from . import cuda

    if not cuda.is_available():
        return "PyTorch with CUDA is not available"
    try:
        import triton  # noqa: F401
    except ImportError:
        return "triton is not installed"
    if constants.get("exact_mode"):
        return "exact_mode is on"
    if not hasattr(room, "shoebox_dim") or room.dim != 3:
        return "GPU ray tracing supports 3D shoebox rooms only"
    if any(d is not None for d in room.mic_array.directivity):
        return "directional microphones are not supported"
    return None


def _ray_tables(room, src):
    """Directions (n, 3) and energies (n, n_bands) as passed to the CPU engine."""
    from .doa import fibonacci_spherical_sampling

    n_rays = int(room.rt_args["n_rays"])
    n_bands = len(room.walls[0].absorption)
    if src.directivity is None:
        dirs = fibonacci_spherical_sampling(n_rays).T
        energies = np.ones((n_rays, n_bands))
    else:
        dirs, energies = src.directivity.sample_rays(n_rays)
        energies = np.broadcast_to(energies, (n_rays, n_bands))
    dirs = np.asarray(dirs, dtype=np.float32)
    energies = np.float32(2.0 / dirs.shape[0]) * np.asarray(energies, dtype=np.float32)
    return dirs, energies


def ray_tracing(rooms, block=128):
    """
    Ray tracing of a list of 3D shoebox rooms in one GPU launch.

    For each room, fills ``room.rt_histograms[m][s]`` (a list with one
    ``(n_bands, n_bins)`` array, omnidirectional receiver) and
    ``room.rt_directional_responses``, exactly like ``Room.ray_tracing``.
    """
    import torch

    from . import cuda, random as pra_random

    device = cuda._device()
    kernel = _get_kernel()
    nbs = {len(r.walls[0].absorption) for r in rooms}
    NB = max(nbs)
    NBP = max(1, 1 << (NB - 1).bit_length())

    dirs_all, e_all, job_i, job_f, walls, mics = [], [], [], [], [], []
    block_job, block_ray0 = [], []
    hist_shapes = []
    ray0 = mic0 = hist0 = 0
    from . import libroom

    eps = float(libroom.get_eps())  # geometric tolerance of the engine
    for room in rooms:
        nb = len(room.walls[0].absorption)
        L = np.asarray(room.shoebox_dim, dtype=np.float32)
        c = room.c
        dist_thres = np.float32(room.rt_args["time_thres"]) * np.float32(c)
        hist_res = np.float32(room.rt_args["hist_bin_size"]) * np.float32(c)
        diag = float(np.linalg.norm(L))
        n_bins = int(math.ceil((dist_thres + diag) / hist_res)) + 2
        wt = np.zeros((6, 2 * NBP + 2), dtype=np.float32)
        for wi, w in enumerate(room.walls):  # 3D engine order: 2 * axis + (high side)
            a = np.asarray(w.absorption, dtype=np.float32)
            s = np.asarray(w.scatter, dtype=np.float32)
            wt[wi, :nb] = 1.0 - a
            wt[wi, NBP : NBP + nb] = s
            wt[wi, 2 * NBP] = s.mean()
            wt[wi, 2 * NBP + 1] = float(s.max() > 0)
        M = room.mic_array.M
        mics.append(np.asarray(room.mic_array.R.T, dtype=np.float32))
        hybrid = int(room.simulator_state["ism_needed"] and room.simulator_state["rt_needed"])
        for src in room.sources:
            d, e = _ray_tables(room, src)
            ep = np.zeros((d.shape[0], NBP), dtype=np.float32)
            ep[:, :nb] = e
            n = d.shape[0]
            e0max = float(e.max()) if e.size else 0.0
            seed = int(pra_random.get_rng().integers(1, 2**30))
            job_i.append([ray0 + n, mic0, M, hist0, n_bins, int(room.max_order), hybrid, seed])
            job_f.append(
                list(L)
                + list(np.asarray(src.position, dtype=np.float32))
                + [
                    e0max * room.rt_args["energy_thres"],
                    dist_thres,
                    room.rt_args["receiver_radius"],
                    hist_res,
                    room.rt_args["energy_thres"],
                ]
                + [0.0] * 5
            )
            walls.append(wt)
            for r in range(ray0, ray0 + n, block):
                block_job.append(len(job_i) - 1)
                block_ray0.append(r)
            dirs_all.append(d)
            e_all.append(ep)
            hist_shapes.append((hist0, M, nb, n_bins))
            hist0 += M * NB * n_bins
            ray0 += n
        mic0 += M

    def T(x, dt):
        return torch.as_tensor(np.ascontiguousarray(np.asarray(x, dtype=dt)), device=device)

    hist = torch.zeros(max(1, hist0), dtype=torch.float32, device=device)
    grid = (len(block_job),)
    if grid[0] > 0:
        kernel[grid](
            T(np.concatenate(dirs_all), np.float32),
            T(np.concatenate(e_all), np.float32),
            T(block_job, np.int32),
            T(block_ray0, np.int32),
            T(job_i, np.int32),
            T(job_f, np.float32),
            T(np.stack(walls), np.float32),
            T(np.concatenate(mics), np.float32),
            hist,
            eps,
            NB=NB,
            NBP=NBP,
            BLOCK=block,
            MAX_BOUNCES=1 << 20,
        )
    hist = hist.cpu().numpy()

    j = 0
    for room in rooms:
        M = room.mic_array.M
        room.rt_directional_responses = [
            {"response": np.ones(1), "is_impulse_response": False} for _ in range(M)
        ]
        room.rt_histograms = [[] for _ in range(M)]
        for s in range(len(room.sources)):
            h0, _, nb, n_bins = hist_shapes[j]
            hs = hist[h0 : h0 + M * NB * n_bins].reshape(M, NB, n_bins)
            for m in range(M):
                room.rt_histograms[m].append([hs[m, :nb].copy()])
            j += 1
        room.simulator_state["rt_done"] = True


# ---------------------------------------------------------------------------
# Tail synthesis (``simulation.rt.compute_rt_rir`` for omnidirectional receivers)
# ---------------------------------------------------------------------------


def _poisson_times(volume, durations, c, t0s, seed, max_rate=10000):
    """Delta times of ``poisson_sequence`` (same recurrence) for many pairs."""
    from . import libroom

    g = np.random.default_rng(seed)
    fpcv = 4 * np.pi * c**3 / volume
    out = []
    for dur, t0 in zip(durations, t0s):
        n = int(max_rate * dur * 1.25) + 64
        while True:
            logs = np.log(1.0 / g.uniform(size=n))
            times, _, done = libroom._poisson_times(
                float(t0), float(dur), float(fpcv), float(max_rate), logs
            )
            if done:
                break
            n *= 4
        out.append(times)
    return out


def rt_tails(room):
    """Ray tracing part of the RIRs of one room, ``out[m][s]``; see :py:func:`rt_tails_batch`."""
    return rt_tails_batch([room])[0]


def rt_tails_batch(rooms):
    """
    Ray tracing part of the RIRs, ``out[i][m][s]`` (float64 numpy arrays) for
    every room ``rooms[i]``, from ``room.rt_histograms``.

    Same steps as ``compute_rt_rir`` for omnidirectional receivers, batched
    over all the source/microphone pairs of all the rooms. With the
    ``rt_tail`` constant set to ``"gaussian"``, the random sequence of signed
    impulses (whose density grows like t^2) is replaced by white Gaussian
    noise of the same power, starting at the direct path time.
    """
    import torch

    from . import cuda, random as pra_random

    device = cuda._device()
    f64 = torch.float64
    gaussian = constants.get("rt_tail") == "gaussian"
    seed = int(pra_random.get_rng().integers(2**62))
    gen = torch.Generator(device=device).manual_seed(seed)

    outs = []
    jobs = []
    for ri, room in enumerate(rooms):
        fs, c = room.fs, room.c
        fdl2 = constants.get("frac_delay_length") // 2
        hbss = int(room.rt_args["hist_bin_size_samples"])
        hist_bin_size = room.rt_args["hist_bin_size"]
        M, S = room.mic_array.M, len(room.sources)
        out = [[None] * S for _ in range(M)]
        outs.append(out)
        air = None if room.air_absorption is None else tuple(np.asarray(room.air_absorption, float))
        for m in range(M):
            for s in range(S):
                h = room.rt_histograms[m][s][0]
                nz = np.nonzero(h.sum(axis=0))[0]
                if len(nz) == 0:
                    out[m][s] = np.zeros(fdl2)
                    continue
                t_max = (int(nz[-1]) + 1) * hist_bin_size
                n_bins = math.ceil(t_max * fs / hbss)
                N = n_bins * hbss
                hb = np.zeros((h.shape[0], n_bins))
                k = min(n_bins, h.shape[1])
                hb[:, :k] = h[:, :k]
                t0 = float(np.linalg.norm(room.sources[s].position - room.mic_array.R[:, m]) / c)
                jobs.append(
                    dict(room=ri, m=m, s=s, N=N, hist=hb, t0=t0, fs=fs, c=c, hbss=hbss,
                         fdl2=fdl2, air=air, ob=room.octave_bands, vol=room.get_volume())
                )
    if not jobs:
        return outs

    if not gaussian:
        g = np.random.default_rng(seed)
        for j in jobs:
            t = _poisson_times(j["vol"], [j["N"] / j["fs"]], j["c"], [j["t0"]], int(g.integers(2**62)))[0]
            if not constants.get("rt_tail_t0_impulse") and len(t) > 1:
                t = t[1:]  # drop the impulse placed exactly at the direct path time
            j["idx"] = (t * j["fs"]).astype(np.int64)

    groups = {}
    for j in jobs:
        n_fft = 2 ** math.ceil(math.log2(2 * j["N"]))
        key = (n_fft, j["fs"], j["c"], j["hbss"], j["hist"].shape[0], j["air"], j["fdl2"])
        groups.setdefault(key, []).append(j)

    for (n_fft, fs, c, hbss, nb, air, fdl2), grp in groups.items():
        P = len(grp)
        Ns = torch.as_tensor([j["N"] for j in grp], device=device)
        Nmax = int(max(j["N"] for j in grp))
        n = torch.arange(Nmax, device=device, dtype=f64)
        inside = n[None, :] < Ns[:, None]

        if gaussian:
            # Same expected local energy as the original sequence. The original
            # draws impulses at rate lam(t) = min(4 pi c^3 / V * (t0 + t)^2, 10^4)
            # (sic, rate evaluated at t0 + t), gives each impulse the amplitude
            # 1 / sqrt(0.5 * #occupied samples in a 2 * hbss window), and adds
            # impulses falling on the same sample. With lam_s = lam / fs per
            # sample, the expected energy per sample is
            #   e = 2 lam_s (1 - exp(-2 hbss lam_s)) / (2 hbss (1 - exp(-lam_s)))
            # (-> 2 lam_s when sparse). It also always has one impulse at t0.
            t0s = torch.as_tensor([j["t0"] for j in grp], device=device, dtype=f64)
            fpcv = torch.as_tensor(
                [4 * np.pi * c**3 / j["vol"] for j in grp], device=device, dtype=f64
            )
            start = torch.floor(t0s * fs).long()

            def energy(lam_s):
                lam_s = lam_s.clamp(min=1e-300)
                return (
                    2 * lam_s * (-torch.expm1(-2 * hbss * lam_s))
                    / (2 * hbss * (-torch.expm1(-lam_s)))
                )

            tt = n[None, :] / fs
            lam_s = torch.clamp(fpcv[:, None] * (t0s[:, None] + tt) ** 2, max=10000.0) / fs
            seq = torch.randn(P, Nmax, generator=gen, device=device, dtype=f64)
            seq = seq * torch.sqrt(energy(lam_s))
            seq = seq * ((n[None, :] >= start[:, None]) & inside)
            # the impulse at t0: amplitude sqrt(E[2 / #occupied]) ~ sqrt(2)
            lam0 = torch.clamp(fpcv * (2 * t0s) ** 2, max=10000.0) / fs
            q = -torch.expm1(-lam0.clamp(min=1e-300))
            first = 2 * (-torch.expm1(-2 * hbss * lam0.clamp(min=1e-300))) / (2 * hbss * q)
            sign = torch.randint(0, 2, (P,), generator=gen, device=device).to(f64) * 2 - 1
            if constants.get("rt_tail_t0_impulse"):
                seq.scatter_add_(1, start[:, None].clamp(max=Nmax - 1), (sign * torch.sqrt(first))[:, None])
        else:
            Lmax = int(max(j["idx"][-1] for j in grp)) + 1
            K = max(len(j["idx"]) for j in grp)
            idx = np.full((P, K), Lmax, dtype=np.int64)  # padding -> dummy column Lmax
            for p, j in enumerate(grp):
                idx[p, : len(j["idx"])] = j["idx"]
            idx = torch.as_tensor(idx, device=device)
            valid = idx < Lmax
            # local power of the binary sequence: 0.5 * sum occ[i-h .. i+h-1]
            occ = torch.zeros(P, Lmax + 1, dtype=f64, device=device)
            occ.scatter_(1, idx, valid.to(f64))
            cs = torch.nn.functional.pad(torch.cumsum(occ[:, :Lmax], 1), (1, 0))
            lo = (idx - hbss).clamp(0, Lmax)
            hi = (idx + hbss).clamp(0, Lmax)
            power = 0.5 * (cs.gather(1, hi) - cs.gather(1, lo))
            good = valid & (power > 1e-10)
            amp = torch.where(good, 1.0 / torch.where(good, power, 1.0), 0.0).sqrt()
            signs = torch.randint(0, 2, (P, K), generator=gen, device=device).to(f64) * 2 - 1
            seq = torch.zeros(P, max(Lmax + 2, Nmax), dtype=f64, device=device)
            seq.scatter_add_(1, idx, signs * amp)
            seq = seq[:, :Nmax] * inside

        H = None
        if nb > 1 or air is not None:
            H = torch.as_tensor(grp[0]["ob"]._squared_windows(n_fft), device=device)

        def bands_of(x):  # (P, Nmax) -> (P, n_bands, Nmax), truncated to each N
            X = torch.fft.rfft(x, n=n_fft)
            y = torch.fft.irfft(X[:, None, :] * H[None], n=n_fft)[..., :Nmax]
            return y * inside[:, None, :]

        rb = bands_of(seq) if nb > 1 else seq[:, None, :]

        # interp_hist: linear between bin centers, first value held, last pad samples zero
        n_in = torch.as_tensor([j["hist"].shape[1] for j in grp], device=device)
        n_in_max = int(n_in.max())
        hist = np.zeros((P, nb, n_in_max))
        for p, j in enumerate(grp):
            hist[p, :, : j["hist"].shape[1]] = j["hist"]
        hist = torch.as_tensor(hist, device=device)
        pad = (Ns // n_in) // 2
        span = (Ns - 1 - 2 * pad).to(f64)
        scale = torch.where(n_in > 1, (n_in - 1).to(f64) / span, torch.zeros_like(span))
        u = ((n[None, :] - pad[:, None]) * scale[:, None]).clamp(min=0)
        u = torch.minimum(u, (n_in - 1).to(f64)[:, None])
        j0 = torch.minimum(torch.floor(u).long(), (n_in - 2).clamp(min=0)[:, None])
        fr = u - j0
        j1 = torch.minimum(j0 + 1, (n_in - 1)[:, None])
        g0 = hist.gather(2, j0[:, None, :].expand(P, nb, Nmax))
        g1 = hist.gather(2, j1[:, None, :].expand(P, nb, Nmax))
        env = g0 * (1 - fr[:, None, :]) + g1 * fr[:, None, :]
        env = env * (n[None, :] < (Ns - pad)[:, None])[:, None, :]
        rb = rb * torch.sqrt(env.clamp(min=0.0))

        if air is not None:
            if rb.shape[1] == 1:
                rb = bands_of(rb[:, 0])
            a = torch.as_tensor(np.asarray(air, np.float64), device=device)
            rb = rb * torch.exp(-0.5 * a[:, None] * n[None, :] / fs * c)[None]

        rir = rb.sum(1).cpu().numpy()
        for p, j in enumerate(grp):
            outs[j["room"]][j["m"]][j["s"]] = np.pad(rir[p, : j["N"]], (fdl2, 0))
    return outs
