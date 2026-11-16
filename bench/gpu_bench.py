"""CPU (Rust, PRA_NUM_THREADS) vs CUDA backend, per room and batched."""
import time

import numpy as np
import torch

import pyroomacoustics as pra

SIG = np.random.default_rng(0).standard_normal(160000)  # 10 s at 16 kHz


def rooms(n=10, kind="hybrid", seed=1, n_rays=None):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        dim = rng.uniform([3, 3, 2.5], [10, 8, 4])
        rt60 = rng.uniform(0.2, 1.0)
        e, o = pra.inverse_sabine(rt60, dim)
        if kind == "hybrid":
            room = pra.ShoeBox(dim, fs=16000, materials=pra.Material(e, 0.1), max_order=3,
                               ray_tracing=True, air_absorption=True)
            if n_rays:
                room.set_ray_tracing(n_rays=n_rays)
        else:
            room = pra.ShoeBox(dim, fs=16000, materials=pra.Material(e), max_order=o)
        room.add_source(dim * rng.uniform(0.2, 0.8, 3), signal=SIG)
        room.add_microphone_array(dim[:, None] * rng.uniform(0.2, 0.8, (3, 4)))
        out.append(room)
    return out


def run(make, mode, simulate=False, tail="poisson"):
    pra.constants.set("rt_tail", tail)
    best = np.inf
    for _ in range(3):
        rs = make()
        torch.cuda.synchronize()
        t = time.perf_counter()
        if mode == "cpu":
            pra.constants.set("backend", "cpu")
            for r in rs:
                r.compute_rir()
        else:
            pra.constants.set("backend", "cuda")
            pra.cuda.compute_rirs(rs, anechoic=simulate)
        if simulate:
            for r in rs:
                if mode == "cpu":
                    r.compute_anechoic_rir()
                r.simulate(return_premix=True, return_anechoic=True)
        torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t)
    pra.constants.set("backend", "cpu")
    pra.constants.set("rt_tail", "poisson")
    return best


# warm up (Triton compilation, CUDA context)
pra.cuda.compute_rirs(rooms(2))
cases = [
    ("10 rooms, hybrid, default n_rays, RIRs only", lambda: rooms(10), False),
    ("10 rooms, hybrid, 100k rays, RIRs only", lambda: rooms(10, n_rays=100_000), False),
    ("50 rooms, hybrid, default n_rays, RIRs only", lambda: rooms(50, seed=2), False),
    ("10 rooms, full-order ISM, RIRs only", lambda: rooms(10, kind="ism"), False),
    ("10 rooms, hybrid, + anechoic + simulate(10 s signal)", lambda: rooms(10), True),
    ("10 rooms, full-order ISM, + anechoic + simulate(10 s)", lambda: rooms(10, kind="ism"), True),
]
print(f"{'workload (4 mics per room)':56s} {'cpu':>8s} {'cuda':>8s} {'cuda+gauss':>10s}")
for label, make, sim in cases:
    t = [run(make, "cpu", sim), run(make, "cuda", sim), run(make, "cuda", sim, "gaussian")]
    print(f"{label:56s} {t[0]:7.3f}s {t[1]:7.3f}s {t[2]:9.3f}s")
