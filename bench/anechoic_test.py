"""Checks the aligned anechoic RIRs (CPU and CUDA) and the CUDA ISM backend."""
import os
import time

import numpy as np

os.environ.setdefault("PRA_NUM_THREADS", "8")
import pyroomacoustics as pra

fs = 16000
rng = np.random.default_rng(0)
ok = True


def check(name, cond, info=""):
    global ok
    ok &= bool(cond)
    print(f"[{'OK' if cond else 'FAIL'}] {name} {info}")


def make_room(multi=False, air=False, rand=False, rt=False, order=10, seed=0):
    r = np.random.default_rng(seed)
    dim = r.uniform([3, 3, 2.4], [9, 8, 3.5])
    if multi:
        mat = pra.Material("hard_surface") if not rt else pra.Material("brickwork", 0.2)
    else:
        mat = pra.Material(0.3, 0.1 if rt else None)
    room = pra.ShoeBox(dim, fs=fs, materials=mat, max_order=order, air_absorption=air,
                       use_rand_ism=rand, ray_tracing=rt)
    room.add_source(r.uniform(0.5, dim - 0.5), signal=r.standard_normal(fs), delay=0.013)
    room.add_source(r.uniform(0.5, dim - 0.5), signal=r.standard_normal(fs // 2))
    room.add_microphone_array(r.uniform(0.5, dim - 0.5, size=(4, 3)).T)
    return room


def first_reflection(room, s, m):
    """arrival time of the earliest order-1 image (CPU ISM)."""
    src = room.sources[s]
    img = src.images[:, src.orders == 1]
    return np.linalg.norm(img - room.mic_array.R[:, m:m + 1], axis=0).min() / room.c


def rel(a, b):
    n = min(len(a), len(b))
    return np.linalg.norm(a[:n] - b[:n]) / np.linalg.norm(b[:n])


fdl2 = pra.constants.get("frac_delay_length") // 2

# 1) the anechoic RIR is the direct-path component of the RIR (CPU).
# The default 10 Hz zero-phase high-pass filter spreads reflections backwards in
# time, so the exact comparison is done with it disabled.
pra.constants.set("rir_hpf_enable", False)
for multi, air, rand in [(False, False, False), (True, False, False), (True, True, False), (False, False, True)]:
    room = make_room(multi=multi, air=air, rand=rand)
    room.compute_rir(backend="cpu")
    room.compute_anechoic_rir(backend="cpu")
    worst, peak_ok = 0.0, True
    for m in range(4):
        for s in range(2):
            h, a = room.rir[m][s], room.anechoic_rir[m][s]
            pos = room.sources[s].images[:, room.sources[s].orders == 0][:, 0]
            d = np.linalg.norm(pos - room.mic_array.R[:, m])
            n0 = (d / room.c) * fs + fdl2  # direct path sample index in both arrays
            peak_ok &= abs(np.argmax(np.abs(a)) - n0) <= 1
            n1 = min(int(first_reflection(room, s, m) * fs), len(a))  # first reflection filter start
            worst = max(worst, np.abs(h[:n1] - a[:n1]).max() / np.abs(a).max())
    tol = 1e-6 if not multi else 2e-3  # octave filters of the reflections have long tails
    check(f"rir == anechoic before 1st reflection (multi={multi} air={air} rand_ism={rand})",
          worst < tol and peak_ok, f"(max dev {worst:.1e}, peak at time of flight: {peak_ok})")
pra.constants.set("rir_hpf_enable", True)

# 2) simulate(return_anechoic=True): aligned, same shape, same scaling
room = make_room(multi=True, air=True)
premix, an = room.simulate(return_premix=True, return_anechoic=True, snr=20)
check("simulate: premix and anechoic same shape", premix.shape == an.shape, str(an.shape))
lags = []
W = 5  # reflections arrive later than that
for s in range(2):
    for m in range(4):
        x, y = premix[s, m], an[s, m]
        c = [np.dot(x[W + k: len(x) - W + k], y[W: len(y) - W]) for k in range(-W, W + 1)]
        lags.append(int(np.argmax(c)) - W)
check("simulate: premix/anechoic cross-correlation peak at lag 0", all(l == 0 for l in lags), str(lags))
# the anechoic signal is the direct-path part of premix: subtracting it removes the
# direct sound and leaves the reverberant part
room.simulator_state["rir_done"] = False

# 4) CUDA backend vs CPU
if pra.cuda.is_available():
    # force the GPU ISM path even for low orders; compare with the high-pass off
    # (it is checked separately below)
    pra.constants.set("cuda_min_ism_work", 0)
    pra.constants.set("rir_hpf_enable", False)
    for kw in [dict(), dict(multi=True), dict(multi=True, air=True), dict(order=40)]:
        errs, aerr = [], []
        for seed in range(3):
            room = make_room(seed=seed, **kw)
            room.compute_rir(backend="cpu")
            room.compute_anechoic_rir(backend="cpu")
            cpu = [[h.copy() for h in row] for row in room.rir]
            cpu_a = [[h.copy() for h in row] for row in room.anechoic_rir]
            room.compute_rir(backend="cuda")
            room.compute_anechoic_rir(backend="cuda")
            for m in range(4):
                for s in range(2):
                    assert len(cpu[m][s]) == len(room.rir[m][s])
                    errs.append(rel(room.rir[m][s], cpu[m][s]))
                    aerr.append(rel(room.anechoic_rir[m][s], cpu_a[m][s]))
        check(f"cuda vs cpu ISM {kw}", max(errs) < 1e-4 and max(aerr) < 1e-4,
              f"(rir rel err {max(errs):.1e}, anechoic rel err {max(aerr):.1e})")
    pra.constants.set("cuda_min_ism_work", 40000)
    pra.constants.set("rir_hpf_enable", True)

    # GPU zero-phase high-pass vs sosfiltfilt (differences only from the edge
    # handling of sosfiltfilt, which matters for short RIRs)
    from scipy.signal import sosfiltfilt
    from pyroomacoustics.utilities import design_highpass_filter_sos
    sos = design_highpass_filter_sos(fs, pra.constants.get("rir_hpf_fc"), **pra.constants.get("rir_hpf_kwargs"))
    for order in [10, 40]:
        room = make_room(order=order)
        pra.constants.set("rir_hpf_enable", False)
        room.compute_rir(backend="cpu")
        pra.constants.set("rir_hpf_enable", True)
        raw = [h for row in room.rir for h in row]
        gpu = pra.cuda.highpass(raw, fs)
        err = max(rel(g, sosfiltfilt(sos, h)) for g, h in zip(gpu, raw))
        check(f"GPU high-pass vs sosfiltfilt (order {order}, RIR ~{len(raw[0]) / fs:.2f} s)", err < 3e-2, f"(rel err {err:.1e})")

    # hybrid: the ray tracing tail is random, compare energies (mean over rooms)
    ec, eg, early = [], [], []
    for seed in range(6):
        room = make_room(seed=seed, multi=True, air=True, rt=True, order=3)
        room.compute_rir(backend="cpu")
        cpu = [[h.copy() for h in row] for row in room.rir]
        room.simulator_state["rt_done"] = False
        room.compute_rir(backend="cuda")
        for m in range(4):
            for s in range(2):
                ec.append((cpu[m][s] ** 2).sum())
                eg.append((room.rir[m][s] ** 2).sum())
    ratio = np.mean(eg) / np.mean(ec)
    # (sample-wise comparison is meaningless: the random tail even starts with an
    # impulse of random sign at the direct path time, in the original too)
    check("cuda vs cpu hybrid: RIR energy", abs(ratio - 1) < 0.05, f"(gpu/cpu energy {ratio:.3f})")

    # timing on dataset-like rooms (inverse sabine, high order)
    rooms = []
    r = np.random.default_rng(1)
    for i in range(10):
        dim = r.uniform([3, 3, 2.5], [10, 8, 4])
        rt60 = r.uniform(0.2, 1.0)
        e, order = pra.inverse_sabine(rt60, dim)
        room = pra.ShoeBox(dim, fs=fs, materials=pra.Material(e), max_order=order)
        room.add_source(r.uniform(0.5, dim - 0.5))
        room.add_microphone_array(r.uniform(0.5, dim - 0.5, size=(4, 3)).T)
        rooms.append(room)
    for backend in ["cpu", "cuda", "cpu", "cuda"]:
        t = time.perf_counter()
        for room in rooms:
            room.compute_rir(backend=backend)
            room.compute_anechoic_rir(backend=backend)
            room.simulator_state["ism_done"] = False
        print(f"inverse_sabine 10 rooms x 4 mics, {backend}: {time.perf_counter() - t:.2f} s")
else:
    print("CUDA not available: skipped GPU checks")

print("ALL OK" if ok else "SOME CHECKS FAILED")
