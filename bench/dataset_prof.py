import cProfile, pstats, sys, time, numpy as np, pyroomacoustics as pra
if hasattr(pra.libroom, "set_parallel_rng"): pra.libroom.set_parallel_rng(True)
def gen(n=10, hybrid=False, seed=1):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        dim = rng.uniform([3, 3, 2.5], [10, 8, 4])
        rt60 = rng.uniform(0.2, 1.0)
        e, o = pra.inverse_sabine(rt60, dim)
        if hybrid:
            room = pra.ShoeBox(dim, fs=16000, materials=pra.Material(e, 0.1), max_order=3, ray_tracing=True, air_absorption=True)
        else:
            room = pra.ShoeBox(dim, fs=16000, materials=pra.Material(e), max_order=o)
        room.add_source(dim * rng.uniform(0.2, 0.8, 3))
        room.add_microphone_array(dim[:, None] * rng.uniform(0.2, 0.8, (3, 4)))
        room.compute_rir()
for hyb in [False, True]:
    t = time.perf_counter(); gen(hybrid=hyb); print("hybrid" if hyb else "ism", time.perf_counter() - t)
    if len(sys.argv) > 1:
        pr = cProfile.Profile(); pr.enable(); gen(hybrid=hyb); pr.disable()
        pstats.Stats(pr).sort_stats("tottime").print_stats(12)
