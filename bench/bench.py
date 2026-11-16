"""Benchmark: full room.simulate() pipelines, default pyroomacoustics settings."""
import json, sys, time, numpy as np, pyroomacoustics as pra

def shoebox(order=17, multiband=False, mics=4, rt=False, scat=None, n_rays=10000):
    if scat is not None:
        mat = pra.Material(energy_absorption="hard_surface", scattering=scat)
    else:
        mat = pra.Material("brickwork") if multiband else pra.Material(0.2)
    room = pra.ShoeBox([8, 6, 3.5], fs=16000, materials=mat, max_order=order, ray_tracing=rt, air_absorption=rt)
    if rt: room.set_ray_tracing(n_rays=n_rays)
    room.add_source([2, 3, 1.5], signal=np.random.default_rng(0).standard_normal(16000))
    room.add_microphone_array(np.array([[5.0 + 0.1*i for i in range(mics)], [2.0]*mics, [1.4]*mics]))
    return room

def lshape(rt):
    corners = np.array([[0,0],[0,6],[3,6],[3,3],[7,3],[7,0]]).T
    m = pra.Material(0.15, 0.2)
    room = pra.Room.from_corners(corners, fs=16000, max_order=5, materials=m, ray_tracing=rt)
    room.extrude(3.0, materials=m)
    room.add_source([1.5, 4.5, 1.5], signal=np.random.default_rng(0).standard_normal(16000))
    room.add_microphone([5.5, 1.5, 1.2]); room.add_microphone([1.0, 1.0, 1.7])
    return room

def dataset(n=20):
    rng = np.random.default_rng(1)
    for _ in range(n):
        dim = rng.uniform([3, 3, 2.5], [10, 8, 4])
        e, o = pra.inverse_sabine(rng.uniform(0.2, 0.8), dim)
        room = pra.ShoeBox(dim, fs=16000, materials=pra.Material(e), max_order=o)
        room.add_source(dim * rng.uniform(0.2, 0.8, 3)); room.add_microphone(dim * rng.uniform(0.2, 0.8, 3))
        room.compute_rir()

cases = {
    "ShoeBox ISM, order 17, 1 band, 4 mics": lambda: shoebox().simulate(),
    "ShoeBox ISM, order 17, octave bands, 4 mics": lambda: shoebox(multiband=True).simulate(),
    "Hybrid ISM(3)+RT 10k rays, no scattering": lambda: shoebox(order=3, rt=True, mics=2).simulate(),
    "Hybrid ISM(3)+RT 10k rays, scattering 0.1": lambda: shoebox(order=3, rt=True, mics=2, scat=0.1).simulate(),
    "L-shaped room (non-convex), ISM order 5": lambda: lshape(False).simulate(),
    "L-shaped room, hybrid ISM+RT with scattering": lambda: lshape(True).simulate(),
    "20 random rooms via inverse_sabine (dataset gen)": lambda: dataset(),
}
if False:
    pra.libroom.set_parallel_rng(True)
res = {}
for name, f in cases.items():
    pra.random.seed(numpy=0, libroom=0)
    f()  # warm-up
    ts = []
    for _ in range(3):
        pra.random.seed(numpy=0, libroom=0)
        t = time.perf_counter(); f(); ts.append(time.perf_counter() - t)
    res[name] = min(ts); print(f"{name:52s} {min(ts):8.3f}s", flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1)
