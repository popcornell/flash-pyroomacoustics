"""Generate RIRs for random rooms and dump them + acoustic parameters."""
import sys, numpy as np, pyroomacoustics as pra
out_path, seed = sys.argv[1], int(sys.argv[2])
if hasattr(pra.libroom, "set_parallel_rng"): pra.libroom.set_parallel_rng(True)
fs = 16000
def params(h):
    h = np.asarray(h, float); e = h**2
    edc = np.cumsum(e[::-1])[::-1]; edc_db = 10*np.log10(edc/edc[0] + 1e-30)
    t = np.arange(len(h))/fs
    def slope_time(lo, hi):
        i0 = np.argmax(edc_db <= lo); i1 = np.argmax(edc_db <= hi)
        return (t[i1]-t[i0]) * 60/(lo-hi) if i1 > i0 else np.nan
    pk = np.argmax(np.abs(h)); w = int(0.0025*fs)
    direct = e[max(0,pk-w):pk+w].sum(); late = e.sum() - direct
    n50 = pk + int(0.05*fs)
    return [slope_time(-5,-25), slope_time(0,-10), 10*np.log10(e[:n50].sum()/max(e[n50:].sum(),1e-30)),
            10*np.log10(direct/max(late,1e-30)), 10*np.log10(e.sum())]
import os
kinds = os.environ.get("KINDS", "ism,ism_octave_air,hybrid_scat,hybrid_octave_scat,lshape_hybrid").split(",")
res = {}
for k_i, kind in enumerate(kinds):
    geo = np.random.default_rng(1000 + k_i)  # room geometry identical across runs
    P, R = [], []
    for r in range(12):
        pra.random.seed(numpy=seed*100+r, libroom=seed*100+r)
        dim = geo.uniform([3, 3, 2.5], [9, 7, 3.5]); rt60 = geo.uniform(0.3, 0.9)
        src = dim*geo.uniform(0.2, 0.8, 3); mics = dim[:, None]*geo.uniform(0.2, 0.8, (3, 2))
        e, o = pra.inverse_sabine(rt60, dim)
        if kind == "ism":
            room = pra.ShoeBox(dim, fs=fs, materials=pra.Material(e), max_order=o)
        elif kind == "ism_octave_air":
            room = pra.ShoeBox(dim, fs=fs, materials=pra.Material("brickwork"), max_order=min(o, 30), air_absorption=True)
        elif kind == "hybrid_scat":
            room = pra.ShoeBox(dim, fs=fs, materials=pra.Material(e, 0.2), max_order=3, ray_tracing=True)
        elif kind == "hybrid_octave_scat":
            mats = pra.make_materials(ceiling="hard_surface", floor="carpet_cotton", east="brickwork", west="brickwork", north="plasterboard", south="plasterboard")
            mats = {k: pra.Material(v.energy_absorption, 0.15) for k, v in mats.items()}
            room = pra.ShoeBox(dim, fs=fs, materials=mats, max_order=3, ray_tracing=True, air_absorption=True)
        else:
            c = np.array([[0,0],[0,6],[3,6],[3,3],[7,3],[7,0]]).T
            room = pra.Room.from_corners(c, fs=fs, max_order=3, materials=pra.Material(0.15, 0.2), ray_tracing=True)
            room.extrude(3.0, materials=pra.Material(0.15, 0.2)); src = np.array([1.5, 4.5, 1.5]); mics = np.array([[5.5, 1.5, 1.2], [1.0, 1.0, 1.7]]).T
        room.add_source(src); room.add_microphone_array(mics); room.compute_rir()
        for m in range(2):
            h = np.asarray(room.rir[m][0]); P.append(params(h)); R.append(h)
    res[kind + "_params"] = np.array(P)
    L = max(len(h) for h in R); res[kind + "_rirs"] = np.stack([np.pad(h, (0, L-len(h))) for h in R])
np.savez(out_path, **res)
