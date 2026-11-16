import time, cProfile, pstats, io, sys
import numpy as np
import pyroomacoustics as pra

def t(label, f):
    t0 = time.perf_counter(); r = f(); print(f"{label:55s} {time.perf_counter()-t0:8.3f}s"); return r

def shoebox_ism(order=17, multiband=False, mics=4):
    m = pra.make_materials(*[("hard_surface" if multiband else 0.2)]*1) if False else None
    mat = pra.Material("brickwork") if multiband else pra.Material(0.2)
    room = pra.ShoeBox([6, 5, 3], fs=16000, materials=mat, max_order=order)
    room.add_source([2, 3, 1.5], signal=np.random.randn(16000))
    room.add_microphone_array(np.c_[[[3.5 + 0.05*i, 2.0, 1.4] for i in range(mics)]].T if False else np.array([[3.5+0.05*i for i in range(mics)],[2.0]*mics,[1.4]*mics]))
    return room

def hybrid(scat=0.1, n_rays=10000, mics=2):
    mat = pra.Material(energy_absorption="hard_surface", scattering=scat) if scat else pra.Material("hard_surface")
    room = pra.ShoeBox([8, 6, 3.5], fs=16000, materials=mat, max_order=3, ray_tracing=True, air_absorption=True)
    room.set_ray_tracing(n_rays=n_rays)
    room.add_source([2, 3, 1.5], signal=np.random.randn(16000))
    room.add_microphone_array(np.array([[5.0+0.1*i for i in range(mics)],[2.0]*mics,[1.4]*mics]))
    return room

def poly(order=4):
    corners = np.array([[0,0],[0,4],[2,6],[5,6],[6,3],[4,0]]).T
    room = pra.Room.from_corners(corners, fs=16000, max_order=order, materials=pra.Material(0.2))
    room.extrude(3.0)
    room.add_source([1.5, 2, 1.5], signal=np.random.randn(16000))
    room.add_microphone([3.5, 3.5, 1.2])
    return room

for name, mk in [("shoebox ISM order17 1band 4mics", lambda: shoebox_ism()),
                 ("shoebox ISM order17 octave 4mics", lambda: shoebox_ism(multiband=True)),
                 ("hybrid ISM3+RT 10k rays scat=0.1 2mics", lambda: hybrid()),
                 ("hybrid ISM3+RT 10k rays no scat 2mics", lambda: hybrid(scat=0)),
                 ("polygon extruded ISM order4", lambda: poly())]:
    room = mk()
    t(name + " | ism", room.image_source_model)
    t(name + " | rt", room.ray_tracing)
    t(name + " | compute_rir", room.compute_rir)
    t(name + " | simulate", room.simulate)
