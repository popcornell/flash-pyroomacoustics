import time, numpy as np, pyroomacoustics as pra
exec(open("profile_ref.py").read().split("for name, mk")[0])
room = hybrid(n_rays=10000); room.image_source_model()
t=time.perf_counter(); room.ray_tracing(); print("rt scat", time.perf_counter()-t)
