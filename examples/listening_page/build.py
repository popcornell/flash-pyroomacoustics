"""Build the RIR listening page: a self-contained HTML file to inspect RIRs and
listen to reverberant speech next to its aligned direct-path and early targets.

Usage:
    python examples/listening_page/build.py SPEECH.wav OUT.html

SPEECH.wav must be 16 kHz mono, e.g. one of the CMU ARCTIC samples in the upstream
repository (examples/input_samples/cmu_arctic_us_aew_a0001.wav). The page embeds
all the audio (about 9 MB for a 4 s utterance).
"""
import base64
import io
import json
import os
import sys
import wave

os.environ.setdefault("PRA_NUM_THREADS", "8")
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft

import pyroomacoustics as pra

FS = 16000
SPEECH = sys.argv[1]
OUT = sys.argv[2]
pra.random.seed(numpy=7, libroom=7)
pra.constants.set("rt_tail_t0_impulse", False)

fs, x = wavfile.read(SPEECH)
assert fs == FS
x = x.astype(np.float64) / 32768.0

ROOMS = [
    dict(id="booth", name="Vocal booth", dim=[2.4, 2.0, 2.3], rt60=0.15, mode="hybrid",
         src=[0.8, 1.0, 1.4], near=[1.4, 1.1, 1.3], far=[2.0, 1.6, 1.2]),
    dict(id="office", name="Small office", dim=[4.0, 3.5, 2.7], rt60=0.35, mode="hybrid",
         src=[1.0, 1.7, 1.5], near=[2.0, 1.8, 1.3], far=[3.4, 2.9, 1.2]),
    dict(id="living", name="Living room", dim=[6.0, 4.5, 2.8], rt60=0.55, mode="hybrid",
         src=[1.5, 2.0, 1.6], near=[2.5, 2.3, 1.2], far=[5.2, 3.8, 1.0]),
    dict(id="class", name="Classroom", dim=[9.0, 7.0, 3.2], rt60=0.8, mode="hybrid",
         src=[1.5, 3.5, 1.7], near=[2.7, 3.2, 1.2], far=[7.8, 5.5, 1.2]),
    dict(id="class_ism", name="Classroom, full-order ISM", dim=[9.0, 7.0, 3.2], rt60=0.8,
         mode="ism", src=[1.5, 3.5, 1.7], near=[2.7, 3.2, 1.2], far=[7.8, 5.5, 1.2]),
    dict(id="hall", name="Concrete hall", dim=[14.0, 10.0, 5.0], rt60=1.6, mode="hybrid",
         src=[3.0, 5.0, 1.7], near=[4.2, 5.4, 1.4], far=[11.5, 8.0, 1.4]),
]


def b64_wav(sig):
    pcm = np.clip(sig * 32767, -32768, 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(FS)
        w.writeframes(pcm.tobytes())
    return base64.b64encode(buf.getvalue()).decode()


def b64_i16(a):
    """int16 base64 with a scale factor, for plotting."""
    s = float(np.abs(a).max()) or 1.0
    q = np.round(a / s * 32767).astype("<i2")
    return {"scale": s, "data": base64.b64encode(q.tobytes()).decode()}


def spectrogram(sig):
    _, _, Z = stft(sig, fs=FS, nperseg=512, noverlap=256)
    p = np.abs(Z[:256]) ** 2
    p = p.reshape(128, 2, -1).mean(1)  # 128 rows, 0-8 kHz
    db = 10 * np.log10(p + 1e-16)
    top = db.max()
    u8 = np.clip((db - (top - 80)) / 80 * 255, 0, 255).astype(np.uint8)
    u8 = u8[::-1]  # high frequencies first (row 0 = top of the image)
    return {"rows": u8.shape[0], "cols": u8.shape[1], "fmax": FS / 2,
            "hop": 256 / FS, "data": base64.b64encode(u8.tobytes()).decode()}


def edc_db(h, step):
    e = np.cumsum((h ** 2)[::-1])[::-1]
    d = 10 * np.log10(e / e[0] + 1e-12)
    return [round(float(v), 2) for v in d[::step]]


def params(h):
    e = h ** 2
    edc = np.cumsum(e[::-1])[::-1]
    edb = 10 * np.log10(edc / edc[0] + 1e-30)
    t = np.arange(len(h)) / FS

    def slope(lo, hi):
        i0, i1 = np.argmax(edb <= lo), np.argmax(edb <= hi)
        return (t[i1] - t[i0]) * 60 / (lo - hi) if i1 > i0 else None

    pk = int(np.argmax(np.abs(h)))
    w = int(0.0025 * FS)
    direct = e[max(0, pk - w):pk + w].sum()
    n50 = pk + int(0.05 * FS)
    return {"t20": slope(-5, -25), "edt": slope(0, -10),
            "c50": float(10 * np.log10(e[:n50].sum() / max(e[n50:].sum(), 1e-30))),
            "drr": float(10 * np.log10(direct / max(e.sum() - direct, 1e-30)))}


out = {"fs": FS, "speech": os.path.basename(SPEECH), "rooms": []}
for cfg in ROOMS:
    dim = np.array(cfg["dim"])
    e_abs, order = pra.inverse_sabine(cfg["rt60"], dim)
    if cfg["mode"] == "hybrid":
        room = pra.ShoeBox(dim, fs=FS, materials=pra.Material(e_abs, 0.15), max_order=3,
                           ray_tracing=True, air_absorption=True)
    else:
        room = pra.ShoeBox(dim, fs=FS, materials=pra.Material(e_abs), max_order=min(order, 60),
                           air_absorption=True)
    room.add_source(cfg["src"], signal=x)
    mics = np.array([cfg["near"], cfg["far"]]).T
    room.add_microphone_array(mics)
    room.compute_rir()
    rir = [np.asarray(room.rir[m][0], float) for m in range(2)]
    room.compute_anechoic_rir()
    ane = [np.asarray(room.anechoic_rir[m][0], float) for m in range(2)]
    room.compute_anechoic_rir(early_ms=50)
    early = [np.asarray(room.anechoic_rir[m][0], float) for m in range(2)]
    room.compute_anechoic_rir()  # restore
    premix, direct = room.simulate(return_premix=True, return_anechoic=True)
    _, early_sig = room.simulate(return_premix=True, return_anechoic=True, early_ms=50)

    mic_out = []
    for m, label in enumerate(["near", "far"]):
        n_clip = len(x) + int(0.5 * FS)
        rev, dry, er = premix[0, m, :n_clip], direct[0, m, :n_clip], early_sig[0, m, :n_clip]
        g = 0.9 / max(np.abs(rev).max(), np.abs(dry).max(), np.abs(er).max())
        n_plot = min(len(rir[m]), int(1.2 * cfg["rt60"] * FS) + 800)
        step = 32
        d = float(np.linalg.norm(np.array(cfg["src"]) - mics[:, m]))
        fdl2 = pra.constants.get("frac_delay_length") // 2
        mic_out.append({
            "label": label,
            "pos": mics[:, m].tolist(),
            "dist": d,
            "t_direct": (d / room.c * FS + fdl2) / FS,
            "params": params(rir[m]),
            "rir": b64_i16(rir[m][:n_plot]),
            "anechoic": b64_i16(ane[m][:n_plot]),
            "early": b64_i16(early[m][:n_plot]),
            "edc": edc_db(rir[m], step),
            "edc_step": step / FS,
            "audio": {k: b64_wav(s * g) for k, s in
                      [("reverberant", rev), ("direct", dry), ("early", er)]},
            "spec": {k: spectrogram(s * g) for k, s in
                     [("reverberant", rev), ("direct", dry), ("early", er)]},
        })
    out["rooms"].append({
        "id": cfg["id"], "name": cfg["name"], "mode": cfg["mode"], "dim": cfg["dim"],
        "rt60_target": cfg["rt60"], "absorption": float(e_abs),
        "max_order": int(room.max_order), "src": cfg["src"], "mics": mic_out,
    })
    print(cfg["name"], [(m["label"], round(m["params"]["t20"] or -1, 2), round(m["params"]["c50"], 1))
                        for m in mic_out])

data = json.dumps(out, separators=(",", ":"))
template = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "template.html")).read()
with open(OUT, "w") as f:
    f.write(template.replace("__DATA__", data))
print("wrote", OUT, os.path.getsize(OUT), "bytes")
