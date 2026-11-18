"""Paired comparison of bench/validate.py outputs.

Usage:
    python compare_validate.py "ref_s{}.npz" "fast_s{}.npz" 3 4 5 6

For every room kind and acoustic parameter (T20, EDT, C50, DRR, energy), prints the
mean difference B - A over rooms x mics x seeds, its standard error, and the
seed-to-seed standard deviation of A (the noise floor of the original itself).
The room geometries are the same in A and B when both were generated with the
same KINDS list (the geometry seed depends on the position in KINDS).
"""
import sys

import numpy as np

NAMES = ["T20 [s]", "EDT [s]", "C50 [dB]", "DRR [dB]", "energy [dB]"]
pat_a, pat_b, seeds = sys.argv[1], sys.argv[2], sys.argv[3:]
A = [np.load(pat_a.format(s)) for s in seeds]
B = [np.load(pat_b.format(s)) for s in seeds]
kinds = sorted(k[:-7] for k in A[0].files if k.endswith("_params") and k in B[0].files)
for kind in kinds:
    pa = np.stack([a[kind + "_params"] for a in A])  # seeds x pairs x params
    pb = np.stack([b[kind + "_params"] for b in B])
    print(f"== {kind}")
    for j, name in enumerate(NAMES):
        d = (pb[..., j] - pa[..., j]).ravel()
        d = d[np.isfinite(d)]
        se = d.std(ddof=1) / np.sqrt(len(d))
        noise = np.nanmean(np.nanstd(pa[..., j], axis=0, ddof=1)) if len(seeds) > 1 else np.nan
        print(f"  {name:12s} diff {d.mean():+8.4f}  SE {se:7.4f}  ({d.mean() / se:+5.1f} SE)"
              f"  seed-to-seed std of A {noise:7.4f}")
