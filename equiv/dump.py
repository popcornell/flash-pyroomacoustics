#!/usr/bin/env python
"""
Run the equivalence scenarios with whichever pyroomacoustics is importable and
save one ``OUTDIR/<scenario>.npz`` per scenario (allow_pickle=False compatible).

Usage:
    python dump.py OUTDIR [--only name1,name2] [--list] [--category cat1,cat2]

Run it from a directory that does not contain a ``pyroomacoustics/`` folder.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import scenarios  # noqa: E402


def to_array(v):
    """Convert a scenario value to a numpy array storable without pickle."""
    if isinstance(v, np.ndarray):
        if v.dtype == object:
            raise TypeError("object arrays are not allowed")
        return v
    if isinstance(v, (bool, np.bool_)):
        return np.array(v, dtype=np.bool_)
    if isinstance(v, str):
        return np.array(v, dtype=np.str_)
    if isinstance(v, (int, float, complex, np.generic)):
        return np.asarray(v)
    raise TypeError(f"unsupported value type {type(v).__name__}")


def save(path, out):
    arrays = {}
    for k, v in out.items():
        try:
            arrays[k] = to_array(v)
        except Exception as e:  # noqa
            arrays[k] = np.array(f"__unstorable__ {type(v).__name__}: {e}", dtype=np.str_)
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)
    # sanity: must load without pickle
    with np.load(path, allow_pickle=False) as f:
        _ = list(f.keys())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outdir", nargs="?")
    ap.add_argument("--only", default=None, help="comma separated scenario names")
    ap.add_argument("--category", default=None, help="comma separated categories")
    ap.add_argument("--list", action="store_true", help="list scenarios and exit")
    args = ap.parse_args()

    import pyroomacoustics as pra

    if args.list:
        for n, c in scenarios.list_scenarios():
            print(f"{c:12s} {n}")
        return 0
    if args.outdir is None:
        ap.error("OUTDIR is required")

    names = [n for n, _ in scenarios.list_scenarios()]
    if args.only:
        req = [s.strip() for s in args.only.split(",") if s.strip()]
        unknown = [s for s in req if s not in scenarios.SCENARIOS]
        if unknown:
            ap.error(f"unknown scenarios: {unknown}")
        names = req
    if args.category:
        cats = set(args.category.split(","))
        names = [n for n in names if scenarios.SCENARIOS[n][0] in cats]

    os.makedirs(args.outdir, exist_ok=True)
    print(f"pyroomacoustics {getattr(pra, '__version__', '?')} from {os.path.dirname(pra.__file__)}")
    print(f"libroom: {getattr(pra.libroom, '__file__', '?')}")

    timings = {}
    n_err = 0
    t_all = time.perf_counter()
    for name in names:
        t0 = time.perf_counter()
        out = scenarios.run_scenario(name)
        dt = time.perf_counter() - t0
        save(os.path.join(args.outdir, f"{name}.npz"), out)
        timings[name] = dt
        status = ""
        if "__error__" in out:
            n_err += 1
            status = f"  ERROR {out['__error__'][:150]}"
        elif "__skipped__" in out:
            status = f"  skipped: {out['__skipped__']}"
        print(f"{dt:8.2f}s  {name:45s} {len(out):5d} keys{status}", flush=True)
    total = time.perf_counter() - t_all

    meta_path = os.path.join(args.outdir, "timings.json")
    old = {}
    if args.only or args.category:
        try:
            with open(meta_path) as f:
                old = json.load(f).get("scenarios", {})
        except Exception:  # noqa
            old = {}
    old.update(timings)
    with open(meta_path, "w") as f:
        json.dump(
            {
                "pyroomacoustics_version": getattr(pra, "__version__", "?"),
                "pyroomacoustics_path": os.path.dirname(pra.__file__),
                "total_seconds_last_run": total,
                "scenarios": old,
            },
            f,
            indent=1,
        )

    print(f"\n{len(names)} scenarios in {total:.1f}s, {n_err} with __error__")
    print("slowest:")
    for n, t in sorted(timings.items(), key=lambda x: -x[1])[:8]:
        print(f"  {t:8.2f}s  {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
