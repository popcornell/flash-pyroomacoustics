#!/usr/bin/env python
"""
Compare two dump directories produced by dump.py, scenario by scenario and key
by key.

Usage:
    python compare.py DIR_A DIR_B [--tol] [--rtol 1e-6] [--atol 1e-9]
                      [--only a,b] [--verbose] [--ignore-layout] [--strict-known]

Default mode: everything must be BITWISE identical.  For every array we report
separately
  * ``equal``   : np.array_equal(a, b, equal_nan=True)   (values, NaN==NaN)
  * ``bitwise`` : identical dtype, shape and raw bytes  (a.tobytes()==b.tobytes())
(they differ e.g. for +0.0 vs -0.0 or different NaN payloads).

--tol mode: additionally evaluates np.allclose(a, b, rtol, atol, equal_nan=True)
for numeric arrays, and the exit code is 0 if every key passes under the
tolerance (non-numeric keys must still be equal).

Exit code: 0 only if everything is bitwise identical (or, with --tol, passes
the tolerance), 1 otherwise.

Known nondeterminism of the reference implementation
----------------------------------------------------
KNOWN_NONDETERMINISTIC below lists (scenario-glob, key-glob, reason) entries
whose value is not reproducible run-to-run *in the reference itself*.  Such
keys are compared and reported but do not affect the exit code (unless
--strict-known is passed).
"""

import argparse
import fnmatch
import glob
import os
import sys

import numpy as np

# (scenario glob, key glob, reason)
KNOWN_NONDETERMINISTIC = [
    # The C++ Wall<2> constructor never initializes `basis` (Eigen::Matrix<float,2,2>);
    # Wall<2>::sample_lambertian_reflection() uses it: `basis * (x_l, y_l) + z_l * normal`.
    # Hence every 2D ray-tracing run with a scattering wall (does_scatter == true)
    # computes scattered directions from uninitialized heap memory.  The dump never
    # stores Wall2D.basis values (only its shape/layout) but the scattered ray
    # directions feed the 2D RT histograms and RIRs.  Run-to-run, the garbage is
    # usually the same (same allocation pattern), so these keys are often equal
    # between two reference runs, but there is no guarantee; a re-implementation
    # cannot reproduce them.  See also the scenario comments in scenarios.py.
    ("hybrid2d_scatter", "rt_hist_*", "2D scattering uses uninitialized Wall2D.basis"),
    ("hybrid2d_scatter", "rir_*", "2D scattering uses uninitialized Wall2D.basis"),
    ("hybrid2d_scatter", "signals", "2D scattering uses uninitialized Wall2D.basis"),
    ("hybrid2d_scatter", "eng_mic*_hist0", "2D scattering uses uninitialized Wall2D.basis"),
    ("poly2d_lshape_hybrid_scatter", "rt_hist_*", "2D scattering uses uninitialized Wall2D.basis"),
    ("poly2d_lshape_hybrid_scatter", "rir_*", "2D scattering uses uninitialized Wall2D.basis"),
    ("poly2d_lshape_hybrid_scatter", "signals", "2D scattering uses uninitialized Wall2D.basis"),
    ("poly2d_lshape_hybrid_scatter", "eng_mic*_hist0", "2D scattering uses uninitialized Wall2D.basis"),
]


def known_reason(scen, key):
    for sg, kg, why in KNOWN_NONDETERMINISTIC:
        if fnmatch.fnmatchcase(scen, sg) and fnmatch.fnmatchcase(key, kg):
            return why
    return None


def load(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def is_numeric(a):
    return a.dtype.kind in "biufc"


def compare_arrays(a, b, rtol, atol):
    """Return a dict describing the comparison."""
    r = {
        "shape_a": a.shape,
        "shape_b": b.shape,
        "dtype_a": str(a.dtype),
        "dtype_b": str(b.dtype),
    }
    # strings: '<U11' vs '<U13' only differ by max length -> compare values
    r["dtype_ok"] = a.dtype == b.dtype or (a.dtype.kind in "US" and a.dtype.kind == b.dtype.kind)
    r["shape_ok"] = a.shape == b.shape
    if not r["shape_ok"]:
        r["equal"] = False
        r["bitwise"] = False
        r["tol_ok"] = False
        return r
    try:
        eq_nan = is_numeric(a) and is_numeric(b) and a.dtype.kind in "fc" and b.dtype.kind in "fc"
        r["equal"] = bool(np.array_equal(a, b, equal_nan=eq_nan))
    except Exception:  # noqa
        r["equal"] = False
    if a.dtype.kind in "US":
        r["bitwise"] = bool(r["dtype_ok"] and r["equal"])
    else:
        r["bitwise"] = bool(r["dtype_ok"] and np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(b).tobytes())
    if is_numeric(a) and is_numeric(b) and a.dtype.kind != "b" and b.dtype.kind != "b":
        af = a.astype(np.complex128 if "c" in (a.dtype.kind + b.dtype.kind) else np.float64)
        bf = b.astype(af.dtype)
        both_nan = np.isnan(af) & np.isnan(bf)
        with np.errstate(invalid="ignore", over="ignore"):
            same_inf = np.isinf(af) & (af == bf)
            diff = np.where(both_nan | same_inf, 0.0, np.abs(af - bf))
        if diff.size:
            with np.errstate(invalid="ignore", divide="ignore"):
                denom = np.maximum(np.abs(af), np.abs(bf))
                rel = np.where(diff == 0, 0.0, diff / np.where(denom == 0, np.inf, denom))
            r["max_abs"] = float(np.nanmax(np.where(np.isnan(diff), np.inf, diff)))
            r["max_rel"] = float(np.nanmax(np.where(np.isnan(rel), np.inf, rel)))
            neq = ~((af == bf) | both_nan)
            r["n_diff"] = int(np.count_nonzero(neq))
        else:
            r["max_abs"] = r["max_rel"] = 0.0
            r["n_diff"] = 0
        r["tol_ok"] = bool(np.allclose(af, bf, rtol=rtol, atol=atol, equal_nan=True))
    else:
        r["tol_ok"] = r["equal"]
        if not r["equal"]:
            try:
                r["n_diff"] = int(np.count_nonzero(a != b))
            except Exception:  # noqa
                r["n_diff"] = -1
    return r


def fmt_val(a, n=60):
    if a.dtype.kind in "US" and a.ndim == 0:
        s = str(a)
    else:
        s = np.array2string(a.ravel()[:6], precision=6, threshold=6)
    return s if len(s) <= n else s[: n - 3] + "..."


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir_a")
    ap.add_argument("dir_b")
    ap.add_argument("--tol", action="store_true", help="also judge pass/fail under rtol/atol")
    ap.add_argument("--rtol", type=float, default=1e-6)
    ap.add_argument("--atol", type=float, default=1e-9)
    ap.add_argument("--only", default=None, help="comma separated scenario names")
    ap.add_argument("--verbose", "-v", action="store_true", help="list every key")
    ap.add_argument("--ignore-layout", action="store_true", help="do not count '__layout' keys (memory order / writeable flags)")
    ap.add_argument("--strict-known", action="store_true", help="known-nondeterministic keys also count as failures")
    ap.add_argument("--max-rows", type=int, default=40, help="max mismatching keys printed per scenario")
    args = ap.parse_args()

    fa = {os.path.basename(p)[:-4] for p in glob.glob(os.path.join(args.dir_a, "*.npz"))}
    fb = {os.path.basename(p)[:-4] for p in glob.glob(os.path.join(args.dir_b, "*.npz"))}
    scen = sorted(fa | fb)
    if args.only:
        sel = set(args.only.split(","))
        scen = [s for s in scen if s in sel]

    tot = dict(scen=0, keys=0, bitwise=0, equal_not_bitwise=0, tol_only=0, differ=0, missing=0, shape=0, dtype=0, known=0, layout_skipped=0)
    failing_scen = []
    print(f"A = {args.dir_a}\nB = {args.dir_b}\n")
    hdr = f"{'scenario':42s} {'keys':>5s} {'bitwise':>7s} {'eq!bit':>6s} {'differ':>6s} {'miss':>5s} {'shape':>5s} {'dtype':>5s} {'known':>5s}" + (f" {'tolOK':>5s}" if args.tol else "")
    print(hdr)
    print("-" * len(hdr))
    details = []
    for s in scen:
        tot["scen"] += 1
        if s not in fa or s not in fb:
            print(f"{s:42s} MISSING SCENARIO in {'A' if s not in fa else 'B'}")
            tot["missing"] += 1
            failing_scen.append(s)
            continue
        A = load(os.path.join(args.dir_a, s + ".npz"))
        B = load(os.path.join(args.dir_b, s + ".npz"))
        c = dict(keys=0, bitwise=0, eqnb=0, differ=0, miss=0, shape=0, dtype=0, known=0, tolok=0)
        rows = []
        fail = False
        for k in sorted(set(A) | set(B)):
            if args.ignore_layout and k.endswith("__layout"):
                tot["layout_skipped"] += 1
                continue
            c["keys"] += 1
            why = known_reason(s, k)
            if k not in A or k not in B:
                c["miss"] += 1
                rows.append(f"  MISSING   {k} (only in {'A' if k in A else 'B'})")
                fail = True
                continue
            r = compare_arrays(A[k], B[k], args.rtol, args.atol)
            if r["tol_ok"]:
                c["tolok"] += 1
            if r["bitwise"]:
                c["bitwise"] += 1
                if args.verbose:
                    rows.append(f"  ok        {k}")
                continue
            # not bitwise
            if why is not None and not args.strict_known:
                c["known"] += 1
                rows.append(f"  KNOWN     {k}: {why}" + (f" (max_abs={r.get('max_abs')}, n_diff={r.get('n_diff')})" if "max_abs" in r else ""))
                continue
            if args.tol and r["tol_ok"] and r["dtype_ok"]:
                pass
            else:
                fail = True
            if not r["shape_ok"]:
                c["shape"] += 1
                rows.append(f"  SHAPE     {k}: {r['shape_a']} vs {r['shape_b']} ({r['dtype_a']} vs {r['dtype_b']})")
                continue
            if not r["dtype_ok"]:
                c["dtype"] += 1
                rows.append(f"  DTYPE     {k}: {r['dtype_a']} vs {r['dtype_b']} equal={r['equal']}" + (f" max_abs={r['max_abs']:.3g}" if "max_abs" in r else ""))
                continue
            if r["equal"]:
                c["eqnb"] += 1
                rows.append(f"  EQ!BIT    {k}: values equal but raw bytes differ (signed zeros / NaN payload)")
                continue
            c["differ"] += 1
            if "max_abs" in r:
                tl = f" tol={'ok' if r['tol_ok'] else 'FAIL'}" if args.tol else ""
                rows.append(f"  DIFF      {k}: n_diff={r['n_diff']}/{A[k].size} max_abs={r['max_abs']:.4g} max_rel={r['max_rel']:.4g}{tl}")
            else:
                rows.append(f"  DIFF      {k}: A={fmt_val(A[k])!r} B={fmt_val(B[k])!r}")
        line = f"{s:42s} {c['keys']:5d} {c['bitwise']:7d} {c['eqnb']:6d} {c['differ']:6d} {c['miss']:5d} {c['shape']:5d} {c['dtype']:5d} {c['known']:5d}"
        if args.tol:
            line += f" {c['tolok']:5d}"
        if fail:
            line += "  <-- FAIL"
            failing_scen.append(s)
        if "__error__" in A or "__error__" in B:
            line += "  [__error__ present]"
        print(line)
        if rows and (fail or args.verbose or c["known"]):
            details.append((s, rows))
        tot["keys"] += c["keys"]
        tot["bitwise"] += c["bitwise"]
        tot["equal_not_bitwise"] += c["eqnb"]
        tot["differ"] += c["differ"]
        tot["missing"] += c["miss"]
        tot["shape"] += c["shape"]
        tot["dtype"] += c["dtype"]
        tot["known"] += c["known"]
        tot["tol_only"] += c["tolok"]

    if details:
        print("\nDetails:")
        for s, rows in details:
            print(f"[{s}]")
            for r in rows[: args.max_rows]:
                print(r)
            if len(rows) > args.max_rows:
                print(f"  ... {len(rows) - args.max_rows} more")

    print()
    summary = (
        f"{tot['scen']} scenarios, {tot['keys']} keys, {tot['bitwise']} bitwise-identical, "
        f"{tot['equal_not_bitwise']} equal-but-not-bitwise, {tot['differ']} differ, "
        f"{tot['missing']} missing, {tot['shape']} shape-mismatch, {tot['dtype']} dtype-mismatch, "
        f"{tot['known']} known-nondeterministic"
    )
    if args.tol:
        summary += f", {tot['tol_only']} within tol (rtol={args.rtol}, atol={args.atol})"
    if args.ignore_layout:
        summary += f", {tot['layout_skipped']} layout keys ignored"
    print(summary)
    ok = not failing_scen
    print(("PASS" if ok else f"FAIL ({len(failing_scen)} scenarios: {', '.join(failing_scen[:20])}{' ...' if len(failing_scen) > 20 else ''})")
          + (" [tolerance mode]" if args.tol else " [bitwise mode]"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
