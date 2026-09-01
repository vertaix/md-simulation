#!/usr/bin/env python3
"""
ATMOS reproduction -- mdCATH structure probe
=============================================

Milestone 1 blocker. h5dump -n gave us the key layout but not shapes, dtypes,
units, or atom names. This answers those so the Appendix E metrics can be
written against the real data rather than assumptions.

Specifically it determines:
  1. coords shape and dtype, so we know frame count and atom count
  2. whether hydrogens are stored (affects every Ca/Cb selection)
  3. how to recover atom names, since there is no atom-name dataset
  4. coordinate units, Angstrom vs nanometre
  5. frame-count consistency across the five replicas
  6. whether our RMSF matches mdCATH's stored rmsf, which is a free unit test

SETUP
-----
    pip install h5py numpy

RUN
---
    python mdcath_probe.py mdcath_dataset_1a02F00.h5

Writes mdcath_probe_report.txt. Send that back.
"""

import argparse
import datetime
import io
import sys

import numpy as np

try:
    import h5py
except ImportError:
    print("FATAL: h5py not installed. Run: pip install h5py")
    sys.exit(1)


TARGET_TEMP = "320"  # ATMOS and TEMPO use the 320 K trajectories


class Tee:
    def __init__(self, path):
        self.buf, self.path, self.stdout = io.StringIO(), path, sys.stdout

    def write(self, s):
        self.stdout.write(s)
        self.buf.write(s)

    def flush(self):
        self.stdout.flush()

    def save(self):
        with open(self.path, "w") as fh:
            fh.write(self.buf.getvalue())


def banner(msg):
    print(f"\n{'=' * 72}\n{msg}\n{'=' * 72}")


def decode(arr):
    """mdCATH stores strings as fixed-width bytes."""
    out = []
    for v in np.asarray(arr).ravel():
        out.append(v.decode("utf-8").strip() if isinstance(v, bytes) else str(v).strip())
    return out


# --------------------------------------------------------------------------
def probe_topology(dom):
    banner("1. Topology (domain level, shared across temps and replicas)")
    info = {}
    for key in ("element", "resid", "resname", "chain", "z"):
        if key not in dom:
            print(f"  [WARN] missing '{key}'")
            continue
        d = dom[key]
        print(f"  {key:<10} shape={str(d.shape):<14} dtype={d.dtype}")
        info[key] = d[()]

    if "z" in info:
        z = np.asarray(info["z"]).ravel()
        n_h = int((z == 1).sum())
        n_tot = z.size
        print(f"\n  Atomic numbers: {n_tot} atoms, {n_h} hydrogens "
              f"({100.0 * n_h / max(n_tot, 1):.1f}%)")
        print(f"  Hydrogens stored: {'YES' if n_h > 0 else 'NO'}")
        if n_h > 0:
            print("  -> Ca/Cb selection must exclude H; heavy atoms = "
                  f"{n_tot - n_h}")

    if "resid" in info:
        resid = np.asarray(info["resid"]).ravel()
        print(f"\n  Residues: {len(np.unique(resid))} unique "
              f"(resid {resid.min()} to {resid.max()})")

    if "resname" in info:
        names = decode(info["resname"])
        print(f"  First 8 resnames: {names[:8]}")
        gly = sum(1 for n in names if n == "GLY")
        print(f"  GLY atoms: {gly}  (Appendix E uses Ca for glycine, Cb otherwise)")

    return info


def probe_atom_names(dom):
    """No atom-name dataset exists, so names must come from pdb or psf."""
    banner("2. Atom names (the key preprocessing problem)")
    for key in ("pdb", "pdbProteinAtoms", "psf"):
        if key not in dom:
            print(f"  [WARN] missing '{key}'")
            continue
        raw = dom[key][()]
        if isinstance(raw, np.ndarray):
            raw = raw.tobytes() if raw.dtype.kind in "SV" else str(raw).encode()
        text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
        lines = text.splitlines()
        print(f"\n  --- {key}: {len(lines)} lines, {len(text)} chars ---")
        for ln in lines[:4]:
            print(f"    {ln[:78]}")

        if key.startswith("pdb"):
            # PDB fixed-width: atom name is columns 13-16
            atom_lines = [l for l in lines if l.startswith(("ATOM", "HETATM"))]
            if atom_lines:
                names = [l[12:16].strip() for l in atom_lines]
                ca = names.count("CA")
                cb = names.count("CB")
                print(f"\n    Parsed {len(atom_lines)} ATOM records")
                print(f"    CA count: {ca}   CB count: {cb}")
                print(f"    First 10 names: {names[:10]}")
                return names
    return None


def probe_trajectory(dom):
    banner(f"3. Trajectory data at {TARGET_TEMP} K")
    if TARGET_TEMP not in dom:
        print(f"  [FAIL] no '{TARGET_TEMP}' group. Present: {list(dom.keys())}")
        return None

    temp = dom[TARGET_TEMP]
    replicas = sorted(temp.keys(), key=lambda x: int(x) if x.isdigit() else x)
    print(f"  Replicas: {replicas}")

    frames = {}
    for r in replicas:
        rep = temp[r]
        print(f"\n  --- replica {r} ---")
        for key in ("coords", "forces", "box", "rmsd", "rmsf",
                    "gyrationRadius", "dssp"):
            if key not in rep:
                continue
            d = rep[key]
            print(f"    {key:<15} shape={str(d.shape):<20} dtype={d.dtype}")
            if key == "coords":
                frames[r] = d.shape[0]

    if frames:
        counts = sorted(set(frames.values()))
        print(f"\n  Frame counts across replicas: {frames}")
        if len(counts) == 1:
            print(f"  [ OK ] consistent: {counts[0]} frames")
        else:
            print(f"  [WARN] INCONSISTENT: {counts}")
            print("         Standardisation policy needed (TEMPO uses 400 frames)")
        print(f"  ATMOS needs 400 frames at 1 ns. Native here: {counts}")
    return temp


def probe_units(temp, topo):
    """
    Ca-Ca virtual bond length is ~3.8 Angstrom = 0.38 nm.
    That distinguishes the two candidate unit systems unambiguously.
    """
    banner("4. Coordinate units")
    rep = temp[sorted(temp.keys())[0]]
    if "coords" not in rep:
        print("  [SKIP] no coords")
        return None

    xyz = rep["coords"][0]  # first frame
    print(f"  Frame 0: shape={xyz.shape}, dtype={xyz.dtype}")
    print(f"  Coordinate range: [{xyz.min():.3f}, {xyz.max():.3f}]")
    print(f"  Bounding box extent: {(xyz.max(axis=0) - xyz.min(axis=0)).round(2)}")

    if "box" in rep:
        print(f"  Box (frame 0): {np.asarray(rep['box'][0]).ravel()[:9].round(3)}")

    # Nearest-neighbour heavy-atom distance is a robust unit probe.
    z = np.asarray(topo.get("z", [])).ravel()
    if z.size == xyz.shape[0]:
        heavy = xyz[z > 1]
        sub = heavy[:400]
        d = np.linalg.norm(sub[:, None, :] - sub[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        nn = np.median(d.min(axis=1))
        print(f"\n  Median nearest-neighbour heavy-atom distance: {nn:.4f}")
        if 1.0 < nn < 2.0:
            print("  [ OK ] units are ANGSTROM (covalent bonds ~1.5 A)")
            return "angstrom"
        if 0.1 < nn < 0.2:
            print("  [ OK ] units are NANOMETRE (covalent bonds ~0.15 nm)")
            return "nanometre"
        print("  [WARN] inconclusive, inspect manually")
    return None


def probe_rmsf_crosscheck(temp, topo, atom_names):
    """
    mdCATH ships precomputed rmsf. Reproducing it validates our RMSF
    implementation and confirms units before we touch the reference column.
    """
    banner("5. RMSF cross-check against mdCATH's stored values")
    rep = temp[sorted(temp.keys())[0]]
    if "rmsf" not in rep or "coords" not in rep:
        print("  [SKIP] missing rmsf or coords")
        return

    stored = np.asarray(rep["rmsf"][()]).ravel()
    xyz = rep["coords"][()]
    n_frames, n_atoms, _ = xyz.shape
    print(f"  stored rmsf: shape={stored.shape}, "
          f"range=[{stored.min():.3f}, {stored.max():.3f}]")
    print(f"  coords: {n_frames} frames x {n_atoms} atoms")

    # Work out what subset the stored rmsf covers.
    z = np.asarray(topo.get("z", [])).ravel()
    n_heavy = int((z > 1).sum()) if z.size else -1
    n_ca = atom_names.count("CA") if atom_names else -1
    print(f"\n  all atoms={n_atoms}  heavy={n_heavy}  CA={n_ca}  "
          f"stored rmsf length={stored.size}")
    if stored.size == n_atoms:
        sel = np.arange(n_atoms); label = "all atoms"
    elif stored.size == n_heavy:
        sel = np.where(z > 1)[0]; label = "heavy atoms"
    elif atom_names and stored.size == n_ca:
        sel = np.array([i for i, n in enumerate(atom_names) if n == "CA"]); label = "CA"
    else:
        print("  [WARN] stored rmsf length matches no obvious subset")
        return
    print(f"  -> stored rmsf appears to be over: {label}")

    # Naive RMSF, no superposition. Expect correlation, not equality.
    sub = xyz[:, sel, :]
    ours = np.sqrt(((sub - sub.mean(axis=0)) ** 2).sum(axis=-1).mean(axis=0))
    r = np.corrcoef(ours, stored)[0, 1]
    ratio = np.median(ours / np.maximum(stored, 1e-8))
    print(f"\n  ours (unaligned): range=[{ours.min():.3f}, {ours.max():.3f}]")
    print(f"  Pearson r vs stored: {r:.4f}")
    print(f"  median ratio ours/stored: {ratio:.4f}")
    if r > 0.9:
        print("  [ OK ] strong correlation, RMSF definition understood")
    else:
        print("  [NOTE] weak correlation, stored version likely superposes first")
    if 0.09 < ratio < 0.11 or 9 < ratio < 11:
        print("  [WARN] ratio near 10x, unit mismatch between coords and rmsf")


def main():
    ap = argparse.ArgumentParser(description="mdCATH structure probe")
    ap.add_argument("h5file", help="path to one mdcath_dataset_*.h5")
    ap.add_argument("--report", default="mdcath_probe_report.txt")
    args = ap.parse_args()

    tee = Tee(args.report)
    sys.stdout = tee
    try:
        print("ATMOS reproduction -- mdCATH structure probe")
        print(f"run: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
        print(f"file: {args.h5file}")
        print(f"h5py {h5py.__version__} | numpy {np.__version__}")

        with h5py.File(args.h5file, "r") as f:
            roots = list(f.keys())
            print(f"root groups: {roots}")
            dom = f[roots[0]]
            print(f"domain: {roots[0]}")
            print(f"temperatures: {[k for k in dom.keys() if k.isdigit()]}")

            topo = probe_topology(dom)
            atom_names = probe_atom_names(dom)
            temp = probe_trajectory(dom)
            if temp is not None:
                probe_units(temp, topo)
                probe_rmsf_crosscheck(temp, topo, atom_names)

        banner("SUMMARY")
        print("  Answers needed before writing the Appendix E metrics:")
        print("    - coords shape and frame count      -> section 3")
        print("    - hydrogens present                 -> section 1")
        print("    - how to get CA/CB indices          -> section 2")
        print("    - units                             -> section 4")
        print("    - RMSF definition validated         -> section 5")
        print(f"\n  Report written to {args.report}")
        return 0
    finally:
        sys.stdout = tee.stdout
        tee.save()


if __name__ == "__main__":
    sys.exit(main())
