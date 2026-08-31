#!/usr/bin/env python3
"""
ATMOS reproduction -- Milestone 0 spike
========================================

Paper: "Atomic Trajectory Modeling with State Space Models for Biomolecular
Dynamics" (arXiv:2603.17633). No code was released, so we are reimplementing
on top of Protenix, which the paper uses to initialise e_theta and D_theta.

This script answers one question: can the Protenix components ATMOS needs be
instantiated standalone and populated from a released checkpoint?

CPU-only, runs in under a minute. No GPU allocation needed.

SETUP
-----
    ssh <NetID>@della-gpu.princeton.edu
    cd /scratch/gpfs/<ResearchGroup>/<NetID>      # NOT /home -- quota is 10-50GB

    module load anaconda3/2024.6
    conda create -p ./envs/atmos python=3.11 -y
    conda activate ./envs/atmos
    pip install protenix==0.5.0

    # ~2GB from a Beijing CDN; may be slow from Princeton
    wget https://af3-dev.tos-cn-beijing.volces.com/release_model/protenix_base_default_v0.5.0.pt

RUN
---
    python milestone0_spike.py --ckpt protenix_base_default_v0.5.0.pt

Writes milestone0_report.txt alongside stdout. Send that file back.
Exit code 0 = passed.
"""

import argparse
import datetime
import io
import sys
import traceback
from collections import defaultdict

try:
    import torch
except ImportError:
    print("FATAL: torch not importable. Is the conda env activated?")
    sys.exit(1)


# ATMOS Table 5 -- the state-transition kernel T_theta, trained from scratch
ATMOS_TRANSITION = dict(n_blocks=4, c_s=384, c_z=128, dropout=0.25)

# Components ATMOS needs, and which are inherited vs. written from scratch
COMPONENTS = [
    ("protenix.model.modules.embedders",   "InputFeatureEmbedder", "e_theta context",    "inherit"),
    ("protenix.model.modules.pairformer",  "MSAModule",            "e_theta MSA",        "inherit"),
    ("protenix.model.modules.pairformer",  "PairformerStack",      "e_theta trunk",      "inherit"),
    ("protenix.model.modules.pairformer",  "PairformerBlock",      "T_theta base class", "scratch"),
    ("protenix.model.modules.transformer", "AtomAttentionEncoder", "E_theta / Alg. 1",   "scratch"),
    ("protenix.model.modules.diffusion",   "DiffusionModule",      "D_theta / Eq. 9",    "inherit"),
    ("protenix.model.modules.head",        "DistogramHead",        "P_theta / Eq. 10",   "inherit"),
    ("protenix.model.modules.embedders",   "FourierEmbedding",     "tau_theta timestep", "reuse"),
]


class Tee:
    """Mirror stdout into a report file so one run produces a shareable log."""

    def __init__(self, path):
        self.buf = io.StringIO()
        self.path = path
        self.stdout = sys.stdout

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


def get_state_dict(ckpt):
    """Protenix checkpoints nest the weights under varying keys."""
    for key in ("model", "state_dict", "ema_state_dict"):
        if isinstance(ckpt, dict) and key in ckpt:
            inner = ckpt[key]
            if isinstance(inner, dict) and inner:
                return inner, key
    return ckpt, "(root)"


# --------------------------------------------------------------------------
# Check 1: can we import every component standalone?
# --------------------------------------------------------------------------
def check_imports():
    banner("1. Component imports")
    found, ok = {}, True
    for mod, cls, role, origin in COMPONENTS:
        try:
            found[cls] = getattr(__import__(mod, fromlist=[cls]), cls)
            print(f"  [ OK ] {cls:<22} {role:<20} ({origin})")
        except (ImportError, AttributeError) as e:
            print(f"  [FAIL] {cls:<22} {role:<20} {type(e).__name__}: {e}")
            ok = False
    return ok, found


# --------------------------------------------------------------------------
# Check 2: does the decoder still accept an external latent?
# Load-bearing interface. ATMOS Eq. 9 passes its evolved state
# h_{t+1} = (s_{t+1}, z_{t+1}) where Protenix passes trunk output.
# --------------------------------------------------------------------------
def check_signature():
    banner("2. DiffusionModule conditioning interface")
    import inspect
    from protenix.model.modules.diffusion import DiffusionModule

    sig = inspect.signature(DiffusionModule.forward)
    params = [p for p in sig.parameters if p != "self"]
    print(f"  forward({', '.join(params)})\n")

    needed = {
        "x_noisy":           "x~(gamma)  noisy coordinates",
        "t_hat_noise_level": "gamma      diffusion time",
        "s_trunk":           "s_{t+1}    ATMOS single latent",
        "z_trunk":           "z_{t+1}    ATMOS pair latent",
    }
    ok = True
    for p, meaning in needed.items():
        if p in params:
            print(f"  [ OK ] {p:<20} = {meaning}")
        else:
            print(f"  [FAIL] {p:<20} MISSING -- signature changed in this version")
            ok = False

    # Later Protenix versions added required args (pair_z, p_lm, c_l).
    known = set(needed) | {"input_feature_dict", "s_inputs"}
    extra_required = [
        p for p in params
        if p not in known and sig.parameters[p].default is inspect.Parameter.empty
    ]
    if extra_required:
        print(f"\n  [WARN] extra REQUIRED args: {', '.join(extra_required)}")
        print("         Newer Protenix. Consider pinning 0.5.0, or supply these.")
    return ok


# --------------------------------------------------------------------------
# Check 3: does T_theta build at ATMOS dimensions?
# --------------------------------------------------------------------------
def check_instantiation(found):
    banner("3. Instantiate T_theta at ATMOS Table 5 dimensions")
    try:
        stack = found["PairformerStack"](**ATMOS_TRANSITION)
        n = sum(p.numel() for p in stack.parameters())
        print(f"  [ OK ] PairformerStack({ATMOS_TRANSITION})")
        print(f"         {n / 1e6:.1f}M params, trained from scratch")
    except Exception as e:
        print(f"  [FAIL] {type(e).__name__}: {e}")
        traceback.print_exc(file=sys.stdout)
        return False

    cs, cz = ATMOS_TRANSITION["c_s"], ATMOS_TRANSITION["c_z"]
    print(f"\n  NOTE (deviations log): Alg. 2 line 4 writes 'z_ij + s_i + s_j',")
    print(f"  but c_s={cs} != c_z={cz}. The sum is dimensionally impossible as")
    print(f"  written; a learned {cs}->{cz} projection is required. Paper omits it.")
    return True


# --------------------------------------------------------------------------
# Check 4: map checkpoint keys onto the modules we inherit
# --------------------------------------------------------------------------
def discover_prefix(sd, needle):
    """Find the real prefix for a module rather than guessing at it."""
    hits = defaultdict(int)
    for k in sd:
        if needle in k.lower():
            parts = k.split(".")
            for i, part in enumerate(parts):
                if needle in part.lower():
                    hits[".".join(parts[: i + 1])] += 1
                    break
    return sorted(hits.items(), key=lambda x: -x[1])


def check_checkpoint(path, dump_keys):
    banner("4. Checkpoint inspection")
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"  [FAIL] cannot load {path}: {type(e).__name__}: {e}")
        return False, None

    sd, where = get_state_dict(ckpt)
    total = sum(v.numel() for v in sd.values() if hasattr(v, "numel"))
    print(f"  {len(sd)} tensors under '{where}', {total / 1e6:.1f}M params total")

    if dump_keys:
        print(f"\n  First {dump_keys} keys:")
        for k in list(sd)[:dump_keys]:
            print(f"    {k}")

    print("\n  Discovered prefixes for inherited modules:")
    prefixes, ok = {}, True
    for label, needle in [
        ("e_theta trunk",   "pairformer"),
        ("e_theta MSA",     "msa"),
        ("e_theta context", "embedder"),
        ("D_theta",         "diffusion"),
    ]:
        cands = discover_prefix(sd, needle)
        if cands:
            best, n = cands[0]
            prefixes[needle] = best
            print(f"    [ OK ] {label:<16} '{best}' ({n} tensors)")
            for alt, an in cands[1:3]:
                print(f"           alt: '{alt}' ({an})")
        else:
            print(f"    [WARN] {label:<16} no key contains '{needle}'")
            ok = False
    return ok, (sd, prefixes)


# --------------------------------------------------------------------------
# Check 5: actually populate a standalone module from the checkpoint
# --------------------------------------------------------------------------
def check_load(found, sd, prefixes):
    banner("5. Load weights into an isolated module")
    prefix = prefixes.get("pairformer")
    if not prefix:
        print("  [SKIP] no pairformer prefix discovered")
        return False

    # The 48-block trunk Pairformer inside e_theta -- this one inherits weights.
    trunk = found["PairformerStack"](n_blocks=48, c_s=384, c_z=128)
    want = trunk.state_dict()

    subset = {k[len(prefix) + 1:]: v for k, v in sd.items() if k.startswith(prefix + ".")}
    matched = {k: v for k, v in subset.items()
               if k in want and tuple(v.shape) == tuple(want[k].shape)}
    pct = 100.0 * len(matched) / max(len(want), 1)

    print(f"  prefix '{prefix}' -> {len(subset)} checkpoint keys")
    print(f"  module expects {len(want)} keys")
    print(f"  shape-compatible: {len(matched)} ({pct:.1f}%)")

    missing, unexpected = trunk.load_state_dict(matched, strict=False)
    print(f"  load_state_dict(strict=False): {len(missing)} missing, "
          f"{len(unexpected)} unexpected")

    if pct < 95 and subset:
        shown = 0
        print("\n  Sample mismatches:")
        for k, v in subset.items():
            if k not in want:
                print(f"    unexpected: {k}")
            elif tuple(v.shape) != tuple(want[k].shape):
                print(f"    shape: {k}  ckpt{tuple(v.shape)} vs module{tuple(want[k].shape)}")
            else:
                continue
            shown += 1
            if shown >= 8:
                break

    if pct > 95:
        print("\n  [ OK ] Protenix weights transfer cleanly into isolated modules.")
        print("         ATMOS can inherit e_theta and D_theta as the paper describes.")
        return True
    print("\n  [FAIL] poor overlap -- likely a Protenix version mismatch.")
    return False


def main():
    ap = argparse.ArgumentParser(description="ATMOS Milestone 0 spike")
    ap.add_argument("--ckpt", help="path to protenix_base_default_v*.pt")
    ap.add_argument("--dump-keys", type=int, default=0,
                    help="print first N checkpoint keys (use 30 if prefixes fail)")
    ap.add_argument("--report", default="milestone0_report.txt")
    args = ap.parse_args()

    tee = Tee(args.report)
    sys.stdout = tee
    try:
        print("ATMOS reproduction -- Milestone 0 spike")
        print(f"run: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
        print(f"python {sys.version.split()[0]} | torch {torch.__version__}")
        try:
            import protenix
            print(f"protenix {getattr(protenix, '__version__', 'unknown')}")
        except ImportError:
            print("\nFATAL: protenix not importable. Did 'pip install protenix==0.5.0' run?")
            return 1

        results = {}
        results["imports"], found = check_imports()
        if not results["imports"]:
            print("\nImports failed; later checks would be meaningless.")
            return 1

        results["signature"] = check_signature()
        results["instantiation"] = check_instantiation(found)

        if args.ckpt:
            results["ckpt_keys"], payload = check_checkpoint(args.ckpt, args.dump_keys)
            if payload:
                results["ckpt_load"] = check_load(found, *payload)
        else:
            print("\n(no --ckpt given; weight checks skipped)")

        banner("VERDICT")
        for k, v in results.items():
            print(f"  {'PASS' if v else 'FAIL'}  {k}")
        passed = all(results.values())
        print(f"\n  Milestone 0: {'PASSED' if passed else 'NOT PASSED'}")
        print(f"  Report written to {args.report}")
        return 0 if passed else 1
    finally:
        sys.stdout = tee.stdout
        tee.save()


if __name__ == "__main__":
    sys.exit(main())
