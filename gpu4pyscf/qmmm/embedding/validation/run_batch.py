# Copyright 2025 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Task 3 -- Batch execution dispatcher.

Takes a directory of ``.xyz`` files together with a ``test_systems.json`` (or
generates the JSON on the fly) and runs the single-molecule pipeline of
:mod:`run_single_job` for every system in turn.

Robustness is the whole point: each system is wrapped in ``try/except`` so a
failure (SCF non-convergence, embedding error, ...) for one molecule is recorded
and the loop continues.  A final aggregated report ``batch_report.json`` collects
the per-system status, the key energies and any tracebacks.

Usage
-----
    # Use an existing JSON:
    python run_batch.py --json test_systems.json --xyz-dir ./sample_xyz \
        --outdir results

    # Or auto-generate the JSON from the xyz directory first:
    python run_batch.py --xyz-dir ./sample_xyz --generate --outdir results
"""

import os
import json
import time
import argparse
import traceback

try:
    from . import run_single_job as single
    from . import generate_inputs as gen
except Exception:                                   # script-mode import
    import run_single_job as single
    import generate_inputs as gen


def _summarise_energies(result):
    """Pull the headline energies out of a single-job result for the report."""
    try:
        e = result["blocks"]["energies"]["energies"]
        return {
            "global_low_pbe": e.get("global_low_pbe"),
            "global_high_b3lyp": e.get("global_high_b3lyp"),
            "embed_b3lyp_in_pbe": e.get("embed_b3lyp_in_pbe"),
            "shifted_reference_b3lyp_in_pbe":
                e.get("shifted_reference_b3lyp_in_pbe"),
            "pbe_in_pbe_error":
                result["blocks"]["energies"]["derived"].get("pbe_in_pbe_error"),
        }
    except Exception:
        return {}


def run_batch(systems, outdir="results", names=None):
    """Run the pipeline for every system in *systems* (a dict of records).

    Parameters
    ----------
    systems : dict
        ``{molecule_name: config}`` mapping (the parsed ``test_systems.json``).
    outdir : str
        Directory for the per-system JSON / cube files and the batch report.
    names : optional list[str]
        Restrict the run to these molecule names.

    Returns
    -------
    dict : the aggregated batch report (also written to ``batch_report.json``).
    """
    os.makedirs(outdir, exist_ok=True)
    if names is None:
        names = list(systems.keys())

    report = {
        "n_total": len(names),
        "n_success": 0,
        "n_failed": 0,
        "results": {},
        "failures": {},
    }

    for name in names:
        config = systems.get(name)
        t0 = time.time()
        if config is None or "error" in (config or {}):
            report["n_failed"] += 1
            report["failures"][name] = {
                "error": (config or {}).get("error", "missing config")}
            print(f"[{name}] SKIP (bad config)")
            continue

        try:
            result = single.process_single(config, name, outdir=outdir)
            elapsed = time.time() - t0
            report["n_success"] += 1
            report["results"][name] = {
                "status": "ok",
                "elapsed_sec": round(elapsed, 3),
                "output_path": result.get("_output_path"),
                "energies": _summarise_energies(result),
            }
            print(f"[{name}] OK  ({elapsed:.1f}s)")
        except Exception as exc:
            # A single bad system must NOT abort the batch.
            elapsed = time.time() - t0
            report["n_failed"] += 1
            report["failures"][name] = {
                "status": "failed",
                "elapsed_sec": round(elapsed, 3),
                "error": str(exc),
                "trace": traceback.format_exc(),
            }
            print(f"[{name}] FAILED: {exc}")

    report_path = os.path.join(outdir, "batch_report.json")
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=4)
    report["_report_path"] = report_path

    print(f"\nBatch complete: {report['n_success']} ok, "
          f"{report['n_failed']} failed -> {report_path}")
    return report


def load_or_generate_systems(args):
    """Return the systems dict, generating the JSON first if requested."""
    if args.generate or not args.json or not os.path.exists(args.json):
        out_json = args.json or os.path.join(args.outdir, "test_systems.json")
        os.makedirs(args.outdir, exist_ok=True)
        print(f"Generating {out_json} from {args.xyz_dir} ...")
        systems = gen.generate(
            args.xyz_dir, out_json, basis_set=args.basis, xc_low=args.xc_low,
            xc_high=args.xc_high, xc_lda=args.xc_lda, charge=args.charge,
            spin=args.spin, fragment_id=args.fragment,
            bond_test_id=(args.bond or []),
            bond_shift_flag=args.bond_shift,
            bond_shift_scale=args.bond_shift_scale)
        return systems
    with open(args.json, "r") as fh:
        return json.load(fh)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description="Batch-run the embedding validation pipeline.")
    p.add_argument("--json", default=None,
                   help="Existing test_systems.json (optional with --generate).")
    p.add_argument("--xyz-dir", default=None,
                   help="Directory of .xyz files (required with --generate).")
    p.add_argument("--generate", action="store_true",
                   help="Generate test_systems.json from --xyz-dir before running.")
    p.add_argument("--outdir", default="results", help="Output directory.")
    p.add_argument("--names", nargs="*", default=None,
                   help="Subset of molecule names to run (default: all).")
    # Generation passthrough options.
    p.add_argument("--basis", default="def2-svp")
    p.add_argument("--xc-low", default="pbe")
    p.add_argument("--xc-high", default="b3lyp")
    p.add_argument("--xc-lda", default="lda,vwn")
    p.add_argument("--charge", type=int, default=0)
    p.add_argument("--spin", type=int, default=0)
    p.add_argument("--fragment", type=int, nargs="+", default=None,
                   metavar="ATOM",
                   help="Active fragment atom indices (required with --generate).")
    p.add_argument("--bond", type=int, nargs=2, default=None, metavar=("I", "J"),
                   help="The [i, j] atom pair whose bond is shifted.")
    p.add_argument("--bond-shift", action="store_true",
                   help="Enable the single-point bond shift (sets bond_shift_flag).")
    p.add_argument("--bond-shift-scale", type=float, default=1.0,
                   help="Ratio applied to the target bond when --bond-shift is set.")
    return p


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)
    if args.generate and not args.xyz_dir:
        raise SystemExit("--generate requires --xyz-dir")
    if args.generate and not args.fragment:
        raise SystemExit("--generate requires --fragment")
    if not args.json and not args.xyz_dir:
        raise SystemExit("Provide --json and/or --xyz-dir")

    systems = load_or_generate_systems(args)
    run_batch(systems, outdir=args.outdir, names=args.names)


if __name__ == "__main__":
    main()
