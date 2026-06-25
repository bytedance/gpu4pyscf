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

import os
import re
import json
import argparse
import numpy as np


def parse_xyz(path):
    """Parse a single ``.xyz`` file.

    Returns
    -------
    (elements, coords) : (list[str], (N,3) float ndarray) in Angstrom.

    Supports the standard XYZ layout (count line, comment line, then
    ``element x y z`` rows) and is tolerant of blank/short files.
    """
    with open(path, "r") as fh:
        raw = [ln.rstrip("\n") for ln in fh]

    lines = [ln for ln in raw if ln.strip() != ""]
    if not lines:
        raise ValueError(f"{path}: empty xyz file")

    # Optional leading atom-count line.
    start = 0
    natm_declared = None
    if re.fullmatch(r"\s*\d+\s*", lines[0]):
        natm_declared = int(lines[0])
        # The line after the count is a free-form comment; skip it if present.
        start = 2 if len(lines) > 1 else 1

    elements, coords = [], []
    for ln in lines[start:]:
        tok = ln.split()
        if len(tok) < 4:
            continue
        sym = tok[0]
        sym = sym[0].upper() + sym[1:].lower() if len(sym) > 1 else sym.upper()
        try:
            x, y, z = float(tok[1]), float(tok[2]), float(tok[3])
        except ValueError:
            continue
        elements.append(sym)
        coords.append([x, y, z])

    if not elements:
        raise ValueError(f"{path}: no atoms parsed")
    if natm_declared is not None and natm_declared != len(elements):
        # Trust the actual parsed rows but warn via the comment in JSON later.
        pass
    return elements, np.asarray(coords, dtype=float)


def molecule_name_from_path(path):
    """Derive a molecule name from the file name (without extension)."""
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    return name


def build_system_entry(path, basis_set, xc_low, xc_high, xc_lda,
                       charge, spin, fragment_id, bond_test_id,
                       bond_shift_flag=False, bond_shift_scale=1.0):
    elements, coords = parse_xyz(path)
    return {
        "element": elements,
        "structure": [[float(x) for x in row] for row in coords],
        "charge": int(charge),
        "spin": int(spin),
        "basis_set": basis_set,
        "xc_lda": xc_lda,
        "xc_low": xc_low,
        "xc_high": xc_high,
        "fragment_id": [int(a) for a in fragment_id],
        "energy_flag": True,        # run the multi-energy block
        "bond_shift_flag": bool(bond_shift_flag),   # apply the bond shift?
        "bond_shift_scale": float(bond_shift_scale),  # ratio for the target bond
        "bond_test_id": [int(a) for a in bond_test_id],  # the [i, j] bond to shift
        "tda_flag": True,           # run the local TDA block
        "population_flag": True,    # run Mulliken + density cube
    }


def generate(xyz_dir, out_path, basis_set="def2-svp", xc_low="pbe",
             xc_high="b3lyp", xc_lda="lda,vwn", charge=0, spin=0,
             fragment_id=None, bond_test_id=None,
             bond_shift_flag=False, bond_shift_scale=1.0):

    if not os.path.isdir(xyz_dir):
        raise NotADirectoryError(xyz_dir)

    fragment_id = list(fragment_id or [])
    bond_test_id = list(bond_test_id or [])

    xyz_files = sorted(
        os.path.join(xyz_dir, f) for f in os.listdir(xyz_dir)
        if f.lower().endswith(".xyz")
    )
    if not xyz_files:
        raise FileNotFoundError(f"No .xyz files found in {xyz_dir}")

    systems = {}
    for path in xyz_files:
        name = molecule_name_from_path(path)
        try:
            systems[name] = build_system_entry(
                path, basis_set, xc_low, xc_high, xc_lda,
                charge=charge, spin=spin, fragment_id=fragment_id,
                bond_test_id=bond_test_id,
                bond_shift_flag=bond_shift_flag,
                bond_shift_scale=bond_shift_scale)
        except Exception as exc:                  # keep going on a bad file
            systems[name] = {"error": f"failed to parse: {exc}"}

    with open(out_path, "w") as fh:
        json.dump(systems, fh, indent=4)
    return systems


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description="Generate test_systems.json from a directory of .xyz files.")
    p.add_argument("--xyz-dir", required=True, help="Directory containing .xyz files.")
    p.add_argument("--out", default="test_systems.json", help="Output JSON path.")
    p.add_argument("--basis", default="def2-svp", help="Basis set for all systems.")
    p.add_argument("--xc-low", default="pbe", help="Low-level (environment) functional.")
    p.add_argument("--xc-high", default="b3lyp", help="High-level (active) functional.")
    p.add_argument("--xc-lda", default="lda,vwn", help="LDA functional label.")
    p.add_argument("--charge", type=int, default=0, help="Total charge for all systems.")
    p.add_argument("--spin", type=int, default=0, help="2S spin for all systems.")
    p.add_argument("--fragment", type=int, nargs="+", required=True,
                   metavar="ATOM",
                   help="Active fragment atom indices (e.g. --fragment 0 1 2).")
    p.add_argument("--bond", type=int, nargs=2, default=None, metavar=("I", "J"),
                   help="The [i, j] atom pair whose bond is shifted.")
    p.add_argument("--bond-shift", action="store_true",
                   help="Enable the single-point bond shift (sets bond_shift_flag).")
    p.add_argument("--bond-shift-scale", type=float, default=1.0,
                   help="Ratio applied to the target bond when --bond-shift is set.")
    return p


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)
    systems = generate(
        args.xyz_dir, args.out, basis_set=args.basis, xc_low=args.xc_low,
        xc_high=args.xc_high, xc_lda=args.xc_lda, charge=args.charge,
        spin=args.spin, fragment_id=args.fragment,
        bond_test_id=(args.bond or []),
        bond_shift_flag=args.bond_shift,
        bond_shift_scale=args.bond_shift_scale)
    n_ok = sum(1 for v in systems.values() if "error" not in v)
    print(f"Wrote {args.out}: {n_ok}/{len(systems)} systems parsed successfully.")
