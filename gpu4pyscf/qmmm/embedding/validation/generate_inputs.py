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


# Minimal covalent radii (Angstrom) for the simple bond-perception heuristic.
_COVALENT_RADII = {
    "H": 0.31, "He": 0.28, "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76,
    "N": 0.71, "O": 0.66, "F": 0.57, "Ne": 0.58, "Na": 1.66, "Mg": 1.41,
    "Al": 1.21, "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02, "Ar": 1.06,
    "K": 2.03, "Ca": 1.76, "Br": 1.20, "I": 1.39,
}
_DEFAULT_RADIUS = 0.77


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


def perceive_bonds(elements, coords, tol=1.3):
    """Very small covalent bond-perception used to pick fragments / test bonds.

    A pair (i, j) is bonded when ``|r_i - r_j| < tol * (R_i + R_j)``.

    Returns a list of (i, j) index pairs with i < j.
    """
    n = len(elements)
    radii = np.array([_COVALENT_RADII.get(e, _DEFAULT_RADIUS) for e in elements])
    bonds = []
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(coords[i] - coords[j]))
            if d < tol * (radii[i] + radii[j]):
                bonds.append((i, j))
    return bonds


def auto_fragment(elements, bonds):
    """Heuristic active fragment: first heavy atom + its bonded hydrogens.

    Falls back to the first two atoms for tiny / all-hydrogen systems.
    """
    heavy = [i for i, e in enumerate(elements) if e != "H"]
    if not heavy:
        return list(range(min(2, len(elements))))

    center = heavy[0]
    frag = {center}
    for (i, j) in bonds:
        if i == center and elements[j] == "H":
            frag.add(j)
        elif j == center and elements[i] == "H":
            frag.add(i)
    # Guarantee at least two atoms so the embedding has a non-trivial region.
    if len(frag) < 2 and len(elements) >= 2:
        frag.add(1 if center != 1 else 0)
    return sorted(frag)


def auto_bond_test_ids(bonds, elements, max_bonds=1):
    """Pick representative heavy-atom bonds (flattened index pairs) for the PES.

    Returns a flat list ``[i0, j0, i1, j1, ...]`` (matching the example schema
    where ``bond_test_id`` is a flat list of atom indices defining the bonds to
    scan).  Prefers heavy-heavy bonds; falls back to any bond.
    """
    heavy_bonds = [(i, j) for (i, j) in bonds
                   if elements[i] != "H" and elements[j] != "H"]
    chosen = (heavy_bonds or bonds)[:max_bonds]
    flat = []
    for (i, j) in chosen:
        flat.extend([int(i), int(j)])
    return flat


def guess_spin(elements, charge):
    """Spin (2S, pyscf convention) guess from electron count parity.

    Even electron count -> singlet (0); odd -> doublet (1).
    """
    z_of = {
        "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
        "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
        "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Br": 35, "I": 53,
    }
    nelec = sum(z_of.get(e, 0) for e in elements) - charge
    return nelec % 2


def build_system_entry(path, basis_set, xc_low, xc_high, xc_lda,
                       charge=0, spin=None, bond_shift_flag=False,
                       bond_shift_scale=1.0):
    """Assemble one ``test_systems.json`` record for a single ``.xyz`` file.

    The bond test is a *single-point* calculation: ``bond_shift_flag`` toggles
    whether the target bond (``bond_test_id`` = a single ``[i, j]`` atom pair)
    is rescaled, and ``bond_shift_scale`` is the ratio applied to that one bond.
    No PES scan / equilibrium fit / fragment motion is performed.
    """
    elements, coords = parse_xyz(path)
    bonds = perceive_bonds(elements, coords)
    fragment = auto_fragment(elements, bonds)
    bond_test_id = auto_bond_test_ids(bonds, elements)
    if spin is None:
        spin = guess_spin(elements, charge)

    return {
        "element": elements,
        "structure": [[float(x) for x in row] for row in coords],
        "charge": int(charge),
        "spin": int(spin),
        "basis_set": basis_set,
        # Three functionals drive the pipeline: LDA + PBE globals, B3LYP high level.
        "xc_lda": xc_lda,
        "xc_low": xc_low,
        "xc_high": xc_high,
        "fragment_id": [int(a) for a in fragment],
        "energy_flag": True,        # run the multi-energy block
        # --- single-point bond test annotations ---------------------------
        "bond_shift_flag": bool(bond_shift_flag),   # apply the bond shift?
        "bond_shift_scale": float(bond_shift_scale),  # ratio for the target bond
        "bond_test_id": bond_test_id,               # the [i, j] bond to shift
        "tda_flag": True,           # run the local TDA block
        "population_flag": True,    # run Mulliken + density cube
    }


def molecule_name_from_path(path):
    """Derive a molecule name from the file name (without extension)."""
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    return name


def generate(xyz_dir, out_path, basis_set="def2-svp", xc_low="pbe",
             xc_high="b3lyp", xc_lda="lda,vwn", charge=0, spin=None,
             bond_shift_flag=False, bond_shift_scale=1.0):
    """Scan *xyz_dir* and write the aggregated ``test_systems.json``.

    Returns the dictionary that was written.
    """
    if not os.path.isdir(xyz_dir):
        raise NotADirectoryError(xyz_dir)

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
                charge=charge, spin=spin,
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
    p.add_argument("--spin", type=int, default=None,
                   help="2S spin for all systems (default: parity guess).")
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
        spin=args.spin, bond_shift_flag=args.bond_shift,
        bond_shift_scale=args.bond_shift_scale)
    n_ok = sum(1 for v in systems.values() if "error" not in v)
    print(f"Wrote {args.out}: {n_ok}/{len(systems)} systems parsed successfully.")


if __name__ == "__main__":
    main()
