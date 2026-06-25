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
Single-molecule core calculation driver.

Reads ONE molecule record from ``test_systems.json`` and runs the full
validation pipeline:

1.  Global LDA / PBE / B3LYP SCF to convergence.  The converged low-level
    density is used as a 1-step guess for the embedding solvers.
2.  ``B3LYP-in-PBE`` and ``PBE-in-PBE`` embedding (delta method).
3.  Multi-dimensional energies:
      * Global low-level (PBE)            E_global^low
      * Global high-level (B3LYP)         E_global^high
      * Embedding energy                  E_embed
      * Shifted reference energy          E_ref^shifted  (new definition)
4.  Single-point bond test: if ``bond_shift_flag`` is set in the JSON config the
    target bond (``bond_test_id`` = one ``[i, j]`` atom pair) is rescaled by the
    given ``bond_shift_scale`` ratio and the global LDA/PBE/B3LYP plus the
    B3LYP-in-PBE embedding energies are evaluated ONCE on that single geometry
    (no PES scan, no equilibrium fit, no fragment-motion bookkeeping).
5.  Core-region HOMO / LUMO (embedding vs reference).
6.  Local excited states via explicit TDA A-matrix assembly + diagonalisation.
7.  Population & density analysis: Mulliken charges and a
    ``(rho_embedding - rho_global_high)`` density-difference cube.

CRITICAL: every density / orbital quantity that comes out of the inner solver
``mf_inner`` is in the *embedding basis*; this driver always projects it back to
the AO basis with B (via :mod:`embedding_analysis`) before any global property.

All scalar results are written to ``result_<molecule>.json``.  The cube file
(binary) is written alongside it.

Usage
-----
    python run_single_job.py --json test_systems.json --name water \
        --outdir results
"""

import os
import json
import argparse
import traceback

import numpy as np

try:
    from . import embedding_analysis as ea
except Exception:                                   # script-mode import
    import embedding_analysis as ea


# Bohr per Angstrom (pyscf default unit is Angstrom for Mole.atom strings).
ANG2BOHR = 1.8897259886


# ---------------------------------------------------------------------------
# gpu4pyscf is imported lazily so that this module (and the pure helpers /
# config parsing) can be imported on a CPU-only box for unit testing.
# ---------------------------------------------------------------------------
def _import_gpu_stack():
    from pyscf import gto
    from gpu4pyscf.dft import rks
    from gpu4pyscf.qmmm.embedding.embedding_dft import SingleFragmentEmbedding
    return gto, rks, SingleFragmentEmbedding


# ---------------------------------------------------------------------------
# Molecule construction
# ---------------------------------------------------------------------------
def build_mol(config, coords=None):
    """Build a pyscf ``Mole`` from a JSON system record.

    Parameters
    ----------
    config : dict
        One ``test_systems.json`` entry (element / structure / charge / spin /
        basis_set).
    coords : optional (N,3) array
        Override geometry (Angstrom); used by the PES scan.  Defaults to
        ``config['structure']``.
    """
    gto, _, _ = _import_gpu_stack()
    elements = config["element"]
    if coords is None:
        coords = np.asarray(config["structure"], dtype=float)
    else:
        coords = np.asarray(coords, dtype=float)

    atom = [[el, tuple(float(x) for x in xyz)]
            for el, xyz in zip(elements, coords)]

    mol = gto.Mole()
    mol.atom = atom
    mol.unit = "Angstrom"
    mol.basis = config.get("basis_set", "def2-svp")
    mol.charge = int(config.get("charge", 0))
    mol.spin = int(config.get("spin", 0))
    mol.verbose = int(config.get("verbose", 0))
    mol.build()
    return mol


def _make_rks(mol, xc, conv_tol=1e-10):
    _, rks, _ = _import_gpu_stack()
    mf = rks.RKS(mol, xc=xc)
    mf.conv_tol = conv_tol
    return mf


# ---------------------------------------------------------------------------
# Energy block (Task 2.2)
# ---------------------------------------------------------------------------
def run_energy_block(config, mol=None):
    """Global SCF energies + embedding energies + shifted reference energy.

    Returns a JSON-serialisable dict.
    """
    _, rks, SingleFragmentEmbedding = _import_gpu_stack()
    if mol is None:
        mol = build_mol(config)

    fragment = [int(a) for a in config["fragment_id"]]
    xc_lda = config.get("xc_lda", "lda,vwn")
    xc_low = config.get("xc_low", "pbe")
    xc_high = config.get("xc_high", "b3lyp")

    # --- 1. Global low-level functionals to convergence -------------------
    mf_lda = _make_rks(mol, xc_lda)
    e_global_lda = float(mf_lda.kernel())

    mf_pbe = _make_rks(mol, xc_low)
    e_global_low = float(mf_pbe.kernel())

    mf_b3lyp = _make_rks(mol, xc_high)
    e_global_high = float(mf_b3lyp.kernel())

    # 1-step guess: the converged PBE density seeds the embedding solvers.
    dm_guess_low = mf_pbe.make_rdm1()

    # --- 2a. B3LYP-in-PBE embedding --------------------------------------
    mf_outer_bp = _make_rks(mol, xc_low)
    mf_outer_bp.kernel(dm0=dm_guess_low)
    mf_inner_bp = _make_rks(mol, xc_high)
    emb_bp = SingleFragmentEmbedding(mf_outer_bp, mf_inner_bp, fragment)
    e_embed_b3lyp_in_pbe = float(emb_bp.kernel())

    # Shifted reference energy for the B3LYP-in-PBE pairing.
    shift_bp = ea.shifted_reference_energy(mf_outer_bp, mf_inner_bp, emb_bp)

    # --- 2b. PBE-in-PBE embedding (sanity / exactness pairing) -----------
    mf_outer_pp = _make_rks(mol, xc_low)
    mf_outer_pp.kernel(dm0=dm_guess_low)
    mf_inner_pp = _make_rks(mol, xc_low)
    emb_pp = SingleFragmentEmbedding(mf_outer_pp, mf_inner_pp, fragment)
    e_embed_pbe_in_pbe = float(emb_pp.kernel())
    shift_pp = ea.shifted_reference_energy(mf_outer_pp, mf_inner_pp, emb_pp)

    results = {
        "functionals": {"lda": xc_lda, "low": xc_low, "high": xc_high},
        "fragment_id": fragment,
        "n_ao": int(mol.nao_nr()),
        "n_emb_b3lyp_in_pbe": int(emb_bp.B[0].shape[1]),
        "energies": {
            "global_lda": e_global_lda,
            "global_low_pbe": e_global_low,
            "global_high_b3lyp": e_global_high,
            "embed_b3lyp_in_pbe": e_embed_b3lyp_in_pbe,
            "embed_pbe_in_pbe": e_embed_pbe_in_pbe,
            "shifted_reference_b3lyp_in_pbe": shift_bp["e_ref_shifted"],
            "shifted_reference_pbe_in_pbe": shift_pp["e_ref_shifted"],
            "delta_xc_shift_b3lyp_in_pbe": shift_bp["delta_xc_shift"],
            "delta_xc_shift_pbe_in_pbe": shift_pp["delta_xc_shift"],
        },
        # Differences that are physically meaningful for downstream analysis.
        "derived": {
            "embed_minus_global_high": e_embed_b3lyp_in_pbe - e_global_high,
            "shifted_ref_minus_embed_bp":
                shift_bp["e_ref_shifted"] - e_embed_b3lyp_in_pbe,
            "pbe_in_pbe_error": e_embed_pbe_in_pbe - e_global_low,
        },
    }
    # Hand back the live objects so the caller can reuse them for the
    # orbital / TDA / population blocks without re-running SCF.
    live = {
        "mol": mol,
        "mf_lda": mf_lda, "mf_pbe": mf_pbe, "mf_b3lyp": mf_b3lyp,
        "emb_b3lyp_in_pbe": emb_bp, "mf_outer_bp": mf_outer_bp,
        "mf_inner_bp": mf_inner_bp,
        "emb_pbe_in_pbe": emb_pp,
    }
    return results, live


# ---------------------------------------------------------------------------
# Single-point bond test (Task 2.3)
# ---------------------------------------------------------------------------
def resolve_bond_geometry(config):
    """Build the single geometry on which the bond test is evaluated.

    Pure-numpy / no-GPU helper so it stays unit-testable on a CPU box.

    Reads the JSON annotations:
      * ``bond_shift_flag``  -- whether to rescale the target bond at all;
      * ``bond_shift_scale`` -- the ratio applied to that one bond;
      * ``bond_test_id``     -- the ``[i, j]`` atom pair defining the bond.

    Returns ``(coords, info)`` where *coords* is the (N,3) geometry to compute on
    (the original structure when the shift is disabled) and *info* records what
    was done.  When the shift is requested but no valid bond pair is available
    the original geometry is returned with ``applied=False``.
    """
    base_coords = np.asarray(config["structure"], dtype=float)
    shift_flag = bool(config.get("bond_shift_flag", False))
    scale = float(config.get("bond_shift_scale", 1.0))
    bond_test_id = [int(a) for a in config.get("bond_test_id", [])]

    info = {
        "bond_shift_flag": shift_flag,
        "bond_shift_scale": scale,
        "bond_atoms": None,
        "applied": False,
        "r0_angstrom": None,
        "r_shifted_angstrom": None,
    }

    if not shift_flag or len(bond_test_id) < 2:
        return base_coords, info

    ia, ja = bond_test_id[0], bond_test_id[1]
    r0 = float(np.linalg.norm(base_coords[ja] - base_coords[ia]))
    coords = ea.scale_bond(base_coords, ia, ja, scale)
    info.update({
        "bond_atoms": [ia, ja],
        "applied": True,
        "r0_angstrom": r0,
        "r_shifted_angstrom": float(r0 * scale),
    })
    return coords, info


def run_bond_block(config):
    """Single-point bond test driven by the JSON ``bond_shift_*`` annotations.

    The geometry is resolved once (original, or with exactly one bond rescaled),
    then the global LDA/PBE/B3LYP and the B3LYP-in-PBE embedding total energies
    are evaluated a single time on that geometry.  There is deliberately no PES
    scan, no equilibrium fit and no fragment-motion handling.
    """
    coords, info = resolve_bond_geometry(config)
    fragment = [int(a) for a in config["fragment_id"]]

    energies = {
        "global_lda": _safe_global_energy(config, coords, config.get("xc_lda", "lda,vwn")),
        "global_pbe": _safe_global_energy(config, coords, config.get("xc_low", "pbe")),
        "global_b3lyp": _safe_global_energy(config, coords, config.get("xc_high", "b3lyp")),
        "embed_b3lyp_in_pbe": _safe_embed_energy(config, coords, fragment),
    }
    return {"shift": info, "energies": energies}


def _safe_global_energy(config, coords, xc):
    """Global SCF energy at *coords*; returns NaN on SCF failure."""
    try:
        mol = build_mol(config, coords=coords)
        mf = _make_rks(mol, xc)
        return float(mf.kernel())
    except Exception:
        return float("nan")


def _safe_embed_energy(config, coords, fragment):
    """B3LYP-in-PBE embedding energy at *coords*; NaN on failure."""
    try:
        _, _, SingleFragmentEmbedding = _import_gpu_stack()
        mol = build_mol(config, coords=coords)
        mf_outer = _make_rks(mol, config.get("xc_low", "pbe"))
        mf_outer.kernel()
        mf_inner = _make_rks(mol, config.get("xc_high", "b3lyp"))
        emb = SingleFragmentEmbedding(mf_outer, mf_inner, fragment)
        return float(emb.kernel())
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Local orbital energies (Task 2.4)
# ---------------------------------------------------------------------------
def run_orbital_block(live):
    """Core-region HOMO / LUMO from the embedding solver vs the global reference."""
    emb = live["emb_b3lyp_in_pbe"]
    mf_inner = emb.mf_inner[0]
    embed_hl = ea.core_homo_lumo(mf_inner)

    # Reference: the global high-level (B3LYP) frontier orbitals.
    ref_hl = ea.core_homo_lumo(live["mf_b3lyp"])

    return {
        "embedding": embed_hl,
        "reference_global_high": ref_hl,
        "homo_shift": (None if embed_hl["homo"] is None or ref_hl["homo"] is None
                       else embed_hl["homo"] - ref_hl["homo"]),
        "lumo_shift": (None if embed_hl["lumo"] is None or ref_hl["lumo"] is None
                       else embed_hl["lumo"] - ref_hl["lumo"]),
    }


# ---------------------------------------------------------------------------
# Local excited states (Task 2.5)
# ---------------------------------------------------------------------------
def run_tda_block(live, nstates=5):
    """Local TDA excitation energies from explicit A-matrix diagonalisation."""
    emb = live["emb_b3lyp_in_pbe"]
    mf_outer = live["mf_outer_bp"]
    return ea.embedded_tda(emb, mf_outer, ifrag=0, singlet=True, nstates=nstates)


# ---------------------------------------------------------------------------
# Population & density analysis (Task 2.6)
# ---------------------------------------------------------------------------
def run_population_block(config, live, outdir, name):
    """Mulliken charges (on the fragment atoms) + density-difference cube.

    Both quantities use AO-basis densities: the embedding density is built as
    ``D_core + B D_emb B^T`` (full AO), and the global B3LYP density is already
    AO-basis.
    """
    mol = live["mol"]
    emb = live["emb_b3lyp_in_pbe"]
    mf_high = live["mf_b3lyp"]

    s_ao = mf_high.get_ovlp()
    nao = int(mol.nao_nr())

    # Embedding total density projected to the full AO basis.
    dm_embed_ao = ea.full_ao_embedding_density(emb, ifrag=0)
    ea.assert_full_ao(dm_embed_ao, nao, "embedding density")

    dm_global_high_ao = mf_high.make_rdm1()
    ea.assert_full_ao(dm_global_high_ao, nao, "global-high density")

    # Mulliken charges on the requested (fragment) atoms.
    frag_atoms = [int(a) for a in config["fragment_id"]]
    charges_embed = ea.mulliken_charges(mol, dm_embed_ao, s_ao, atom_ids=frag_atoms)
    charges_high = ea.mulliken_charges(mol, dm_global_high_ao, s_ao, atom_ids=frag_atoms)

    # Density-difference cube.
    cube_path = os.path.join(outdir, f"density_diff_{name}.cube")
    try:
        ea.density_difference_cube(mol, dm_embed_ao, dm_global_high_ao, cube_path)
        cube_written = cube_path
    except Exception as exc:
        cube_written = f"cube failed: {exc}"

    return {
        "mulliken_embedding": {str(k): v for k, v in charges_embed.items()},
        "mulliken_global_high": {str(k): v for k, v in charges_high.items()},
        "mulliken_diff": {str(k): charges_embed[k] - charges_high[k]
                          for k in charges_embed},
        "cube_file": cube_written,
    }


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------
def process_single(config, name, outdir="results"):
    """Run every enabled block for one molecule and return the result dict."""
    os.makedirs(outdir, exist_ok=True)
    result = {"molecule_name": name, "status": "ok", "blocks": {}}

    # Energy block (also yields the live SCF/embedding objects).
    if config.get("energy_flag", True):
        energy_res, live = run_energy_block(config)
        result["blocks"]["energies"] = energy_res
    else:
        _, live = run_energy_block(config)  # still needed for later blocks

    # Orbital energies.
    try:
        result["blocks"]["orbitals"] = run_orbital_block(live)
    except Exception as exc:
        result["blocks"]["orbitals"] = {"error": str(exc)}

    # Local TDA.
    if config.get("tda_flag", True):
        try:
            result["blocks"]["tda"] = run_tda_block(live)
        except Exception as exc:
            result["blocks"]["tda"] = {"error": str(exc),
                                       "trace": traceback.format_exc()}

    # Population & density.
    if config.get("population_flag", True):
        try:
            result["blocks"]["population"] = run_population_block(
                config, live, outdir, name)
        except Exception as exc:
            result["blocks"]["population"] = {"error": str(exc),
                                              "trace": traceback.format_exc()}

    # Single-point bond test (driven by the JSON bond_shift_* annotations).
    if config.get("bond_shift_flag", False) or config.get("bond_test_id"):
        try:
            result["blocks"]["bond"] = run_bond_block(config)
        except Exception as exc:
            result["blocks"]["bond"] = {"error": str(exc),
                                        "trace": traceback.format_exc()}

    out_path = os.path.join(outdir, f"result_{name}.json")
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=4)
    result["_output_path"] = out_path
    return result


def load_config(json_path, name):
    """Load a single molecule record from ``test_systems.json``."""
    with open(json_path, "r") as fh:
        systems = json.load(fh)
    if name not in systems:
        raise KeyError(f"'{name}' not found in {json_path}. "
                       f"Available: {list(systems)}")
    return systems[name]


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description="Run the embedding validation pipeline for one molecule.")
    p.add_argument("--json", required=True, help="Path to test_systems.json.")
    p.add_argument("--name", required=True, help="Molecule name (JSON key).")
    p.add_argument("--outdir", default="results", help="Output directory.")
    return p


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)
    config = load_config(args.json, args.name)
    result = process_single(config, args.name, outdir=args.outdir)
    print(f"[{args.name}] status={result['status']} -> {result['_output_path']}")


if __name__ == "__main__":
    main()
