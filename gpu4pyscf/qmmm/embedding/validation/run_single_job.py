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
Batch molecule core calculation driver.

Reads ALL molecule records from a generated ``test_systems.json`` and runs the full
validation pipeline. The output JSON strictly mirrors the input structure, with a
uniform ``results`` dictionary appended to each task.
"""

import os
import json
import argparse
import traceback
import numpy as np
import cupy as cp
from gpu4pyscf.qmmm.embedding.validation import embedding_analysis as ea
from pyscf.data.nist import BOHR

# Bohr per Angstrom (pyscf default unit is Angstrom for Mole.atom strings).
ANG2BOHR = 1 / BOHR

# ---------------------------------------------------------------------------
# gpu4pyscf is imported lazily so that this module (and the pure helpers /
# config parsing) can be imported on a CPU-only box for unit testing.
# ---------------------------------------------------------------------------
def _import_gpu_stack():
    from pyscf import gto
    from gpu4pyscf.dft import rks
    from gpu4pyscf.qmmm.embedding.embedding_dft_try1_ml import SingleFragmentEmbedding_ML, OneStepRKS
    return gto, rks, SingleFragmentEmbedding_ML, OneStepRKS

# ---------------------------------------------------------------------------
# ML Evaluator Factory
# ---------------------------------------------------------------------------
def make_dummy_eval_density_func(guess_xc):
    """
    Creates an evaluation function for OneStepRKS that generates the converged
    density using `guess_xc`, then evaluates the potentials using the target `xc`.
    """
    def eval_func(mol, target_xc, grids):
        from gpu4pyscf.dft import rks
        
        # 1. Obtain converged density based on the specified guess functional
        mf_guess = rks.RKS(mol, xc=guess_xc)
        if grids.coords is not None:
            mf_guess.grids = grids
        mf_guess.verbose = 0
        mf_guess.conv_tol = 1.0E-10
        mf_guess.kernel()
        dm = cp.asarray(mf_guess.make_rdm1())
        
        # 2. Evaluate Exact potentials and energies using target_xc on that guess density
        mf_target = rks.RKS(mol, xc=target_xc)
        if grids.coords is not None:
            mf_target.grids = grids
            
        vj, vk = mf_target.get_jk(mol, dm)
        e_j = 0.5 * float(cp.sum(dm * vj))
        
        is_hybrid = mf_target._numint.libxc.is_hybrid_xc(target_xc)
        if is_hybrid:
            hyb = mf_target._numint.libxc.hybrid_coeff(target_xc, spin=mol.spin)
            vk = vk * hyb
            e_k = 0.5 * float(cp.sum(dm * vk))
        else:
            vk = None
            e_k = 0.0
            
        _, e_xc, vxc = mf_target._numint.nr_rks(mol, grids, target_xc, dm)
        int_rho_vxc = float(cp.sum(dm * vxc))
        
        return vj, vk, vxc, e_j, e_k, float(e_xc), int_rho_vxc
    
    return eval_func

# ---------------------------------------------------------------------------
# Uniform Result Template (Ensures strict key alignment)
# ---------------------------------------------------------------------------
def get_default_results():
    """Define a strictly uniform dictionary structure for all possible properties."""
    return {
        "energies": {
            "functionals": None,
            "fragment_id": None,
            "n_ao": None,
            "n_emb_high_in_low": None,
            "energies": {
                "global_lda": None,
                "global_low": None,
                "global_high": None,
                "embed_high_in_low_low_guess": None,
                "embed_high_in_low_lda_guess": None,
                "embed_low_in_low_low_guess": None,
                "embed_low_in_low_lda_guess": None,
                "shifted_reference_high_in_low": None,
                "shifted_reference_low_in_low": None,
                "delta_xc_shift_high_in_low": None,
                "delta_xc_shift_low_in_low": None,
            },
            "derived": {
                "embed_minus_global_high_low_guess": None,
                "embed_minus_global_high_lda_guess": None,
                "low_in_low_error_low_guess": None,
                "low_in_low_error_lda_guess": None,
            },
            "error": None,
            "trace": None
        },
        "orbitals": {
            "embedding": None,
            "reference_global_high": None,
            "homo_shift": None,
            "lumo_shift": None,
            "error": None,
            "trace": None
        },
        "tda": {
            "excitation_energies_ev": None,
            "eigenvectors": None,
            "oscillator_strengths": None,
            "nocc": None,
            "nvir": None,
            "error": None,
            "trace": None
        },
        "population": {
            "mulliken_embedding": None,
            "mulliken_global_high": None,
            "mulliken_diff": None,
            "cube_file": None,
            "error": None,
            "trace": None
        },
        "bond": {
            "shift": {
                "bond_shift_flag": None,
                "bond_shift_scale": None,
                "bond_atoms": None,
                "applied": None,
                "r0_angstrom": None,
                "r_shifted_angstrom": None
            },
            "energies": {
                "global_lda": None,
                "global_low": None,
                "global_high": None,
                "embed_high_in_low": None
            },
            "error": None,
            "trace": None
        }
    }

# ---------------------------------------------------------------------------
# Molecule construction
# ---------------------------------------------------------------------------
def build_mol(config, coords=None):
    """Build a pyscf ``Mole`` from a JSON system record."""
    gto, _, _, _ = _import_gpu_stack()
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
    _, rks, _, _ = _import_gpu_stack()
    mf = rks.RKS(mol, xc=xc)
    mf.conv_tol = conv_tol
    return mf

# ---------------------------------------------------------------------------
# Energy block (Task 2.2)
# ---------------------------------------------------------------------------
def run_energy_block(config, mol=None):
    """Global SCF energies + ML embedding energies (with 2 different guesses)."""
    _, rks, SingleFragmentEmbedding_ML, OneStepRKS = _import_gpu_stack()
    if mol is None:
        mol = build_mol(config)

    fragment = [int(a) for a in config["fragment_id"]]
    xc_lda = config.get("xc_lda", "svwn")
    xc_low = config.get("xc_low", "pbe")
    xc_high = config.get("xc_high", "b3lyp")

    # --- 1. Global functionalities to convergence -------------------
    mf_lda = _make_rks(mol, xc_lda)
    e_global_lda = float(mf_lda.kernel())

    mf_low = _make_rks(mol, xc_low)
    e_global_low = float(mf_low.kernel())

    mf_high = _make_rks(mol, xc_high)
    e_global_high = float(mf_high.kernel())

    # --- ML Guess functions ---
    eval_low_guess = make_dummy_eval_density_func(xc_low)
    eval_lda_guess = make_dummy_eval_density_func(xc_lda)

    # --- 2a. High-in-Low embedding (Low guess) ------------------------
    mf_outer_hl_low = OneStepRKS(mol, eval_low_guess, xc=xc_low)
    mf_inner_hl_low = _make_rks(mol, xc_high)
    emb_hl_low = SingleFragmentEmbedding_ML(mf_outer_hl_low, mf_inner_hl_low, fragment)
    e_embed_high_in_low_low_guess = float(emb_hl_low.kernel())

    # --- 2b. High-in-Low embedding (LDA guess) ------------------------
    mf_outer_hl_lda = OneStepRKS(mol, eval_lda_guess, xc=xc_low)
    mf_inner_hl_lda = _make_rks(mol, xc_high)
    emb_hl_lda = SingleFragmentEmbedding_ML(mf_outer_hl_lda, mf_inner_hl_lda, fragment)
    e_embed_high_in_low_lda_guess = float(emb_hl_lda.kernel())

    # --- 2c. Low-in-Low embedding (Low guess) --------------------------
    mf_outer_ll_low = OneStepRKS(mol, eval_low_guess, xc=xc_low)
    mf_inner_ll_low = _make_rks(mol, xc_low)
    emb_ll_low = SingleFragmentEmbedding_ML(mf_outer_ll_low, mf_inner_ll_low, fragment)
    e_embed_low_in_low_low_guess = float(emb_ll_low.kernel())

    # --- 2d. Low-in-Low embedding (LDA guess) --------------------------
    mf_outer_ll_lda = OneStepRKS(mol, eval_lda_guess, xc=xc_low)
    mf_inner_ll_lda = _make_rks(mol, xc_low)
    emb_ll_lda = SingleFragmentEmbedding_ML(mf_outer_ll_lda, mf_inner_ll_lda, fragment)
    e_embed_low_in_low_lda_guess = float(emb_ll_lda.kernel())

    # Try shifted reference energy (may fail with CAS-DFT due to missing ONIOM correction attributes)
    try:
        shift_hl = ea.shifted_reference_energy(mf_outer_hl_low, mf_inner_hl_low, emb_hl_low)
    except Exception:
        shift_hl = {"e_ref_shifted": None, "delta_xc_shift": None}
        
    try:
        shift_ll = ea.shifted_reference_energy(mf_outer_ll_low, mf_inner_ll_low, emb_ll_low)
    except Exception:
        shift_ll = {"e_ref_shifted": None, "delta_xc_shift": None}

    results = {
        "functionals": {"lda": xc_lda, "low": xc_low, "high": xc_high},
        "fragment_id": fragment,
        "n_ao": int(mol.nao_nr()),
        "n_emb_high_in_low": int(emb_hl_low.B[0].shape[1]),
        "energies": {
            "global_lda": e_global_lda,
            "global_low": e_global_low,
            "global_high": e_global_high,
            "embed_high_in_low_low_guess": e_embed_high_in_low_low_guess,
            "embed_high_in_low_lda_guess": e_embed_high_in_low_lda_guess,
            "embed_low_in_low_low_guess": e_embed_low_in_low_low_guess,
            "embed_low_in_low_lda_guess": e_embed_low_in_low_lda_guess,
            "shifted_reference_high_in_low": shift_hl["e_ref_shifted"],
            "shifted_reference_low_in_low": shift_ll["e_ref_shifted"],
            "delta_xc_shift_high_in_low": shift_hl["delta_xc_shift"],
            "delta_xc_shift_low_in_low": shift_ll["delta_xc_shift"],
        },
        "derived": {
            "embed_minus_global_high_low_guess": e_embed_high_in_low_low_guess - e_global_high,
            "embed_minus_global_high_lda_guess": e_embed_high_in_low_lda_guess - e_global_high,
            "low_in_low_error_low_guess": e_embed_low_in_low_low_guess - e_global_low,
            "low_in_low_error_lda_guess": e_embed_low_in_low_lda_guess - e_global_low,
        },
    }
    
    # We pass the Low-guess High-in-Low instance to 'live' for downstream Orbital/TDA blocks
    live = {
        "mol": mol,
        "mf_lda": mf_lda, "mf_low": mf_low, "mf_high": mf_high,
        "emb_high_in_low": emb_hl_low, 
        "mf_outer_hl": mf_outer_hl_low,
        "mf_inner_hl": mf_inner_hl_low,
        "emb_low_in_low": emb_ll_low,
    }
    return results, live

# ---------------------------------------------------------------------------
# Single-point bond test (Task 2.3)
# ---------------------------------------------------------------------------
def resolve_bond_geometry(config):
    """Build the single geometry on which the bond test is evaluated."""
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
    """Single-point bond test driven by the JSON ``bond_shift_*`` annotations."""
    coords, info = resolve_bond_geometry(config)
    fragment = [int(a) for a in config["fragment_id"]]

    energies = {
        "global_lda": _safe_global_energy(config, coords, config.get("xc_lda", "lda,vwn")),
        "global_low": _safe_global_energy(config, coords, config.get("xc_low", "pbe")),
        "global_high": _safe_global_energy(config, coords, config.get("xc_high", "b3lyp")),
        "embed_high_in_low": _safe_embed_energy(config, coords, fragment),
    }
    return {"shift": info, "energies": energies}

def _safe_global_energy(config, coords, xc):
    try:
        mol = build_mol(config, coords=coords)
        mf = _make_rks(mol, xc)
        return float(mf.kernel())
    except Exception:
        return float("nan")

def _safe_embed_energy(config, coords, fragment):
    try:
        _, _, SingleFragmentEmbedding_ML, OneStepRKS = _import_gpu_stack()
        mol = build_mol(config, coords=coords)
        
        # Use low xc guess by default for the bond stretch ML embedding
        eval_low = make_dummy_eval_density_func(config.get("xc_low", "pbe"))
        mf_outer = OneStepRKS(mol, eval_low, xc=config.get("xc_low", "pbe"))
        mf_inner = _make_rks(mol, config.get("xc_high", "b3lyp"))
        
        emb = SingleFragmentEmbedding_ML(mf_outer, mf_inner, fragment)
        return float(emb.kernel())
    except Exception:
        return float("nan")

# ---------------------------------------------------------------------------
# Local orbital energies (Task 2.4)
# ---------------------------------------------------------------------------
def run_orbital_block(live):
    """Core-region HOMO / LUMO from the embedding solver vs the global reference."""
    emb = live["emb_high_in_low"]
    mf_inner = emb.mf_inner[0]
    embed_hl = ea.core_homo_lumo(mf_inner)

    ref_hl = ea.core_homo_lumo(live["mf_high"])

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
    emb = live["emb_high_in_low"]
    mf_outer = live["mf_outer_hl"]
    return ea.embedded_tda(emb, mf_outer, ifrag=0, singlet=True, nstates=nstates)

# ---------------------------------------------------------------------------
# Population & density analysis (Task 2.6)
# ---------------------------------------------------------------------------
def run_population_block(config, live, outdir, name):
    """Mulliken charges (on the fragment atoms) + density-difference cube."""
    mol = live["mol"]
    emb = live["emb_high_in_low"]
    mf_high = live["mf_high"]

    s_ao = mf_high.get_ovlp()
    nao = int(mol.nao_nr())

    dm_embed_ao = ea.full_ao_embedding_density(emb, ifrag=0)
    ea.assert_full_ao(dm_embed_ao, nao, "embedding density")

    dm_global_high_ao = mf_high.make_rdm1()
    ea.assert_full_ao(dm_global_high_ao, nao, "global-high density")

    frag_atoms = [int(a) for a in config["fragment_id"]]
    charges_embed = ea.mulliken_charges(mol, dm_embed_ao, s_ao, atom_ids=frag_atoms)
    charges_high = ea.mulliken_charges(mol, dm_global_high_ao, s_ao, atom_ids=frag_atoms)

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
    """Run every enabled block for one molecule and return the uniform result dict."""
    os.makedirs(outdir, exist_ok=True)
    
    # Pre-fill all fields with None to guarantee perfectly matching keys
    blocks = get_default_results()
    
    try:
        # Energy block
        if config.get("energy_flag", True):
            energy_res, live = run_energy_block(config)
            blocks["energies"].update(energy_res)
        else:
            _, live = run_energy_block(config)
    except Exception as exc:
        blocks["energies"]["error"] = str(exc)
        blocks["energies"]["trace"] = traceback.format_exc()
        return blocks, "failed" # Stop here: we need 'live' for remaining blocks

    # Orbital energies
    try:
        orbital_res = run_orbital_block(live)
        blocks["orbitals"].update(orbital_res)
    except Exception as exc:
        blocks["orbitals"]["error"] = str(exc)
        blocks["orbitals"]["trace"] = traceback.format_exc()

    # Local TDA
    if config.get("tda_flag", True):
        try:
            tda_res = run_tda_block(live)
            blocks["tda"].update(tda_res)
        except Exception as exc:
            blocks["tda"]["error"] = str(exc)
            blocks["tda"]["trace"] = traceback.format_exc()

    # Population & density
    if config.get("population_flag", True):
        try:
            pop_res = run_population_block(config, live, outdir, name)
            blocks["population"].update(pop_res)
        except Exception as exc:
            blocks["population"]["error"] = str(exc)
            blocks["population"]["trace"] = traceback.format_exc()

    # Single-point bond test
    if config.get("bond_shift_flag", False) or config.get("bond_test_id"):
        try:
            bond_res = run_bond_block(config)
            blocks["bond"].update(bond_res)
        except Exception as exc:
            blocks["bond"]["error"] = str(exc)
            blocks["bond"]["trace"] = traceback.format_exc()

    return blocks, "ok"

def _build_arg_parser():
    p = argparse.ArgumentParser(
        description="Run the embedding validation pipeline for all tasks in a JSON.")
    p.add_argument("--json", required=True, help="Path to input test_systems.json.")
    p.add_argument("--outjson", default="results.json", help="Path to output JSON.")
    p.add_argument("--outdir", default="results", help="Directory for cube files etc.")
    return p

def main(argv=None):
    args = _build_arg_parser().parse_args(argv)
    
    with open(args.json, "r") as fh:
        systems = json.load(fh)
        
    for task_name, config in systems.items():
        print(f"Processing task: {task_name} ...")
        
        if "error" in config:
            print(f"  Skipping {task_name} due to input parsing error: {config['error']}")
            continue
            
        blocks, status = process_single(config, task_name, outdir=args.outdir)
        
        # Inject uniform results directly into the structure
        config["status"] = status
        config["results"] = blocks
        
    with open(args.outjson, "w") as fh:
        json.dump(systems, fh, indent=4)
        
    print(f"\nAll tasks processed. Results saved to {args.outjson}")

if __name__ == "__main__":
    main()