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
                "shifted_reference_high_in_low": None,
                "delta_xc_shift_high_in_low": None,
            },
            "derived": {
                "embed_minus_global_high_low_guess": None,
                "embed_minus_global_high_lda_guess": None,
            },
            "error": None,
            "trace": None
        },
        "orbitals": {
            "embedding_low_guess": None,
            "embedding_lda_guess": None,
            "reference_global_high": None,
            "reference_global_low": None, # Added missing PBE reference
            "reference_global_lda": None, # Added missing LDA reference
            "homo_shift_low_guess": None,
            "lumo_shift_low_guess": None,
            "homo_shift_lda_guess": None,
            "lumo_shift_lda_guess": None,
            "error": None,
            "trace": None
        },
        "tda": {
            "excitation_energies_ev_low_guess": None,
            "excitation_energies_ev_lda_guess": None,
            "excitation_energies_ev_ref_high": None, 
            "excitation_energies_ev_ref_low": None,  
            "excitation_energies_ev_ref_lda": None,  
            "oscillator_strengths_low_guess": None,
            "oscillator_strengths_lda_guess": None,
            "nocc_low_guess": None,
            "nvir_low_guess": None,
            "nocc_lda_guess": None,
            "nvir_lda_guess": None,
            "error": None,
            "trace": None
        },
        "population": {
            "mulliken_embedding_low_guess": None,
            "mulliken_embedding_lda_guess": None,
            "mulliken_global_high": None,
            "mulliken_global_low": None,
            "mulliken_global_lda": None, # Added missing LDA reference
            "mulliken_diff_low_guess": None,
            "mulliken_diff_lda_guess": None,
            "cube_file_low_guess": None,
            "cube_file_lda_guess": None,
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
    mol.verbose = int(config.get("verbose", 4))
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
    emb_hl_low = SingleFragmentEmbedding_ML(mf_outer_hl_low, mf_inner_hl_low, fragment, verbose=4)
    e_embed_high_in_low_low_guess = float(emb_hl_low.kernel())

    # --- 2b. High-in-Low embedding (LDA guess) ------------------------
    mf_outer_hl_lda = OneStepRKS(mol, eval_lda_guess, xc=xc_low)
    mf_inner_hl_lda = _make_rks(mol, xc_high)
    emb_hl_lda = SingleFragmentEmbedding_ML(mf_outer_hl_lda, mf_inner_hl_lda, fragment, verbose=4)
    e_embed_high_in_low_lda_guess = float(emb_hl_lda.kernel())

    # Try shifted reference energy (may fail with CAS-DFT due to missing ONIOM correction attributes)
    try:
        shift_hl = ea.shifted_reference_energy(mf_outer_hl_low, mf_inner_hl_low, emb_hl_low)
    except Exception:
        shift_hl = {"e_ref_shifted": None, "delta_xc_shift": None}

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
            "shifted_reference_high_in_low": shift_hl["e_ref_shifted"],
            "delta_xc_shift_high_in_low": shift_hl["delta_xc_shift"],
        },
        "derived": {
            "embed_minus_global_high_low_guess": e_embed_high_in_low_low_guess - e_global_high,
            "embed_minus_global_high_lda_guess": e_embed_high_in_low_lda_guess - e_global_high,
        },
    }
    
    # Passing both Low-guess and LDA-guess to 'live' for downstream blocks
    live = {
        "mol": mol,
        "mf_lda": mf_lda, "mf_low": mf_low, "mf_high": mf_high,
        "emb_high_in_low": emb_hl_low, 
        "emb_high_in_low_lda": emb_hl_lda,
        "mf_outer_hl": mf_outer_hl_low,
        "mf_outer_hl_lda": mf_outer_hl_lda,
        "mf_inner_hl": mf_inner_hl_low,
        "mf_inner_hl_lda": mf_inner_hl_lda,
    }
    return results, live

# ---------------------------------------------------------------------------
# Local orbital energies (Task 2.4)
# ---------------------------------------------------------------------------
def run_orbital_block(live):
    """Core-region HOMO / LUMO from both embedding solvers vs the global references."""
    # 1. Embedded Orbitals - Low Guess
    emb_low = live["emb_high_in_low"]
    embed_hl_low = ea.core_homo_lumo(emb_low.mf_inner[0])
    
    # 2. Embedded Orbitals - LDA Guess
    emb_lda = live["emb_high_in_low_lda"]
    embed_hl_lda = ea.core_homo_lumo(emb_lda.mf_inner[0])

    # 3. Reference Orbitals
    ref_high = ea.core_homo_lumo(live["mf_high"])
    ref_low = ea.core_homo_lumo(live["mf_low"])
    ref_lda = ea.core_homo_lumo(live["mf_lda"])

    return {
        "embedding_low_guess": embed_hl_low,
        "embedding_lda_guess": embed_hl_lda,
        "reference_global_high": ref_high,
        "reference_global_low": ref_low,
        "reference_global_lda": ref_lda,
        # Note: Shifts remain calculated explicitly vs global_high target as default tracking
        "homo_shift_low_guess": (None if embed_hl_low["homo"] is None or ref_high["homo"] is None
                                 else embed_hl_low["homo"] - ref_high["homo"]),
        "lumo_shift_low_guess": (None if embed_hl_low["lumo"] is None or ref_high["lumo"] is None
                                 else embed_hl_low["lumo"] - ref_high["lumo"]),
        "homo_shift_lda_guess": (None if embed_hl_lda["homo"] is None or ref_high["homo"] is None
                                 else embed_hl_lda["homo"] - ref_high["homo"]),
        "lumo_shift_lda_guess": (None if embed_hl_lda["lumo"] is None or ref_high["lumo"] is None
                                 else embed_hl_lda["lumo"] - ref_high["lumo"]),
    }

# ---------------------------------------------------------------------------
# Local excited states (Task 2.5)
# ---------------------------------------------------------------------------
def run_tda_block(live, nstates=5):
    """Local TDA excitation energies from explicit A-matrix diagonalisation and global references."""
    from gpu4pyscf import tdscf
    from pyscf.data.nist import HARTREE2EV

    # 1a. Embedded TDA - Low Guess
    emb_low = live["emb_high_in_low"]
    mf_outer_low = live["mf_outer_hl"]
    res_low = ea.embedded_tda(emb_low, mf_outer_low, ifrag=0, singlet=True, nstates=nstates)

    # 1b. Embedded TDA - LDA Guess
    emb_lda = live["emb_high_in_low_lda"]
    mf_outer_lda = live["mf_outer_hl_lda"]
    res_lda = ea.embedded_tda(emb_lda, mf_outer_lda, ifrag=0, singlet=True, nstates=nstates)

    # Re-structure for uniform keys (removes eigenvectors implicitly)
    res = {
        "excitation_energies_ev_low_guess": res_low.get("excitation_energies_ev"),
        "oscillator_strengths_low_guess": res_low.get("oscillator_strengths"),
        "nocc_low_guess": res_low.get("nocc"),
        "nvir_low_guess": res_low.get("nvir"),
        
        "excitation_energies_ev_lda_guess": res_lda.get("excitation_energies_ev"),
        "oscillator_strengths_lda_guess": res_lda.get("oscillator_strengths"),
        "nocc_lda_guess": res_lda.get("nocc"),
        "nvir_lda_guess": res_lda.get("nvir"),
    }

    # 2. Reference TDA - Global High
    td_high = live["mf_high"].TDA()
    td_high.nstates = nstates
    td_high.kernel()
    if td_high.e is not None:
        res["excitation_energies_ev_ref_high"] = [float(e * HARTREE2EV) for e in td_high.e]
        
    # 3. Reference TDA - Global Low
    td_low = live["mf_low"].TDA()
    td_low.nstates = nstates
    td_low.kernel()
    if td_low.e is not None:
        res["excitation_energies_ev_ref_low"] = [float(e * HARTREE2EV) for e in td_low.e]
        
    # 4. Reference TDA - Global LDA
    td_lda = live["mf_lda"].TDA()
    td_lda.nstates = nstates
    td_lda.kernel()
    if td_lda.e is not None:
        res["excitation_energies_ev_ref_lda"] = [float(e * HARTREE2EV) for e in td_lda.e]

    return res

# ---------------------------------------------------------------------------
# Population & density analysis (Task 2.6)
# ---------------------------------------------------------------------------
def run_population_block(config, live, outdir, name):
    """Mulliken charges (on the fragment atoms) + density-difference cubes for both guesses."""
    mol = live["mol"]
    emb_low = live["emb_high_in_low"]
    emb_lda = live["emb_high_in_low_lda"]
    mf_high = live["mf_high"]
    mf_low = live["mf_low"]
    mf_lda = live["mf_lda"]

    s_ao = mf_high.get_ovlp()
    nao = int(mol.nao_nr())

    # Densities
    dm_embed_ao_low = ea.full_ao_embedding_density(emb_low, ifrag=0)
    dm_embed_ao_lda = ea.full_ao_embedding_density(emb_lda, ifrag=0)
    dm_global_high_ao = mf_high.make_rdm1()
    dm_global_low_ao = mf_low.make_rdm1()
    dm_global_lda_ao = mf_lda.make_rdm1()
    
    ea.assert_full_ao(dm_embed_ao_low, nao, "embedding density low guess")
    ea.assert_full_ao(dm_global_high_ao, nao, "global-high density")

    frag_atoms = [int(a) for a in config["fragment_id"]]
    
    # Charges
    charges_embed_low = ea.mulliken_charges(mol, dm_embed_ao_low, s_ao, atom_ids=frag_atoms)
    charges_embed_lda = ea.mulliken_charges(mol, dm_embed_ao_lda, s_ao, atom_ids=frag_atoms)
    charges_high = ea.mulliken_charges(mol, dm_global_high_ao, s_ao, atom_ids=frag_atoms)
    charges_low = ea.mulliken_charges(mol, dm_global_low_ao, s_ao, atom_ids=frag_atoms)
    charges_lda = ea.mulliken_charges(mol, dm_global_lda_ao, s_ao, atom_ids=frag_atoms)

    # Cube files
    cube_path_low = os.path.join(outdir, f"density_diff_low_guess_{name}.cube")
    try:
        ea.density_difference_cube(mol, dm_embed_ao_low, dm_global_high_ao, cube_path_low)
        cube_written_low = cube_path_low
    except Exception as exc:
        cube_written_low = f"cube failed: {exc}"

    cube_path_lda = os.path.join(outdir, f"density_diff_lda_guess_{name}.cube")
    try:
        ea.density_difference_cube(mol, dm_embed_ao_lda, dm_global_high_ao, cube_path_lda)
        cube_written_lda = cube_path_lda
    except Exception as exc:
        cube_written_lda = f"cube failed: {exc}"

    return {
        "mulliken_embedding_low_guess": {str(k): v for k, v in charges_embed_low.items()},
        "mulliken_embedding_lda_guess": {str(k): v for k, v in charges_embed_lda.items()},
        "mulliken_global_high": {str(k): v for k, v in charges_high.items()},
        "mulliken_global_low": {str(k): v for k, v in charges_low.items()},
        "mulliken_global_lda": {str(k): v for k, v in charges_lda.items()},
        "mulliken_diff_low_guess": {str(k): charges_embed_low[k] - charges_high[k] for k in charges_embed_low},
        "mulliken_diff_lda_guess": {str(k): charges_embed_lda[k] - charges_high[k] for k in charges_embed_lda},
        "cube_file_low_guess": cube_written_low,
        "cube_file_lda_guess": cube_written_lda,
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
            # Remove del eigenvectors line here, managed cleanly in run_tda_block now
            tda_res.pop("eigenvectors", None) 
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