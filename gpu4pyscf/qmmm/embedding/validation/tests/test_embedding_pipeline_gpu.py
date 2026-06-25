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

"""GPU-backed unit tests for the embedding-validation pipeline.

These exercise the three properties the user explicitly asked us to pin down:

* limiting case  -- when Low == High (PBE-in-PBE) the functional-upgrade shift
  Delta E_xc_shift^act is *exactly* zero;
* full-dimension exactness -- with the QM region spanning every atom, the
  shifted reference energy equals the high-level energy evaluated
  non-self-consistently on the low-level converged density;
* dimension assertions -- Mulliken, the TDA transition density P_trans and the
  cube density are all built from full N_AO x N_AO matrices.

The whole module is skipped when no usable GPU / gpu4pyscf runtime is present.
"""

import os
import tempfile

import numpy as np
import pytest

from conftest import requires_gpu, GPU_AVAILABLE

import embedding_analysis as ea

if GPU_AVAILABLE:
    import cupy as cp
    from pyscf import gto
    from gpu4pyscf.dft import rks
    from gpu4pyscf.qmmm.embedding.embedding_dft import SingleFragmentEmbedding


# ---------------------------------------------------------------------------
# Small shared molecules
# ---------------------------------------------------------------------------
def _ethane():
    mol = gto.Mole()
    mol.atom = '''
        C      -0.76091    -0.00000     0.00000
        C       0.76091    -0.00000     0.00000
        H      -1.16001     1.02029     0.00000
        H      -1.16001    -0.51014    -0.88357
        H      -1.16001    -0.51014     0.88357
        H       1.16001    -1.02029     0.00000
        H       1.16001     0.51014     0.88357
        H       1.16001     0.51014    -0.88357
    '''
    mol.basis = '6-31g'
    mol.spin = 0
    mol.charge = 0
    mol.verbose = 0
    mol.build()
    return mol


def _water():
    mol = gto.Mole()
    mol.atom = '''
        O   0.000000   0.000000   0.117790
        H   0.000000   0.755453  -0.471161
        H   0.000000  -0.755453  -0.471161
    '''
    mol.basis = 'sto-3g'
    mol.spin = 0
    mol.charge = 0
    mol.verbose = 0
    mol.build()
    return mol


# ---------------------------------------------------------------------------
# 1. Limiting case: shift == 0 when Low == High
# ---------------------------------------------------------------------------
@requires_gpu
class TestShiftLimitingCase:
    def test_shift_zero_pbe_in_pbe(self):
        mol = _ethane()
        mf_low = rks.RKS(mol, xc='PBE'); mf_low.conv_tol = 1e-11
        mf_high = rks.RKS(mol, xc='PBE'); mf_high.conv_tol = 1e-11

        mf_low.kernel()
        emb = SingleFragmentEmbedding(mf_low, mf_high, [0, 2, 3, 4])
        emb.kernel()

        shift = ea.shifted_reference_energy(mf_low, mf_high, emb)
        assert abs(shift["delta_xc_shift"]) < 1e-9, (
            f"PBE-in-PBE shift must vanish, got {shift['delta_xc_shift']}")

    def test_delta_shift_helper_zero_same_functional(self):
        # Directly exercise the shift helper with identical low/high functionals.
        mol = _water()
        mf = rks.RKS(mol, xc='PBE'); mf.conv_tol = 1e-11
        mf.kernel()
        dm = mf.make_rdm1()
        delta = ea.delta_exc_shift_active(mf, mf, dm)
        assert abs(delta) < 1e-10

    def test_shift_zero_b3lyp_in_b3lyp(self):
        mol = _water()
        mf_low = rks.RKS(mol, xc='B3LYP'); mf_low.conv_tol = 1e-11
        mf_high = rks.RKS(mol, xc='B3LYP'); mf_high.conv_tol = 1e-11
        mf_low.kernel()
        emb = SingleFragmentEmbedding(mf_low, mf_high, [0, 1, 2])
        emb.kernel()
        shift = ea.shifted_reference_energy(mf_low, mf_high, emb)
        # Same hybrid on both sides -> both semilocal and exchange diffs vanish.
        assert abs(shift["delta_xc_shift"]) < 1e-9


# ---------------------------------------------------------------------------
# 2. Full-dimension exactness of the shifted reference energy
# ---------------------------------------------------------------------------
@requires_gpu
class TestFullDimensionExactness:
    def test_shifted_ref_equals_nonscf_high(self):
        """QM region = all atoms => E_ref^shifted == E_high[D_low^conv]."""
        mol = _water()
        mf_low = rks.RKS(mol, xc='PBE'); mf_low.conv_tol = 1e-11
        mf_high = rks.RKS(mol, xc='B3LYP'); mf_high.conv_tol = 1e-11
        mf_low.kernel()

        all_atoms = list(range(mol.natm))
        emb = SingleFragmentEmbedding(mf_low, mf_high, all_atoms)
        emb.kernel()

        shift = ea.shifted_reference_energy(mf_low, mf_high, emb)
        e_nonscf_high = ea.high_level_nonscf_energy(mf_low, mf_high)

        assert abs(shift["e_ref_shifted"] - e_nonscf_high) < 1e-7, (
            f"Full-dim shifted ref {shift['e_ref_shifted']} != "
            f"non-SCF high {e_nonscf_high}")

    def test_full_dim_projection_recovers_density(self):
        """With a full-dimension B, B (B^T S D S B) B^T must reproduce D."""
        mol = _water()
        mf_low = rks.RKS(mol, xc='PBE'); mf_low.conv_tol = 1e-11
        mf_high = rks.RKS(mol, xc='B3LYP'); mf_high.conv_tol = 1e-11
        mf_low.kernel()
        emb = SingleFragmentEmbedding(mf_low, mf_high, list(range(mol.natm)))
        emb.kernel()

        s_ao = mf_low.get_ovlp()
        B = emb.B[0]
        dm_low = mf_low.make_rdm1()
        dm_act = ea.project_dm_ao_to_emb(dm_low, B, s_ao)
        dm_back = ea.project_dm_emb_to_ao(dm_act, B)
        err = float(cp.abs(cp.asarray(dm_back) - cp.asarray(dm_low)).max())
        assert err < 1e-7, f"Full-dim projection round-trip error {err}"


# ---------------------------------------------------------------------------
# 3. Dimension assertions for global-property evaluation
# ---------------------------------------------------------------------------
@requires_gpu
class TestDimensionAssertions:
    @classmethod
    def setup_class(cls):
        cls.mol = _ethane()
        cls.mf_low = rks.RKS(cls.mol, xc='PBE'); cls.mf_low.conv_tol = 1e-10
        cls.mf_high = rks.RKS(cls.mol, xc='B3LYP'); cls.mf_high.conv_tol = 1e-10
        cls.mf_low.kernel()
        cls.mf_high.kernel()
        cls.emb = SingleFragmentEmbedding(cls.mf_low, cls.mf_high, [0, 2, 3, 4])
        cls.emb.kernel()
        cls.nao = int(cls.mol.nao_nr())

    def test_embedding_density_is_full_ao(self):
        dm = ea.full_ao_embedding_density(self.emb, ifrag=0)
        assert dm.shape == (self.nao, self.nao)

    def test_mulliken_requires_full_ao(self):
        s_ao = self.mf_high.get_ovlp()
        dm_ao = ea.full_ao_embedding_density(self.emb, ifrag=0)
        # Correct AO-sized density works.
        charges = ea.mulliken_charges(self.mol, dm_ao, s_ao, atom_ids=[0])
        assert 0 in charges
        # Passing the raw embedding-basis density must trip the guard.
        dm_emb = self.emb.mf_inner[0].make_rdm1()
        assert dm_emb.shape[0] < self.nao
        with pytest.raises(AssertionError):
            ea.mulliken_charges(self.mol, dm_emb, s_ao, atom_ids=[0])

    def test_ptrans_is_full_ao(self):
        # Project cluster MOs to AO and assemble the TDA A matrix; the internal
        # P_trans block is asserted to be (nocc, nvir, N_AO, N_AO).
        B = self.emb.B[0]
        mf_inner = self.emb.mf_inner[0]
        mo_ao = ea.project_mo_emb_to_ao(mf_inner.mo_coeff, B)
        assert mo_ao.shape[0] == self.nao
        res = ea.build_tda_amatrix(self.mf_low, mo_ao, mf_inner.mo_energy,
                                   mf_inner.mo_occ, singlet=True)
        # A is (nocc*nvir) square and excitation energies are real & positive-ish.
        n = res["nocc"] * res["nvir"]
        assert res["a_matrix"].shape == (n, n)
        assert len(res["excitation_energies"]) == n

    def test_ptrans_rejects_emb_basis_mo(self):
        # Feeding embedding-basis MO coefficients must raise (leading dim != N_AO).
        mf_inner = self.emb.mf_inner[0]
        mo_emb = mf_inner.mo_coeff
        if mo_emb.shape[0] != self.nao:
            with pytest.raises(AssertionError):
                ea.build_tda_amatrix(self.mf_low, mo_emb, mf_inner.mo_energy,
                                     mf_inner.mo_occ, singlet=True)

    def test_cube_requires_full_ao(self):
        dm_embed_ao = ea.full_ao_embedding_density(self.emb, ifrag=0)
        dm_high_ao = self.mf_high.make_rdm1()
        outdir = tempfile.mkdtemp()
        cube = os.path.join(outdir, "diff.cube")
        ea.density_difference_cube(self.mol, dm_embed_ao, dm_high_ao, cube,
                                   nx=20, ny=20, nz=20)
        assert os.path.exists(cube)
        # Embedding-basis density must be rejected.
        dm_emb = self.emb.mf_inner[0].make_rdm1()
        with pytest.raises(AssertionError):
            ea.density_difference_cube(self.mol, dm_emb, dm_high_ao,
                                       os.path.join(outdir, "bad.cube"))


# ---------------------------------------------------------------------------
# 4. End-to-end smoke test of the single-job driver
# ---------------------------------------------------------------------------
@requires_gpu
class TestSingleJobSmoke:
    def test_process_single_water(self):
        import run_single_job as single
        config = {
            "element": ["O", "H", "H"],
            "structure": [[0.0, 0.0, 0.117790],
                          [0.0, 0.755453, -0.471161],
                          [0.0, -0.755453, -0.471161]],
            "charge": 0, "spin": 0, "basis_set": "sto-3g",
            "xc_lda": "lda,vwn", "xc_low": "pbe", "xc_high": "b3lyp",
            "fragment_id": [0, 1, 2],
            "energy_flag": True, "bond_approx_flag": False,
            "tda_flag": True, "population_flag": True,
            "bond_test_id": [0, 1],
        }
        outdir = tempfile.mkdtemp()
        result = single.process_single(config, "water_smoke", outdir=outdir)
        assert result["status"] == "ok"
        energies = result["blocks"]["energies"]["energies"]
        # Sanity: all headline energies present and finite.
        for key in ("global_low_pbe", "global_high_b3lyp",
                    "embed_b3lyp_in_pbe", "shifted_reference_b3lyp_in_pbe"):
            assert np.isfinite(energies[key])
        assert os.path.exists(result["_output_path"])


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
