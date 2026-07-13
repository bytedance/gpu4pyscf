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

import unittest
import numpy as np
import cupy as cp
from pyscf import gto
from gpu4pyscf.dft import rks
from gpu4pyscf.qmmm.embedding.embedding_dmet import _as_cupy
from gpu4pyscf.qmmm.embedding.embedding_dmet_dft_try1 import SingleFragmentEmbedding


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.Mole()
        cls.mol.atom = '''
            C      -0.76091    -0.00000     0.00000
            C       0.76091    -0.00000     0.00000
            H      -1.16001     1.02029     0.00000
            H      -1.16001    -0.51014    -0.88357
            H      -1.16001    -0.51014     0.88357
            H       1.16001    -1.02029     0.00000
            H       1.16001     0.51014     0.88357
            H       1.16001     0.51014    -0.88357
        '''
        cls.mol.basis = '6-31g'
        cls.mol.spin = 0
        cls.mol.charge = 0
        cls.mol.verbose = 0
        cls.mol.build()

        cls.methyl_fragment = [0, 2, 3, 4]

    @classmethod
    def tearDownClass(cls):
        del cls.mol

    def test_b3lyp_in_b3lyp(self):
        mf_outer = rks.RKS(self.mol, xc='B3LYP')
        mf_inner_template = rks.RKS(self.mol, xc='B3LYP')
        mf_outer.conv_tol = 1e-10
        mf_inner_template.conv_tol = 1e-10

        emb_obj = SingleFragmentEmbedding(mf_outer, mf_inner_template, self.methyl_fragment, threshold=1e-3)
        emb_obj.kernel()

        mf_outer.mo_coeff = None
        e_ref = mf_outer.kernel()

        self.assertTrue(np.abs(e_ref - emb_obj.e_tot) < 1e-4,
                        f"Reference energy {e_ref} != Embedding energy {emb_obj.e_tot}, diff={np.abs(e_ref - emb_obj.e_tot)}")

    def test_b3lyp_in_pbe_full_region(self):
        mf_outer = rks.RKS(self.mol, xc='PBE')
        mf_inner_template = rks.RKS(self.mol, xc='B3LYP')
        mf_outer.conv_tol = 1e-10
        mf_inner_template.conv_tol = 1e-10

        all_atoms = [i for i in range(8)]
        emb_obj = SingleFragmentEmbedding(mf_outer, mf_inner_template, all_atoms, threshold=1e-2)
        emb_obj.kernel()

        mf_inner_template.mo_coeff = None
        e_ref = mf_inner_template.kernel()

        self.assertTrue(np.abs(e_ref - emb_obj.e_tot) < 1e-4,
                        f"Reference energy {e_ref} != Embedding energy {emb_obj.e_tot}, diff={np.abs(e_ref - emb_obj.e_tot)}")

    def test_algebraic_properties(self):
        mf_outer = rks.RKS(self.mol, xc='PBE')
        mf_inner = rks.RKS(self.mol, xc='PBE')
        mf_outer.conv_tol = 1e-10
        mf_inner.conv_tol = 1e-10

        emb_obj = SingleFragmentEmbedding(mf_outer, mf_inner, [0, 1, 2], threshold=1e-2)
        emb_obj.kernel()

        S_ao = cp.asarray(mf_outer.get_ovlp())
        B = emb_obj.B[0]
        D_core = emb_obj.dm_core[0]

        # In non-orthogonal DMET, B^T S B is S_emb, NOT the identity matrix.
        ortho_check = B.T @ S_ao @ B
        S_AA = S_ao[cp.ix_(emb_obj.frag_idx[0], emb_obj.frag_idx[0])]
        n_frag = S_AA.shape[0]
        max_ortho_err = float(cp.abs(ortho_check[:n_frag, :n_frag] - S_AA).max())
        self.assertTrue(max_ortho_err < 1e-6,
                        f"Basis B is not orthogonal, max error: {max_ortho_err}")

        # Check Spatial Isolation (Core DM projected onto the active space must be small)
        core_overlap = B.T @ S_ao @ D_core @ S_ao @ B
        max_overlap_err = float(cp.abs(core_overlap).max())
        self.assertTrue(max_overlap_err < 1e-4,
                        f"Core DM leaks into Active Space, max error: {max_overlap_err}")

    def test_electron_conservation(self):
        mf_outer = rks.RKS(self.mol, xc='PBE')
        mf_inner = rks.RKS(self.mol, xc='B3LYP')
        mf_outer.conv_tol = 1e-10
        mf_inner.conv_tol = 1e-10

        emb_obj = SingleFragmentEmbedding(mf_outer, mf_inner, [0, 1], threshold=1e-2)
        emb_obj.kernel()

        S_ao = cp.asarray(mf_outer.get_ovlp())
        D_emb_high = cp.asarray(emb_obj.mf_inner[0].make_rdm1())
        D_core = emb_obj.dm_core[0]
        B = emb_obj.B[0]

        # Project local active density back to full AO basis
        D_emb_ao = B @ D_emb_high @ B.T
        D_total_ao = D_core + D_emb_ao

        n_elec_calc = float(cp.trace(D_total_ao @ S_ao))
        n_elec_exact = float(self.mol.nelectron)

        self.assertAlmostEqual(n_elec_calc, n_elec_exact, places=5,
                               msg=f"Electron loss: {n_elec_calc} != {n_elec_exact}")

    def test_template_isolation_and_convergence(self):
        mf_outer = rks.RKS(self.mol, xc='PBE')
        mf_inner_template = rks.RKS(self.mol, xc='B3LYP')
        mf_outer.conv_tol = 1e-10
        mf_inner_template.conv_tol = 1e-10

        emb_obj = SingleFragmentEmbedding(mf_outer, mf_inner_template,
                                          self.methyl_fragment, threshold=1e-2)
        emb_obj.kernel()

        mf_inner_template.mo_coeff = None
        mf_inner_template.kernel()

        self.assertTrue(mf_inner_template.converged,
                        "Template object was poisoned and failed to converge!")

    def test_returns_energy(self):
        mf_outer = rks.RKS(self.mol, xc='PBE')
        mf_inner = rks.RKS(self.mol, xc='PBE')
        mf_outer.conv_tol = 1e-10
        mf_inner.conv_tol = 1e-10

        emb_obj = SingleFragmentEmbedding(mf_outer, mf_inner, self.methyl_fragment, threshold=1e-2)
        try:
            emb_obj.kernel()
        except AttributeError as e:
            self.fail(f"Embedding failed due to missing vk attribute or similar: {e}")

        self.assertTrue(emb_obj.e_tot is not None, "DFT embedding failed to return an energy.")


if __name__ == '__main__':
    print("Full Tests for DMET-DFT try1 (density matrix diagonalization)")
    unittest.main()
