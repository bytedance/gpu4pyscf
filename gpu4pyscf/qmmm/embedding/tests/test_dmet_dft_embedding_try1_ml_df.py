# Copyright 2021-2025 The PySCF Developers. All Rights Reserved.
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

from gpu4pyscf.qmmm.embedding.embedding_dmet_dft_try1 import SingleFragmentEmbedding
from gpu4pyscf.qmmm.embedding.embedding_dmet_dft_try1_ml import OneStepRKS, SingleFragmentEmbedding_ML
from gpu4pyscf.qmmm.embedding.embedding_dmet import _as_cupy


def dummy_eval_density_func(mol, xc, grids):
    """
    Dummy ML density evaluator.
    It performs a highly converged standard RKS calculation to provide an 'exact'
    reference density, ensuring that the ML framework mathematically reduces to
    the exact global evaluation when the 'ML prediction' is perfect.
    """
    mf = rks.RKS(mol).density_fit()
    mf.xc = xc
    mf.grids = grids
    mf.verbose = 0
    mf.conv_tol = 1.0E-12
    mf.kernel()

    dm = cp.asarray(mf.make_rdm1())

    # Calculate exact J and K matrices
    vj, vk = mf.get_jk(mol, dm)
    e_j = 0.5 * float(cp.sum(dm * vj))

    is_hybrid = mf._numint.libxc.is_hybrid_xc(xc)
    if is_hybrid:
        hyb = mf._numint.libxc.hybrid_coeff(xc, spin=mol.spin)
        vk = vk * hyb
        e_k = 0.5 * float(cp.sum(dm * vk))
    else:
        vk = None
        e_k = 0.0

    # Calculate exact Vxc and Exc
    _, e_xc, vxc = mf._numint.nr_rks(mol, grids, xc, dm)
    int_rho_vxc = float(cp.sum(dm * vxc))

    return vj, vk, vxc, e_j, e_k, float(e_xc), int_rho_vxc


class TestDMETMLEmbeddingCAS(unittest.TestCase):
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
        cls.full_fragment = [i for i in range(cls.mol.natm)]

    @classmethod
    def tearDownClass(cls):
        del cls.mol

    def test_harris_rks_exactness(self):
        mf_ref = rks.RKS(self.mol, xc='PBE').density_fit()
        mf_ref.verbose = 0
        mf_ref.conv_tol = 1.0E-12
        e_ref = mf_ref.kernel()

        mf_harris = OneStepRKS(self.mol, dummy_eval_density_func, xc='PBE').density_fit()
        mf_harris.verbose = 0
        e_harris = mf_harris.kernel()

        self.assertAlmostEqual(e_ref, e_harris, places=6,
                               msg=f"OneStepRKS energy {e_harris} differs from exact RKS {e_ref}")

    def test_full_system_pbe_in_pbe(self):
        mf_outer = OneStepRKS(self.mol, dummy_eval_density_func, xc='PBE').density_fit()
        mf_inner = rks.RKS(self.mol, xc='PBE').density_fit()
        mf_inner.conv_tol = 1.0E-10
        mf_outer.conv_tol = 1.0E-10

        emb_obj = SingleFragmentEmbedding_ML(mf_outer, mf_inner, self.full_fragment,
                                              threshold=1e-2, verbose=0)
        emb_obj.kernel()

        mf_outer.mo_coeff = None
        e_global = mf_outer.kernel()
        e_emb = emb_obj.e_tot

        self.assertAlmostEqual(e_global, e_emb, places=4,
                               msg="Full-system PBE-in-PBE failed exact cancellation under CAS-like framework.")

    def test_equivalence_to_standard_embedding(self):
        # Standard CAS-like embedding (Iterative SCF outer)
        mf_outer_std = rks.RKS(self.mol, xc='PBE').density_fit()
        mf_outer_std.conv_tol = 1.0E-10
        mf_inner_std = rks.RKS(self.mol, xc='B3LYP').density_fit()
        mf_inner_std.conv_tol = 1.0E-10
        emb_std = SingleFragmentEmbedding(mf_outer_std, mf_inner_std, self.methyl_fragment,
                                           threshold=1e-2, verbose=0)
        e_std = emb_std.kernel()

        # ML-driven CAS-like embedding (Harris 1-step outer)
        mf_outer_ml = OneStepRKS(self.mol, dummy_eval_density_func, xc='PBE').density_fit()
        mf_outer_ml.conv_tol = 1.0E-10
        mf_inner_ml = rks.RKS(self.mol, xc='B3LYP').density_fit()
        mf_inner_ml.conv_tol = 1.0E-10
        emb_ml = SingleFragmentEmbedding_ML(mf_outer_ml, mf_inner_ml, self.methyl_fragment,
                                             threshold=1e-2, verbose=0)
        e_ml = emb_ml.kernel()

        # They should match since dummy_eval_density_func provides the exact same converged density
        self.assertAlmostEqual(e_std, e_ml, places=4,
                               msg=f"ML CAS-like Embedding {e_ml} diverged from Standard CAS-like Embedding {e_std}!")

    def test_one_step_rks_max_cycle_override(self):
        mf_harris = OneStepRKS(self.mol, dummy_eval_density_func, xc='PBE').density_fit()
        mf_harris.max_cycle = 100
        mf_harris.verbose = 0

        mf_harris.kernel()

        self.assertEqual(mf_harris.max_cycle, 1,
                         "OneStepRKS failed to override malicious max_cycle setting!")

    def test_returns_valid_energy(self):
        mf_outer = rks.RKS(self.mol, xc='PBE').density_fit()
        mf_inner = rks.RKS(self.mol, xc='B3LYP').density_fit()
        mf_outer.conv_tol = 1e-10
        mf_inner.conv_tol = 1e-10

        emb_obj = SingleFragmentEmbedding_ML(mf_outer, mf_inner, self.methyl_fragment,
                                              threshold=1e-2, verbose=0)
        emb_obj.kernel()
        self.assertTrue(emb_obj.e_tot is not None, "ML embedding returned None energy.")


if __name__ == '__main__':
    print("Full Tests for DMET-DFT-ML (density matrix diagonalization)")
    unittest.main()
