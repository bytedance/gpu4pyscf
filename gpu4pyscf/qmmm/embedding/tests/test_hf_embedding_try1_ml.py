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
from gpu4pyscf.scf import hf as gpu_hf
from gpu4pyscf.qmmm.embedding.embedding_hf import SingleFragmentEmbedding
from gpu4pyscf.qmmm.embedding.embedding_hf_try1_ml import OneStepRHF, SingleFragmentEmbedding_ML


def dummy_eval_density_func(mol, xc, grids):
    mf = gpu_hf.RHF(mol)
    mf.verbose = 0
    mf.conv_tol = 1.0E-12
    mf.kernel()
    
    dm = cp.asarray(mf.make_rdm1())
    
    # Calculate exact J and K matrices
    vj, vk = mf.get_jk(mol, dm)
    e_j = 0.5 * float(cp.sum(dm * vj))
    e_k = 0.25 * float(cp.sum(dm * vk))
        
    # For HF, there is no Exchange-Correlation potential/energy from DFT
    vxc = cp.zeros_like(vj)
    e_xc = 0.0
    int_rho_vxc = 0.0
    
    return vj, vk, vxc, e_j, e_k, float(e_xc), int_rho_vxc


class TestMLEmbedding(unittest.TestCase):
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

    def test_onestep_rhf_exactness(self):
        mf_ref = gpu_hf.RHF(self.mol)
        mf_ref.verbose = 0
        e_ref = mf_ref.kernel()

        mf_onestep = OneStepRHF(self.mol, dummy_eval_density_func)
        mf_onestep.verbose = 0
        e_onestep = mf_onestep.kernel()

        self.assertAlmostEqual(e_ref, e_onestep, places=8, 
                               msg=f"OneStepRHF energy {e_onestep} differs from exact RHF {e_ref}")

    def test_full_system_hf_in_hf(self):
        mf_outer = OneStepRHF(self.mol, dummy_eval_density_func)
        mf_inner = gpu_hf.RHF(self.mol)
        mf_inner.conv_tol = 1.0E-12
        
        emb_obj = SingleFragmentEmbedding_ML(mf_outer, mf_inner, self.full_fragment, verbose=0)
        emb_obj.kernel()
        
        mf_outer.kernel()
        e_global = mf_outer.e_tot
        e_emb = emb_obj.e_tot
        
        self.assertAlmostEqual(e_global, e_emb, places=8, 
                               msg="Full-system HF-in-HF failed exact cancellation.")

    def test_equivalence_to_standard_embedding(self):

        mf_outer_std = gpu_hf.RHF(self.mol)
        mf_outer_std.conv_tol = 1.0E-12
        mf_inner_std = gpu_hf.RHF(self.mol)
        mf_inner_std.conv_tol = 1.0E-12
        emb_std = SingleFragmentEmbedding(mf_outer_std, mf_inner_std, self.methyl_fragment, verbose=0)
        e_std = emb_std.kernel()

        mf_outer_ml = OneStepRHF(self.mol, dummy_eval_density_func)
        mf_inner_ml = gpu_hf.RHF(self.mol)
        mf_inner_ml.conv_tol = 1.0E-12
        emb_ml = SingleFragmentEmbedding_ML(mf_outer_ml, mf_inner_ml, self.methyl_fragment, verbose=0)
        e_ml = emb_ml.kernel()

        self.assertAlmostEqual(e_std, e_ml, places=8, 
                               msg=f"ML Embedding {e_ml} diverged from Standard Embedding {e_std}!")

    def test_onestep_max_cycle_override(self):

        mf_onestep = OneStepRHF(self.mol, dummy_eval_density_func)
        mf_onestep.max_cycle = 100 
        mf_onestep.verbose = 0
        
        mf_onestep.kernel()
        
        self.assertEqual(mf_onestep.max_cycle, 1, 
                         "OneStepRHF failed to override malicious max_cycle setting!")

if __name__ == '__main__':
    print("Full Tests for ML-Driven ONIOM-like Embedding (HF Version)...")
    unittest.main()