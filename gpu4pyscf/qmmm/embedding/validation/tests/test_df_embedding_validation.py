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

import os
import unittest
import numpy as np
import cupy as cp
from pyscf import gto
from gpu4pyscf.dft import rks
from gpu4pyscf.qmmm.embedding.embedding_dft_try1 import SingleFragmentEmbedding
from gpu4pyscf.qmmm.embedding.validation.embedding_analysis import (
    shifted_reference_energy, 
    high_level_nonscf_energy, 
    core_homo_lumo,
    embedded_tda,
    full_ao_embedding_density,
    mulliken_charges,
    density_difference_cube
)


class TestEmbeddingDFT(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Build the water molecule system
        cls.mol = gto.Mole()
        cls.mol.atom = '''
        O   0.000000  0.000000  0.119720
        H   0.000000  0.761561 -0.478879
        H   0.000000 -0.761561 -0.478879
        '''
        cls.mol.basis = 'def2-svp'
        cls.mol.verbose = 0
        cls.mol.build()

        cls.qm_fragment = [0, 1, 2]

        # Run global low-level (Environment)
        cls.mf_low = rks.RKS(cls.mol, xc='pbe').density_fit()
        cls.mf_low.conv_tol = 1e-12
        cls.mf_low.kernel()

        # Run global high-level (Reference)
        cls.mf_high = rks.RKS(cls.mol, xc='b3lyp').density_fit()
        cls.mf_high.conv_tol = 1e-12
        cls.mf_high.kernel()

    @classmethod
    def tearDownClass(cls):
        del cls.mol
        del cls.mf_low
        del cls.mf_high

    def test_pbe_in_pbe_exactness(self):
        mf_inner_template = rks.RKS(self.mol, xc='pbe').density_fit()
        mf_inner_template.conv_tol = 1e-12
        
        emb_obj = SingleFragmentEmbedding(self.mf_low, mf_inner_template, self.qm_fragment, threshold=1e-5, verbose=0)
        emb_obj.kernel()

        # 1. Energy Comparison
        self.assertAlmostEqual(emb_obj.e_tot, self.mf_low.e_tot, places=8, 
                               msg="PBE-in-PBE total energy differs from global PBE energy")

        # 2. Shifted Reference Energy Validation
        res_energy = shifted_reference_energy(self.mf_low, self.mf_low, emb_obj, ifrag=0)
        self.assertAlmostEqual(res_energy['delta_xc_shift'], 0.0, places=8, 
                               msg="Delta XC shift for PBE-in-PBE must be exactly 0")
        self.assertAlmostEqual(res_energy['e_ref_shifted'], self.mf_low.e_tot, places=8, 
                               msg="Shifted ref energy for PBE-in-PBE should exactly match low-level energy")

    def test_b3lyp_in_pbe_properties(self):
        mf_inner_template = rks.RKS(self.mol, xc=self.mf_high.xc).density_fit()
        mf_inner_template.conv_tol = 1e-12
        
        emb_obj = SingleFragmentEmbedding(self.mf_low, mf_inner_template, self.qm_fragment, threshold=1e-5, verbose=0)
        emb_obj.kernel()

        # 1. Total Energy (Full dimension QM = Global QM)
        self.assertAlmostEqual(emb_obj.e_tot, self.mf_high.e_tot, places=7, 
                               msg="B3LYP-in-PBE (full system) energy differs from global B3LYP energy")

        # 2. Shifted Reference Energy
        res_energy = shifted_reference_energy(self.mf_low, self.mf_high, emb_obj, ifrag=0)
        nonscf_e = high_level_nonscf_energy(self.mf_low, self.mf_high)
        self.assertAlmostEqual(res_energy['e_ref_shifted'], nonscf_e, places=8, 
                               msg="Shifted reference energy diverges from non-SCF high-level energy")

        # 3. Orbital Energies (HOMO/LUMO)
        res_orb = core_homo_lumo(emb_obj.mf_inner[0])
        mo_occ = cp.asnumpy(self.mf_high.mo_occ)
        mo_energy = cp.asnumpy(self.mf_high.mo_energy)
        occ_idx = np.where(mo_occ > 1e-8)[0]
        vir_idx = np.where(mo_occ <= 1e-8)[0]
        
        ref_homo = float(mo_energy[occ_idx].max())
        ref_lumo = float(mo_energy[vir_idx].min())

        self.assertAlmostEqual(res_orb['homo'], ref_homo, places=7, msg="HOMO energy mismatch")
        self.assertAlmostEqual(res_orb['lumo'], ref_lumo, places=7, msg="LUMO energy mismatch")

        # 4. Local TDA Excitation
        res_tda = embedded_tda(emb_obj, self.mf_high, ifrag=0, singlet=True, nstates=3)
        td = self.mf_high.TDA()
        td.verbose = 0
        ref_tda_energies = td.kernel()[0] * 27.211386245988 # Convert au to eV
        
        for i in range(min(3, len(ref_tda_energies))):
            self.assertAlmostEqual(res_tda['excitation_energies_ev'][i], ref_tda_energies[i], places=4, 
                                   msg=f"TDA State {i+1} excitation energy mismatch")

        # 5. Population Analysis (Mulliken)
        dm_emb_ao = full_ao_embedding_density(emb_obj, ifrag=0)
        s_ao = self.mf_low.get_ovlp()
        emb_charges = mulliken_charges(self.mol, dm_emb_ao, s_ao, atom_ids=self.qm_fragment)
        
        ref_dm = cp.asarray(self.mf_high.make_rdm1())
        ref_charges = mulliken_charges(self.mol, ref_dm, s_ao, atom_ids=self.qm_fragment)
        
        for atom_idx in self.qm_fragment:
            self.assertAlmostEqual(emb_charges[atom_idx], ref_charges[atom_idx], places=5, 
                                   msg=f"Mulliken charge mismatch on atom {atom_idx}")

        # 6. Density Difference Cube Generation
        outfile = "test_water_dimer_diff.cube"
        density_difference_cube(self.mol, dm_emb_ao, ref_dm, outfile, nx=20, ny=20, nz=20)
        self.assertTrue(os.path.exists(outfile), msg="Cube file was not generated successfully")
        
        # Clean up the generated file
        if os.path.exists(outfile):
            os.remove(outfile)


if __name__ == '__main__':
    print("Full tests for validation embedding results...")
    unittest.main()