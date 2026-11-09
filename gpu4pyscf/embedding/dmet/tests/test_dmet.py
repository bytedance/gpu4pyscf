# Copyright 2026 The PySCF Developers. All Rights Reserved.
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
import cupy
import pyscf
from gpu4pyscf import scf
from gpu4pyscf.embedding.dmet import LODMET, AODMET

def setUpModule():
    global mol, mol1
    mol = pyscf.M(
    atom='''
H1 0 0 0
H2 0 0 1
H  0 0 2
H  0 0 3
H  0 0 4
H  0 0 5
''',
    basis = 'sto-3g',
    output = '/dev/null'
)

    mol1 = pyscf.M(
    atom='''
H1 0 0 0
H2 0 0 1
H  0 0 2
H  0 0 3
H  0 0 4
''',
    basis = 'sto-3g',
    spin = 1,
    output = '/dev/null'
)

def tearDownModule():
    global mol, mol1
    mol.stdout.close()
    mol1.stdout.close()
    del mol, mol1

class KnownValues(unittest.TestCase):
    def test_fragment(self):
        mf = scf.RHF(mol).density_fit()
        mf.conv_tol = 1e-12
        mf.kernel()
        
        dmet = LODMET(mf, imp=0, orth='lowdin')
        assert np.allclose(dmet.imp, [0])

        dmet = LODMET(mf, imp=1, orth='lowdin')
        assert np.allclose(dmet.imp, [1])

        dmet = LODMET(mf, imp='H1.*', orth='lowdin')
        assert np.allclose(dmet.imp, [0])

        dmet = LODMET(mf, imp='H2.*', orth='lowdin')
        assert np.allclose(dmet.imp, [1])

        dmet = LODMET(mf, imp=['H1.*', 'H2.*'], orth='lowdin')
        assert np.allclose(dmet.imp, [0, 1])

    def test_rhf_lodmet(self):
        mf = scf.RHF(mol).density_fit()
        mf.conv_tol = 1e-12
        mf.kernel()
        
        dmet = LODMET(mf, imp='H1.*', orth='lowdin')
        dmet.kernel()
        global_mf = dmet.make_global_mf()
        self.assertAlmostEqual(dmet.e_tot - mf.e_tot, 0, 12)
        self.assertAlmostEqual(global_mf.energy_tot() - mf.e_tot, 0, 12)

        dmet = LODMET(mf, imp='H1.*', orth='iplo')
        dmet.kernel()
        global_mf = dmet.make_global_mf()
        self.assertAlmostEqual(dmet.e_tot - mf.e_tot, 0, 12)
        self.assertAlmostEqual(global_mf.energy_tot() - mf.e_tot, 0, 12)

    def test_rohf_lodmet(self):
        mf = scf.ROHF(mol1).density_fit()
        mf.conv_tol = 1e-12
        mf.kernel()
        
        dmet = LODMET(mf, imp='H1.*', orth='lowdin')
        dmet.kernel()
        global_mf = dmet.make_global_mf()
        self.assertAlmostEqual(dmet.e_tot - mf.e_tot, 0, 12)
        self.assertAlmostEqual(global_mf.energy_tot() - mf.e_tot, 0, 12)

        dmet = LODMET(mf, imp='H1.*', orth='iplo')
        dmet.kernel()
        global_mf = dmet.make_global_mf()
        self.assertAlmostEqual(dmet.e_tot - mf.e_tot, 0, 12)
        self.assertAlmostEqual(global_mf.energy_tot() - mf.e_tot, 0, 12)

    def test_rhf_aodmet(self):
        mf = scf.RHF(mol).density_fit()
        mf.conv_tol = 1e-12
        mf.kernel()
        
        dmet = AODMET(mf, imp='H1.*', orth='lowdin')
        dmet.kernel()
        global_mf = dmet.make_global_mf()
        self.assertAlmostEqual(dmet.e_tot - mf.e_tot, 0, 12)
        self.assertAlmostEqual(global_mf.energy_tot() - mf.e_tot, 0, 12)

        dmet = AODMET(mf, imp='H1.*', orth='cholesky')
        dmet.kernel()
        global_mf = dmet.make_global_mf()
        self.assertAlmostEqual(dmet.e_tot - mf.e_tot, 0, 12)
        self.assertAlmostEqual(global_mf.energy_tot() - mf.e_tot, 0, 12)

    def test_rohf_aodmet(self):
        mf = scf.ROHF(mol1).density_fit()
        mf.conv_tol = 1e-12
        mf.kernel()
        
        dmet = AODMET(mf, imp='H1.*', orth='lowdin')
        dmet.kernel()
        global_mf = dmet.make_global_mf()
        self.assertAlmostEqual(dmet.e_tot - mf.e_tot, 0, 12)
        self.assertAlmostEqual(global_mf.energy_tot() - mf.e_tot, 0, 12)

        dmet = AODMET(mf, imp='H1.*', orth='cholesky')
        dmet.kernel()
        global_mf = dmet.make_global_mf()
        self.assertAlmostEqual(dmet.e_tot - mf.e_tot, 0, 12)
        self.assertAlmostEqual(global_mf.energy_tot() - mf.e_tot, 0, 12)

    def test_aodmet_orth(self):
        mf = scf.RHF(mol).density_fit()
        mf.conv_tol = 1e-12
        mf.kernel()
        
        dmet = AODMET(mf, imp='H1.*', orth='lowdin')
        dmet.kernel()
        es_mp2 = pyscf.mp.MP2(dmet.es_mf)
        es_mp2.kernel()
        e1 = dmet.e_core + es_mp2.e_corr

        dmet = AODMET(mf, imp='H1.*', orth='cholesky')
        dmet.kernel()
        es_mp2 = pyscf.mp.MP2(dmet.es_mf)
        es_mp2.kernel()
        e2 = dmet.e_core + es_mp2.e_corr
        self.assertAlmostEqual(e1 - e2, 0, 12)

    def test_global_mf(self):
        mf = scf.RHF(mol).density_fit()
        mf.conv_tol = 1e-12
        mf.kernel()
        s = mf.get_ovlp()
        identity = cupy.eye(6)
        
        dmet = LODMET(mf, imp='H1.*')
        dmet.kernel()
        global_mf = dmet.make_global_mf()
        mo = global_mf.mo_coeff
        assert np.abs(mo.conj().T@s@mo - identity).max() < 1e-8, "Global MOs are not orthogonal"

        dmet = AODMET(mf, imp='H1.*')
        dmet.kernel()
        global_mf = dmet.make_global_mf()
        mo = global_mf.mo_coeff
        assert np.abs(mo.conj().T@s@mo - identity).max() < 1e-8, "Global MOs are not orthogonal"

if __name__ == "__main__":
    print("Full Tests for DMET")
    unittest.main()