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
from pyscf.pbc import gto as pbcgto
from gpu4pyscf.pbc import scf as pbcscf
from gpu4pyscf.embedding.dmet.pbc import LODMET, AODMET

def setUpModule():
    global cell, cell1
    cell = pbcgto.Cell(
    atom='''
H1 0 0 0
H2 0 0 1
H  0 0 2
H  0 0 3
H  0 0 4
H  0 0 5
''',
    a = ((5,0,0),(0,5,0),(0,0,6)),
    basis = 'sto-3g',
    output = '/dev/null'
)

    cell1 = pbcgto.Cell(
    atom='''
H1 0 0 0
H2 0 0 1
H  0 0 2
H  0 0 3
H  0 0 4
''',
    a = ((5,0,0),(0,5,0),(0,0,6)),
    basis = 'sto-3g',
    spin = 1,
    output = '/dev/null'
)

def tearDownModule():
    global cell, cell1
    cell.stdout.close()
    cell1.stdout.close()
    del cell, cell1

class KnownValues(unittest.TestCase):
    def test_fragment(self):
        mf = pbcscf.RHF(cell, exxdiv=None).density_fit()
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
        mf = pbcscf.RHF(cell, exxdiv=None).density_fit()
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
        mf = pbcscf.ROHF(cell1, exxdiv=None).density_fit()
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
        mf = pbcscf.RHF(cell, exxdiv=None).density_fit()
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
        mf = pbcscf.ROHF(cell1, exxdiv=None).density_fit()
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
        mf = pbcscf.RHF(cell, exxdiv=None).density_fit()
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
        mf = pbcscf.RHF(cell, exxdiv=None).density_fit()
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
    print("Full Tests for Gamma-point DMET")
    unittest.main()