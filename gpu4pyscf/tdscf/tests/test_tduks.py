#!/usr/bin/env python
#
# Copyright 2024 The PySCF Developers. All Rights Reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import unittest
import numpy as np
import cupy as cp
from pyscf import lib, gto
from gpu4pyscf import tdscf

class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mol = gto.Mole()
        mol.verbose = 5
        mol.output = '/dev/null'
        mol.atom = '''
        O     0.   0.       0.
        H     0.   -0.757   0.587
        H     0.   0.757    0.587'''
        mol.spin = 2
        mol.basis = '631g'
        cls.mol = mol.build()

        mol1 = gto.Mole()
        mol1.verbose = 0
        mol1.atom = '''
        O     0.   0.       0.
        H     0.   -0.757   0.587
        H     0.   0.757    0.587'''
        mol1.basis = '631g'
        cls.mol1 = mol1.build()

        cls.mf_uhf = mf_uhf = mol.UHF().to_gpu().run()
        cls.td_hf = mf_uhf.TDHF().run(conv_tol=1e-6)

        mf_lda = mol.UKS().set(xc='lda', conv_tol=1e-12).to_gpu()
        mf_lda.grids.prune = None
        cls.mf_lda = mf_lda.run()

        mf_bp86 = mol.UKS().set(xc='b88,p86', conv_tol=1e-12).to_gpu()
        mf_bp86.grids.prune = None
        cls.mf_bp86 = mf_bp86.run()

        mf_b3lyp = mol.UKS().set(xc='b3lyp5', conv_tol=1e-12).to_gpu()
        mf_b3lyp.grids.prune = None
        cls.mf_b3lyp = mf_b3lyp.density_fit().run()

        cls.mf_m06l = mol.UKS().to_gpu().run(xc='m06l')

    @classmethod
    def tearDownClass(cls):
        cls.mol.stdout.close()

    def test_nohybrid_lda(self):
        mf_lda = self.mf_lda
        td = mf_lda.CasidaTDDFT()
        assert td.device == 'gpu'
        es = td.kernel(nstates=4)[0]
        e_ref = td.to_cpu().kernel(nstates=4)[0]
        self.assertAlmostEqual(abs(es[:3]-e_ref[:3]).max(), 0, 8)
        self.assertAlmostEqual(lib.fp(es[:3]), 0.04757678947994772, 6)

        mol1 = self.mol1
        mf = mol1.UKS().run(xc='lda, vwn_rpa').run()
        td = mf.CasidaTDDFT().to_gpu()
        assert td.device == 'gpu'
        td.nstates = 5
        es = td.kernel()[0]
        ref = td.to_cpu().kernel()[0]
        self.assertAlmostEqual(abs(es - ref).max(), 0, 8)

    def test_nohybrid_b88p86(self):
        mf_bp86 = self.mf_bp86
        td = mf_bp86.CasidaTDDFT()
        assert td.device == 'gpu'
        es = td.kernel(nstates=4)[0]
        e_ref = td.to_cpu().kernel(nstates=4)[0]
        self.assertAlmostEqual(abs(es[:3]-e_ref[:3]).max(), 0, 8)
        self.assertAlmostEqual(lib.fp(es[:3]), 0.05374466608355896, 6)

    def test_tddft_lda(self):
        mf_lda = self.mf_lda
        td = mf_lda.TDDFT()
        assert td.device == 'gpu'
        es = td.kernel(nstates=4)[0]
        ref = td.to_cpu().kernel(nstates=4)[0]
        self.assertAlmostEqual(abs(es - ref).max(), 0, 8)
        self.assertAlmostEqual(lib.fp(es[:3]), 0.04757678948025743, 6)

    def test_tddft_b88p86(self):
        mf_bp86 = self.mf_bp86
        td = mf_bp86.TDDFT()
        assert td.device == 'gpu'
        es = td.kernel(nstates=5)[0]
        ref = td.to_cpu().kernel(nstates=5)[0]
        self.assertAlmostEqual(abs(es - ref).max(), 0, 8)
        self.assertAlmostEqual(lib.fp(es[:3]), 0.05374466608381663, 6)

        mol1 = self.mol1
        mf = mol1.UKS().run(xc='b88,p86').run()
        td = mf.TDDFT().to_gpu()
        assert td.device == 'gpu'
        es = td.kernel(nstates=5)[0]
        ref = td.to_cpu().kernel(nstates=5)[0]
        self.assertAlmostEqual(abs(es - ref).max(), 0, 8)

    def test_tddft_b3lyp(self):
        mf_b3lyp = self.mf_b3lyp
        td = mf_b3lyp.TDDFT()
        assert td.device == 'gpu'
        es = td.kernel(nstates=4)[0]
        ref = td.to_cpu().kernel(nstates=4)[0]
        self.assertAlmostEqual(abs(es - ref).max(), 0, 8)
        self.assertAlmostEqual(lib.fp(es[:3]), 0.047793873508724743, 6)

    def test_tddft_camb3lyp(self):
        mol1 = self.mol1
        mf = mol1.UKS(xc='camb3lyp').run()
        td = mf.TDDFT().to_gpu()
        assert td.device == 'gpu'
        es = td.kernel(nstates=4)[0]
        e_ref = td.to_cpu().kernel(nstates=4)[0]
        self.assertAlmostEqual(abs(es[:3]-e_ref[:3]).max(), 0, 8)
        self.assertAlmostEqual(lib.fp(es[:3]), 0.2827429269753051, 6)

    def test_tda_b3lyp(self):
        mf_b3lyp = self.mf_b3lyp
        td = mf_b3lyp.TDA()
        assert td.device == 'gpu'
        es = td.kernel(nstates=4)[0]
        ref = td.to_cpu().kernel()[0]
        self.assertAlmostEqual(abs(es - ref).max(), 0, 8)
        self.assertAlmostEqual(lib.fp(es[:3]), 0.052638024165134974, 6)

    def test_tda_lda(self):
        mf_lda = self.mf_lda
        td = mf_lda.TDA()
        assert td.device == 'gpu'
        es = td.kernel(nstates=5)[0]
        ref = td.to_cpu().kernel(nstates=5)[0]
        self.assertAlmostEqual(abs(es - ref).max(), 0, 8)
        self.assertAlmostEqual(lib.fp(es[:3]), 0.05358614041555257, 6)

        mol1 = self.mol1
        mf = mol1.UKS().run(xc='lda,vwn').run()
        td = mf.TDA().to_gpu()
        assert td.device == 'gpu'
        td.nstates = 5
        es = td.kernel()[0]
        ref = td.to_cpu().kernel()[0]
        self.assertAlmostEqual(abs(es - ref).max(), 0, 8)

    def test_tda_m06l(self):
        mf_m06l = self.mf_m06l
        td = mf_m06l.TDA()
        assert td.device == 'gpu'
        es = td.kernel(nstates=5)[0]
        ref = td.to_cpu().kernel(nstates=5)[0]
        self.assertAlmostEqual(abs(es - ref[:5]).max(), 0, 8)
        self.assertAlmostEqual(lib.fp(es), -0.7531765338077471, 6)

    def test_tda_vind(self):
        mf = self.mf_bp86
        nocca, noccb = mf.nelec
        nmo = mf.mo_energy[0].size
        nvira = nmo - nocca
        nvirb = nmo - noccb
        zs = np.random.rand(3,nocca*nvira+noccb*nvirb)
        ref = mf.to_cpu().TDA().gen_vind()[0](zs)
        dat = mf.TDA().gen_vind()[0](cp.asarray(zs))
        self.assertAlmostEqual(abs(ref - dat).max(), 0, 9)

    def test_tddft_vind(self):
        mf = self.mf_b3lyp
        nocca, noccb = mf.nelec
        nmo = mf.mo_energy[0].size
        nvira = nmo - nocca
        nvirb = nmo - noccb
        zs = np.random.rand(3,2,nocca*nvira+noccb*nvirb)
        ref = mf.to_cpu().TDDFT().gen_vind()[0](zs)
        dat = mf.TDDFT().gen_vind()[0](cp.asarray(zs))
        self.assertAlmostEqual(abs(ref - dat).max(), 0, 9)

    def test_casida_tddft_vind(self):
        mf = self.mf_lda
        nocca, noccb = mf.nelec
        nmo = mf.mo_energy[0].size
        nvira = nmo - nocca
        nvirb = nmo - noccb
        zs = np.random.rand(3,nocca*nvira+noccb*nvirb)
        ref = mf.to_cpu().CasidaTDDFT().gen_vind()[0](zs)
        dat = mf.CasidaTDDFT().gen_vind()[0](cp.asarray(zs))
        self.assertAlmostEqual(abs(ref - dat).max(), 0, 9)

if __name__ == "__main__":
    print("Full Tests for TD-UKS")
    unittest.main()
