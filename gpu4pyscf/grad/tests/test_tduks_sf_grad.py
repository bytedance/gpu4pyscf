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

import pyscf
import numpy as np
import unittest
import pytest
from pyscf import scf, dft, tdscf
import gpu4pyscf
from gpu4pyscf import scf as gpu_scf
from packaging import version
from gpu4pyscf.lib.multi_gpu import num_devices

atom = """
O       0.0000000000     0.0000000000     0.0000000000
H       0.0000000000    -0.7570000000     0.5870000000
H       0.0000000000     0.7570000000     0.5870000000
"""

pyscf_25 = version.parse(pyscf.__version__) <= version.parse("2.5.0")

bas0 = "cc-pvdz"

def setUpModule():
    global mol
    mol = pyscf.M(
        atom=atom,
        basis=bas0,
        max_memory=32000,
        charge=1,
        spin=1,
        output="/dev/null",
        verbose=1,
    )


def tearDownModule():
    global mol
    mol.stdout.close()
    del mol


def benchmark_with_cpu(mol, xc, nstates=3, lindep=1.0e-12, tda=False, extype=0):
    mf = dft.UKS(mol, xc=xc).to_gpu().run()
    tdsf = mf.SFTDA()
    tdsf.extype = extype
    tdsf.collinear = 'mcol'
    tdsf.nstates=5
    tdsf.collinear_samples=10
    output = tdsf.kernel()

    g = tdsf.Gradients()
    g.kernel()

    return g.de


def _check_grad(mol, xc, tol=1e-5, lindep=1.0e-12, disp=None, tda=True, method="cpu", extype=0):
    if not tda:
        raise NotImplementedError("spin-flip TDDFT gradients is not implemented")
    if method == "cpu":
        grad_gpu = benchmark_with_cpu(mol, xc, nstates=5, lindep=lindep, tda=tda, extype=extype)
    else:
        raise NotImplementedError("Only compared with CPU")
        
    return grad_gpu


class KnownValues(unittest.TestCase):
    @unittest.skipIf(num_devices > 1, '')
    def test_grad_b3lyp_tda_spinflip_up_cpu(self):
        grad_gpu = _check_grad(mol, xc="b3lyp", tol=5e-10, method="cpu")
        # ref from pyscf-forge
        ref = np.array([[ 8.79547051e-16,  8.63728537e-14,  1.87755267e-01],
                        [-4.31890391e-16,  2.15026042e-01, -9.38746716e-02],
                        [-4.50003252e-16, -2.15026042e-01, -9.38746716e-02]])
        assert abs(grad_gpu - ref).max() < 1e-5
        
    def test_grad_b3lyp_tda_spinflip_down_cpu(self):
        grad_gpu = _check_grad(mol, xc="b3lyp", tol=5e-10, method="cpu", extype=1)
        # ref from pyscf-forge
        ref = np.array([[-3.01640558e-16,  1.52982216e-13,  5.10689029e-02],
                        [ 1.36165869e-16,  4.52872857e-02, -2.55387304e-02],
                        [-3.08111636e-17, -4.52872857e-02, -2.55387304e-02],])
        assert abs(grad_gpu - ref).max() < 1e-5

    def test_grad_svwn_tda_spinflip_down_cpu(self):
        grad_gpu = _check_grad(mol, xc="svwn", tol=5e-10, method="cpu", extype=1)
        # ref from pyscf-forge
        ref = np.array([[-8.15030724e-16, -6.13885762e-14,  6.41681368e-02],
                        [ 1.12931062e-16,  5.34632826e-02, -3.20887796e-02],
                        [ 7.97399496e-17, -5.34632826e-02, -3.20887796e-02],])
        assert abs(grad_gpu - ref).max() < 1e-5

    def test_grad_camb3lyp_tda_spinflip_down_cpu(self):
        grad_gpu = _check_grad(mol, xc="camb3lyp", tol=5e-10, method="cpu", extype=1)
        # ref from pyscf-forge
        ref = np.array([[-7.43754261e-18, -1.56347842e-13,  4.99263503e-02],
                        [-1.84572351e-17,  4.52908126e-02, -2.49673842e-02],
                        [ 2.40683934e-17, -4.52908126e-02, -2.49673842e-02],])
        assert abs(grad_gpu - ref).max() < 1e-5
    

if __name__ == "__main__":
    print("Full Tests for spin-flip TD-UKS Gradient")
    unittest.main()
