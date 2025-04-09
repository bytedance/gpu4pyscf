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

atom = """
O       0.0000000000     0.0000000000     0.0000000000
H       0.0000000000    -0.7570000000     0.5870000000
H       0.0000000000     0.7570000000     0.5870000000
"""

pyscf_25 = version.parse(pyscf.__version__) <= version.parse("2.5.0")

bas0 = "cc-pvdz"

def diagonalize(a, b, nroots=5):
    a_aa, a_ab, a_bb = a
    b_aa, b_ab, b_bb = b
    nocc_a, nvir_a, nocc_b, nvir_b = a_ab.shape
    a_aa = a_aa.reshape((nocc_a * nvir_a, nocc_a * nvir_a))
    a_ab = a_ab.reshape((nocc_a * nvir_a, nocc_b * nvir_b))
    a_bb = a_bb.reshape((nocc_b * nvir_b, nocc_b * nvir_b))
    b_aa = b_aa.reshape((nocc_a * nvir_a, nocc_a * nvir_a))
    b_ab = b_ab.reshape((nocc_a * nvir_a, nocc_b * nvir_b))
    b_bb = b_bb.reshape((nocc_b * nvir_b, nocc_b * nvir_b))
    a = np.block([[a_aa, a_ab], [a_ab.T, a_bb]])
    b = np.block([[b_aa, b_ab], [b_ab.T, b_bb]])
    abba = np.asarray(np.block([[a, b], [-b.conj(), -a.conj()]]))
    e, xy = np.linalg.eig(abba)
    sorted_indices = np.argsort(e)

    e_sorted = e[sorted_indices]
    xy_sorted = xy[:, sorted_indices]

    e_sorted_final = e_sorted[e_sorted > 1e-3]
    xy_sorted = xy_sorted[:, e_sorted > 1e-3]
    return e_sorted_final[:nroots], xy_sorted[:, :nroots]


def diagonalize_tda(a, nroots=5):
    a_aa, a_ab, a_bb = a
    nocc_a, nvir_a, nocc_b, nvir_b = a_ab.shape
    a_aa = a_aa.reshape((nocc_a * nvir_a, nocc_a * nvir_a))
    a_ab = a_ab.reshape((nocc_a * nvir_a, nocc_b * nvir_b))
    a_bb = a_bb.reshape((nocc_b * nvir_b, nocc_b * nvir_b))
    a = np.block([[a_aa, a_ab], [a_ab.T, a_bb]])
    e, xy = np.linalg.eig(a)
    sorted_indices = np.argsort(e)

    e_sorted = e[sorted_indices]
    xy_sorted = xy[:, sorted_indices]

    e_sorted_final = e_sorted[e_sorted > 1e-3]
    xy_sorted = xy_sorted[:, e_sorted > 1e-3]
    return e_sorted_final[:nroots], xy_sorted[:, :nroots]


def cal_analytic_gradient(mol, td, tdgrad, nocc_a, nvir_a, nocc_b, nvir_b, tda):
    a, b = td.get_ab()

    if tda:
        atmlst = range(mol.natm)
        e_diag, xy_diag = diagonalize_tda(a)
        nsize = nocc_a * nvir_a
        norm1 = np.linalg.norm(xy_diag[:nsize, 0])
        norm2 = np.linalg.norm(xy_diag[nsize:, 0])
        x_aa = xy_diag[:nsize, 0].reshape(nocc_a, nvir_a) * np.sqrt(
            1 / (norm1**2 + norm2**2))
        x_bb = xy_diag[nsize:, 0].reshape(nocc_b, nvir_b) * np.sqrt(
            1 / (norm1**2 + norm2**2))
        de_td = tdgrad.grad_elec(((x_aa, x_bb), (0, 0)))
        gradient_ana = de_td + tdgrad.grad_nuc(atmlst=atmlst)
    else:
        atmlst = range(mol.natm)
        e_diag, xy_diag = diagonalize(a, b)
        nsize1 = nocc_a * nvir_a
        nsize2 = nocc_b * nvir_b
        norm_1 = np.linalg.norm(xy_diag[:nsize1, 0])
        norm_2 = np.linalg.norm(xy_diag[nsize1 : nsize1 + nsize2, 0])
        norm_3 = np.linalg.norm(xy_diag[nsize1 + nsize2 : nsize1 + nsize2 + nsize1, 0])
        norm_4 = np.linalg.norm(xy_diag[nsize1 + nsize2 + nsize1 :, 0])
        norm_factor = np.sqrt(1 / (norm_1**2 + norm_2**2 - norm_3**2 - norm_4**2))
        x_aa = xy_diag[:nsize1, 0] * norm_factor
        x_bb = xy_diag[nsize1 : nsize1 + nsize2, 0] * norm_factor
        y_aa = xy_diag[nsize1 + nsize2 : nsize1 + nsize2 + nsize1, 0] * norm_factor
        y_bb = xy_diag[nsize1 + nsize2 + nsize1 :, 0] * norm_factor
        x_aa = x_aa.reshape(nocc_a, nvir_a)
        x_bb = x_bb.reshape(nocc_b, nvir_b)
        y_aa = y_aa.reshape(nocc_a, nvir_a)
        y_bb = y_bb.reshape(nocc_b, nvir_b)
        x = (x_aa, x_bb)
        y = (y_aa, y_bb)

        de_td = tdgrad.grad_elec((x, y))
        gradient_ana = de_td + tdgrad.grad_nuc(atmlst=atmlst)

    return gradient_ana


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


def benchmark_with_cpu(mol, nstates=3, lindep=1.0e-12, tda=False):
    mf = scf.UHF(mol).to_gpu().run()
    if tda:
        td = mf.TDA()
    else:
        td = mf.TDHF()
    mo_occ = mf.mo_occ
    occidxa = np.where(mo_occ[0]>0)[0]
    occidxb = np.where(mo_occ[1]>0)[0]
    viridxa = np.where(mo_occ[0]==0)[0]
    viridxb = np.where(mo_occ[1]==0)[0]
    nocca = len(occidxa)
    noccb = len(occidxb)
    nvira = len(viridxa)
    nvirb = len(viridxb)

    tdgrad_gpu = gpu4pyscf.grad.tduhf.Gradients(td)
    gpu_gradient = cal_analytic_gradient(mol, td, tdgrad_gpu, nocca, nvira, noccb, nvirb, tda)
    tdgrad_cpu = tdgrad_gpu.to_cpu()
    cpu_gradient = cal_analytic_gradient(mol, td, tdgrad_cpu, nocca, nvira, noccb, nvirb, tda)

    return cpu_gradient, gpu_gradient


def benchmark_with_finite_diff(
        mol_input, delta=0.1, nstates=3, lindep=1.0e-12, tda=False):
    mol = mol_input.copy()
    mf = scf.UHF(mol).to_gpu()
    mf.kernel()
    mo_occ = mf.mo_occ
    occidxa = np.where(mo_occ[0] > 0)[0]
    occidxb = np.where(mo_occ[1] > 0)[0]
    viridxa = np.where(mo_occ[0] == 0)[0]
    viridxb = np.where(mo_occ[1] == 0)[0]
    nocca = len(occidxa)
    noccb = len(occidxb)
    nvira = len(viridxa)
    nvirb = len(viridxb)
    if tda:
        td = gpu4pyscf.tdscf.uhf.TDA(mf)
    else:
        td = gpu4pyscf.tdscf.uhf.TDHF(mf)
    assert td.device == "gpu"
    tdgrad = gpu4pyscf.grad.tduhf.Gradients(td)

    gradient_ana = cal_analytic_gradient(mol, td, tdgrad, nocca, nvira, noccb, nvirb, tda)

    coords = mol.atom_coords(unit="Ang") * 1.0
    natm = coords.shape[0]
    grad = np.zeros((natm, 3))
    for i in range(natm):
        for j in range(3):
            coords_new = coords * 1.0
            coords_new[i, j] += delta
            mol.set_geom_(coords_new, unit="Ang")
            mol.build()
            mf_add = scf.UHF(mol).to_gpu()
            mf_add.run()

            if tda:
                td_add = gpu4pyscf.tdscf.uhf.TDA(mf_add)
                a, b = td_add.get_ab()
                e1 = diagonalize_tda(a)[0]
            else:
                td_add = gpu4pyscf.tdscf.uhf.TDHF(mf_add)
                a, b = td_add.get_ab()
                e1 = diagonalize(a, b)[0]

            e_add = e1[0] + mf_add.e_tot

            coords_new = coords * 1.0
            coords_new[i, j] -= delta
            mol.set_geom_(coords_new, unit="Ang")
            mol.build()
            mf_minus = scf.UHF(mol).to_gpu()
            mf_minus.run()

            if tda:
                td_minus = gpu4pyscf.tdscf.uhf.TDA(mf_minus)
                a, b = td_minus.get_ab()
                e1 = diagonalize_tda(a)[0]
            else:
                td_minus = gpu4pyscf.tdscf.uhf.TDHF(mf_minus)
                a, b = td_minus.get_ab()
                e1 = diagonalize(a, b)[0]

            e_minus = e1[0] + mf_minus.e_tot

            grad[i, j] = (e_add - e_minus) / (delta * 2.0) * 0.52917721092
    return gradient_ana, grad


def _check_grad(mol, tol=1e-6, lindep=1.0e-12, disp=None, tda=False, method="cpu"):
    if method == "cpu":
        gradi_cpu, grad_gpu = benchmark_with_cpu(
            mol, nstates=5, lindep=lindep, tda=tda)
        norm_diff = np.linalg.norm(gradi_cpu - grad_gpu)
    elif method == "numerical":
        grad_gpu, grad = benchmark_with_finite_diff(
            mol, delta=0.005, nstates=5, lindep=lindep, tda=tda)
        norm_diff = np.linalg.norm(grad_gpu - grad)
    assert norm_diff < tol
    return grad_gpu


class KnownValues(unittest.TestCase):
    def test_grad_tda_spinconserve_cpu(self):
        grad_gpu = _check_grad(mol, tol=5e-10, tda=True, method="cpu")
        ref = np.array([[-3.0185736020771e-15,  1.5100260822340e-15, -6.4464824062032e-02],
                        [ 1.7200484393589e-15,  7.0426620850669e-02, 3.2232412031016e-02],
                        [ 1.2985251627181e-15, -7.0426620850669e-02, 3.2232412031018e-02]])
        assert abs(grad_gpu - ref).max() < 1e-5

    def test_grad_tda_spinconserve_numerical(self):
        _check_grad(mol, tol=1e-4, tda=True, method="numerical")

    def test_grad_tdhf_spinconserve_cpu(self):
        grad_gpu = _check_grad(mol, tol=5e-10, lindep=1.0e-6, tda=False, method="cpu")
        ref = np.array([[ 2.4083810840674e-15,  1.1005204522931e-15, -6.4107899188727e-02],
                        [-1.5974943646053e-15,  6.9613558808388e-02, 3.2053949594365e-02],
                        [-8.1088671946197e-16, -6.9613558808391e-02, 3.2053949594365e-02]])
        assert abs(grad_gpu - ref).max() < 1e-5

    # def test_grad_tdhf_spinconserve_numerical(self):
    #     _check_grad(mol, tol=1e-4, lindep=1.0e-6, tda=False, method="numerical")


if __name__ == "__main__":
    print("Full Tests for TD-UHF Gradient")
    unittest.main()
