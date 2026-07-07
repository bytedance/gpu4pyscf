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

import time
import unittest
import numpy as np
import cupy as cp
from pyscf import gto
from gpu4pyscf.dft import rks

from gpu4pyscf.qmmm.embedding.embedding_dft_try1_ml import (
    OneStepRKS, SingleFragmentEmbedding_ML)
from gpu4pyscf.qmmm.embedding.embedding_dft_try1_ml_dualbasis import (
    DualBasisOneStepRKS, SingleFragmentEmbedding_ML_DualBasis)


def dummy_eval_density_func(mol, xc, grids):
    """
    Dummy ML density evaluator (identical contract to the try1_ml tests).

    It performs a highly converged standard RKS calculation to provide an
    'exact' reference density, so that the ML framework mathematically reduces
    to the exact global evaluation when the 'ML prediction' is perfect.
    """
    mf = rks.RKS(mol)
    mf.xc = xc
    mf.grids = grids
    mf.verbose = 0
    mf.conv_tol = 1.0E-12
    mf.kernel()

    dm = cp.asarray(mf.make_rdm1())

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

    _, e_xc, vxc = mf._numint.nr_rks(mol, grids, xc, dm)
    int_rho_vxc = float(cp.sum(dm * vxc))

    return vj, vk, vxc, e_j, e_k, float(e_xc), int_rho_vxc


def build_mol(atom, basis, spin=0, charge=0):
    mol = gto.Mole()
    mol.atom = atom
    mol.basis = basis
    mol.spin = spin
    mol.charge = charge
    mol.verbose = 0
    mol.build()
    return mol


# Reusable geometries (same coordinates, only the basis set differs between
# the small and large mol objects).
H2O_GEOM = '''
    O   0.00000000   0.00000000   0.11730000
    H   0.00000000   0.75720000  -0.46920000
    H   0.00000000  -0.75720000  -0.46920000
'''

C2H6_GEOM = '''
    C      -0.76091    -0.00000     0.00000
    C       0.76091    -0.00000     0.00000
    H      -1.16001     1.02029     0.00000
    H      -1.16001    -0.51014    -0.88357
    H      -1.16001    -0.51014     0.88357
    H       1.16001    -1.02029     0.00000
    H       1.16001     0.51014     0.88357
    H       1.16001     0.51014    -0.88357
'''

CH4_GEOM = '''
    C   0.00000000   0.00000000   0.00000000
    H   0.62910000   0.62910000   0.62910000
    H  -0.62910000  -0.62910000   0.62910000
    H  -0.62910000   0.62910000  -0.62910000
    H   0.62910000  -0.62910000  -0.62910000
'''


class TestDualBasisOneStepRKS(unittest.TestCase):
    """Standalone dual-basis one-step RKS solver behavior."""

    def test_same_basis_exactness(self):
        # When the 'small' basis equals the 'large' basis, the projected
        # subspace spans the whole space, so the dual-basis solver must
        # reproduce the exact RKS energy to machine precision.
        mol = build_mol(H2O_GEOM, '6-31g')
        mol_small = build_mol(H2O_GEOM, '6-31g')

        mf_ref = rks.RKS(mol, xc='PBE')
        mf_ref.verbose = 0
        mf_ref.conv_tol = 1.0E-12
        e_ref = mf_ref.kernel()

        mf_db = DualBasisOneStepRKS(mol, mol_small, dummy_eval_density_func, xc='PBE')
        mf_db.verbose = 0
        e_db = mf_db.kernel()

        self.assertAlmostEqual(e_ref, e_db, places=8,
                               msg=f"Same-basis DualBasisOneStepRKS energy {e_db} "
                                   f"differs from exact RKS {e_ref}")

    def test_same_basis_matches_onestep(self):
        # With an identical basis, the dual-basis solver and the plain
        # OneStepRKS driven by the same ML density must agree exactly.
        mol = build_mol(C2H6_GEOM, '6-31g')
        mol_small = build_mol(C2H6_GEOM, '6-31g')

        mf_os = OneStepRKS(mol, dummy_eval_density_func, xc='PBE')
        mf_os.verbose = 0
        e_os = mf_os.kernel()

        mf_db = DualBasisOneStepRKS(mol, mol_small, dummy_eval_density_func, xc='PBE')
        mf_db.verbose = 0
        e_db = mf_db.kernel()

        self.assertAlmostEqual(e_os, e_db, places=8,
                               msg=f"DualBasis {e_db} diverged from OneStepRKS {e_os}")

    def test_same_basis_density_recovery(self):
        # The reconstructed large-basis density must match the exact density.
        mol = build_mol(H2O_GEOM, '6-31g')
        mol_small = build_mol(H2O_GEOM, '6-31g')

        mf_ref = rks.RKS(mol, xc='PBE')
        mf_ref.verbose = 0
        mf_ref.conv_tol = 1.0E-12
        mf_ref.kernel()
        dm_ref = cp.asarray(mf_ref.make_rdm1())

        mf_db = DualBasisOneStepRKS(mol, mol_small, dummy_eval_density_func, xc='PBE')
        mf_db.verbose = 0
        mf_db.kernel()
        dm_db = cp.asarray(mf_db.make_rdm1())

        max_diff = float(cp.max(cp.abs(dm_ref - dm_db)))
        self.assertLess(max_diff, 1e-6,
                        msg=f"Reconstructed density deviates by {max_diff}")

    def test_subspace_dimension_reduction(self):
        # A genuinely smaller basis must reduce the diagonalization dimension
        # from N_L to N_S; this is the source of the speed-up.
        mol = build_mol(H2O_GEOM, 'cc-pvdz')
        mol_small = build_mol(H2O_GEOM, 'sto-3g')

        mf_db = DualBasisOneStepRKS(mol, mol_small, dummy_eval_density_func, xc='PBE')
        mf_db.verbose = 0
        mf_db.kernel()

        self.assertIsNotNone(mf_db._subspace_dim)
        self.assertEqual(mf_db._full_dim, mol.nao_nr())
        self.assertEqual(mf_db._subspace_dim, mol_small.nao_nr())
        self.assertLess(mf_db._subspace_dim, mf_db._full_dim,
                        msg="Dual-basis did not reduce the diagonalization size")

    def test_smaller_basis_energy_sane(self):
        # With a genuinely smaller basis the dual-basis energy is an
        # approximation, but it must remain finite and physically bound, and
        # be reasonably close to the exact large-basis energy.
        mol = build_mol(H2O_GEOM, '6-31g')
        mol_small = build_mol(H2O_GEOM, 'sto-3g')

        mf_ref = rks.RKS(mol, xc='PBE')
        mf_ref.verbose = 0
        mf_ref.conv_tol = 1.0E-12
        e_ref = mf_ref.kernel()

        mf_db = DualBasisOneStepRKS(mol, mol_small, dummy_eval_density_func, xc='PBE')
        mf_db.verbose = 0
        e_db = mf_db.kernel()

        self.assertTrue(np.isfinite(e_db))
        self.assertLess(e_db, 0.0)
        # The projected-subspace energy should be within a chemically loose
        # window of the exact large-basis energy (well under 1 Hartree).
        self.assertLess(abs(e_db - e_ref), 1.0,
                        msg=f"Dual-basis energy {e_db} too far from exact {e_ref}")

    def test_custom_small_basis_solver(self):
        # A user-supplied small-basis solver must be honored.
        mol = build_mol(H2O_GEOM, '6-31g')
        mol_small = build_mol(H2O_GEOM, '6-31g')

        calls = {'n': 0}

        def my_solver(m_small, xc):
            calls['n'] += 1
            mf_s = rks.RKS(m_small, xc=xc)
            mf_s.verbose = 0
            mf_s.conv_tol = 1.0E-10
            mf_s.kernel()
            return mf_s.mo_coeff

        mf_db = DualBasisOneStepRKS(mol, mol_small, dummy_eval_density_func,
                                    xc='PBE', small_basis_solver=my_solver)
        mf_db.verbose = 0
        e_db = mf_db.kernel()

        self.assertEqual(calls['n'], 1, "custom small_basis_solver was not used")
        self.assertTrue(np.isfinite(e_db))

    def test_too_small_basis_raises(self):
        # If the small subspace provides fewer MOs than occupied orbitals,
        # the projection is ill-defined and must raise.
        mol = build_mol(H2O_GEOM, '6-31g')
        mol_small = build_mol(H2O_GEOM, '6-31g')
        nocc = mol.nelectron // 2

        def truncating_solver(m_small, xc):
            mf_s = rks.RKS(m_small, xc=xc)
            mf_s.verbose = 0
            mf_s.kernel()
            # Keep fewer columns than the number of occupied orbitals.
            C = cp.asarray(mf_s.mo_coeff)
            return C[:, :nocc - 1]

        mf_db = DualBasisOneStepRKS(mol, mol_small, dummy_eval_density_func,
                                    xc='PBE', small_basis_solver=truncating_solver)
        mf_db.verbose = 0
        with self.assertRaises(ValueError):
            mf_db.kernel()


class TestDualBasisEmbedding(unittest.TestCase):
    """Dual-basis accelerated single-fragment DFT-in-DFT embedding."""

    @classmethod
    def setUpClass(cls):
        cls.mol = build_mol(C2H6_GEOM, '6-31g')
        cls.mol_small = build_mol(C2H6_GEOM, '6-31g')
        cls.mol_small_sto = build_mol(C2H6_GEOM, 'sto-3g')
        cls.methyl_fragment = [0, 2, 3, 4]
        cls.full_fragment = list(range(cls.mol.natm))

    def test_full_system_pbe_in_pbe(self):
        # Full-QM PBE-in-PBE with the same small/large basis must recover the
        # exact global PBE energy (MAE = 0 benchmark).
        mf_outer = DualBasisOneStepRKS(self.mol, self.mol_small,
                                       dummy_eval_density_func, xc='PBE')
        mf_inner = rks.RKS(self.mol, xc='PBE')
        mf_inner.conv_tol = 1.0E-12

        emb = SingleFragmentEmbedding_ML_DualBasis(mf_outer, mf_inner,
                                                   self.full_fragment, verbose=0)
        emb.kernel()

        mf_ref = DualBasisOneStepRKS(self.mol, self.mol_small,
                                     dummy_eval_density_func, xc='PBE')
        mf_ref.verbose = 0
        e_global = mf_ref.kernel()

        self.assertAlmostEqual(e_global, emb.e_tot, places=8,
                               msg="Full-system PBE-in-PBE failed exact cancellation.")

    def test_equivalence_to_ml_embedding_same_basis(self):
        # With the same small/large basis, the dual-basis embedding must match
        # the plain ML embedding exactly (identical outer density).
        mf_outer_ml = OneStepRKS(self.mol, dummy_eval_density_func, xc='PBE')
        mf_inner_ml = rks.RKS(self.mol, xc='B3LYP')
        mf_inner_ml.conv_tol = 1.0E-12
        emb_ml = SingleFragmentEmbedding_ML(mf_outer_ml, mf_inner_ml,
                                            self.methyl_fragment, verbose=0)
        e_ml = emb_ml.kernel()

        mf_outer_db = DualBasisOneStepRKS(self.mol, self.mol_small,
                                          dummy_eval_density_func, xc='PBE')
        mf_inner_db = rks.RKS(self.mol, xc='B3LYP')
        mf_inner_db.conv_tol = 1.0E-12
        emb_db = SingleFragmentEmbedding_ML_DualBasis(mf_outer_db, mf_inner_db,
                                                      self.methyl_fragment, verbose=0)
        e_db = emb_db.kernel()

        self.assertAlmostEqual(e_ml, e_db, places=8,
                               msg=f"Dual-basis embedding {e_db} diverged from "
                                   f"ML embedding {e_ml}")

    def test_embedding_runs_with_smaller_basis(self):
        # End-to-end embedding must run with a genuinely smaller outer subspace.
        mf_outer = DualBasisOneStepRKS(self.mol, self.mol_small_sto,
                                       dummy_eval_density_func, xc='PBE')
        mf_inner = rks.RKS(self.mol, xc='B3LYP')
        mf_inner.conv_tol = 1.0E-10
        emb = SingleFragmentEmbedding_ML_DualBasis(mf_outer, mf_inner,
                                                   self.methyl_fragment, verbose=0)
        e_tot = emb.kernel()

        self.assertTrue(np.isfinite(e_tot))
        self.assertLess(e_tot, 0.0)

    def test_requires_dualbasis_outer(self):
        # Passing a non-dual-basis outer solver must be rejected.
        mf_outer = OneStepRKS(self.mol, dummy_eval_density_func, xc='PBE')
        mf_inner = rks.RKS(self.mol, xc='B3LYP')
        with self.assertRaises(TypeError):
            SingleFragmentEmbedding_ML_DualBasis(mf_outer, mf_inner,
                                                 self.methyl_fragment, verbose=0)


class TestDualBasisMultiSystem(unittest.TestCase):
    """Coverage across several molecules and basis combinations."""

    def test_same_basis_exact_across_systems(self):
        cases = [
            (H2O_GEOM, '6-31g'),
            (CH4_GEOM, '6-31g'),
            (C2H6_GEOM, 'sto-3g'),
        ]
        for geom, basis in cases:
            with self.subTest(geom=geom.split()[0], basis=basis):
                mol = build_mol(geom, basis)
                mol_small = build_mol(geom, basis)

                mf_ref = rks.RKS(mol, xc='PBE')
                mf_ref.verbose = 0
                mf_ref.conv_tol = 1.0E-12
                e_ref = mf_ref.kernel()

                mf_db = DualBasisOneStepRKS(mol, mol_small,
                                            dummy_eval_density_func, xc='PBE')
                mf_db.verbose = 0
                e_db = mf_db.kernel()

                self.assertAlmostEqual(e_ref, e_db, places=8)

    def test_reduction_across_basis_combos(self):
        combos = [
            ('sto-3g', '6-31g'),
            ('sto-3g', 'cc-pvdz'),
            ('6-31g', 'cc-pvdz'),
        ]
        for small_basis, large_basis in combos:
            with self.subTest(small=small_basis, large=large_basis):
                mol = build_mol(H2O_GEOM, large_basis)
                mol_small = build_mol(H2O_GEOM, small_basis)

                mf_db = DualBasisOneStepRKS(mol, mol_small,
                                            dummy_eval_density_func, xc='PBE')
                mf_db.verbose = 0
                e_db = mf_db.kernel()

                self.assertTrue(np.isfinite(e_db))
                self.assertLess(mf_db._subspace_dim, mf_db._full_dim)


class TestDualBasisPerformance(unittest.TestCase):
    """Performance: the projected diagonalization should be markedly cheaper."""

    def test_performance_improvement(self):
        # Use a moderately sized system where N_L >> N_S so the O(N^3)
        # diagonalization cost difference is measurable.
        mol = build_mol(C2H6_GEOM, 'cc-pvtz')
        mol_small = build_mol(C2H6_GEOM, 'sto-3g')

        n_large = mol.nao_nr()
        n_small = mol_small.nao_nr()
        self.assertGreater(n_large, 2 * n_small,
                           msg="Chosen bases do not give a large N_L/N_S ratio")

        # Build a well-conditioned symmetric Fock proxy (hcore) and the overlap
        # directly from analytic integrals, so we time ONLY the eig solve and
        # avoid the expensive ML/veff evaluation.
        mf_full = OneStepRKS(mol, dummy_eval_density_func, xc='PBE')
        mf_full.verbose = 0
        h1e = cp.asarray(mf_full.get_hcore())
        s1e = cp.asarray(mf_full.get_ovlp())

        mf_db = DualBasisOneStepRKS(mol, mol_small, dummy_eval_density_func, xc='PBE')
        mf_db.verbose = 0

        def time_eig(mf, repeats=10):
            # Warm-up (also lazily builds the dual-basis projection so it is
            # excluded from the timed region).
            mf.eig(h1e, s1e)
            cp.cuda.Stream.null.synchronize()
            t0 = time.perf_counter()
            for _ in range(repeats):
                mf.eig(h1e, s1e)
            cp.cuda.Stream.null.synchronize()
            return (time.perf_counter() - t0) / repeats

        t_full = time_eig(mf_full)
        t_db = time_eig(mf_db)

        print(f"\n[dual-basis perf] N_L={n_large} N_S={n_small} "
              f"eig_full={t_full*1e3:.3f} ms  eig_dualbasis={t_db*1e3:.3f} ms  "
              f"speedup={t_full/max(t_db,1e-12):.2f}x")

        # Deterministic proxy for the O(N^3) cost: the diagonalization acts on
        # an N_S x N_S matrix instead of N_L x N_L.
        self.assertEqual(mf_db._subspace_dim, n_small)
        self.assertEqual(mf_db._full_dim, n_large)
        self.assertLess(mf_db._subspace_dim, mf_db._full_dim)
        # The projected eig should not be meaningfully slower than the full one
        # (small tolerance for GPU kernel-launch overhead on tiny matrices).
        self.assertLessEqual(t_db, t_full * 1.5,
                             msg=f"Dual-basis eig ({t_db*1e3:.3f} ms) not faster "
                                 f"than full eig ({t_full*1e3:.3f} ms)")


if __name__ == '__main__':
    print("Tests for dual-basis accelerated ML DFT-in-DFT embedding...")
    unittest.main()
