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
from gpu4pyscf.scf import hf as gpu_hf
from gpu4pyscf.qmmm.embedding.embedding_dmet import (
    DMET, density_matrix_decompose, get_fragment_ao_indices,
    _orthogonalize, build_core_dm, _as_cupy
)
from gpu4pyscf.qmmm.embedding import embedding_dmet


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.Mole()
        cls.mol.atom = '''
            H 0.0 0.0 0.0
            H 0.0 0.0 1.0
            H 0.0 0.0 2.0
            H 0.0 0.0 3.0
        '''
        cls.mol.basis = 'sto-3g'
        cls.mol.spin = 0
        cls.mol.charge = 0
        cls.mol.verbose = 4
        cls.mol.build()

        cls.fragments = [[0, 1], [2, 3]]

        cls.mf_outer = gpu_hf.RHF(cls.mol)
        cls.mf_outer.conv_tol = 1e-14
        cls.mf_inner_template = gpu_hf.RHF(cls.mol)
        cls.mf_inner_template.conv_tol = 1e-14

        cls.mol2 = gto.Mole()
        cls.mol2.atom = '''
            C      -0.76091    -0.00000     0.00000
            C       0.76091    -0.00000     0.00000
            H      -1.16001     1.02029     0.00000
            H      -1.16001    -0.51014    -0.88357
            H      -1.16001    -0.51014     0.88357
            H       1.16001    -1.02029     0.00000
            H       1.16001     0.51014     0.88357
            H       1.16001     0.51014    -0.88357
        '''
        cls.mol2.basis = '6-31g'
        cls.mol2.spin = 0
        cls.mol2.charge = 0
        cls.mol2.verbose = 4
        cls.mol2.build()

        cls.fragments2 = [[0, 2, 3, 4], [1, 5, 6, 7]]

        cls.mf_outer2 = gpu_hf.RHF(cls.mol2)
        cls.mf_outer2.conv_tol = 1e-12
        cls.mf_inner_template2 = gpu_hf.RHF(cls.mol2)
        cls.mf_inner_template2.conv_tol = 1e-12

        cls.mol3 = gto.Mole()
        cls.mol3.atom = '''
            C   1.4522500000  -2.8230000000   0.0000000000
            C   1.4522500000  -1.2830000000   0.0000000000
            C   0.0002500000  -0.7700000000   0.0000000000
            C   0.0002500000   0.7700000000   0.0000000000
            C  -1.4517500000   1.2830000000   0.0000000000
            C  -1.4517500000   2.8230000000   0.0000000000
            H   2.4792500000  -3.1870000000   0.0000000000
            H   0.9382500000  -3.1870000000   0.8900000000
            H   0.9382500000  -3.1870000000  -0.8900000000
            H   1.9652500000  -0.9200000000   0.8900000000
            H   1.9652500000  -0.9200000000  -0.8900000000
            H  -0.5137500000  -1.1330000000  -0.8900000000
            H  -0.5137500000  -1.1330000000   0.8900000000
            H   0.5132500000   1.1330000000   0.8900000000
            H   0.5132500000   1.1330000000  -0.8900000000
            H  -1.9657500000   0.9200000000  -0.8900000000
            H  -1.9657500000   0.9200000000   0.8900000000
            H  -2.4797500000   3.1870000000   0.0000000000
            H  -0.9377500000   3.1870000000   0.8900000000
            H  -0.9377500000   3.1870000000  -0.8900000000
        '''
        cls.mol3.basis = 'def2svp'
        cls.mol3.spin = 0
        cls.mol3.charge = 0
        cls.mol3.verbose = 4
        cls.mol3.build()

        cls.fragments3 = [[0, 6, 7, 8],
                          [1, 9, 10],
                          [2, 11, 12],
                          [3, 13, 14],
                          [4, 15, 16],
                          [5, 17, 18, 19]]

        cls.mf_outer3 = gpu_hf.RHF(cls.mol3)
        cls.mf_outer3.conv_tol = 1e-12
        cls.mf_inner_template3 = gpu_hf.RHF(cls.mol3)
        cls.mf_inner_template3.conv_tol = 1e-12

    @classmethod
    def tearDownClass(cls):
        del cls.mol
        del cls.mf_outer
        del cls.mf_inner_template
        del cls.mol2
        del cls.mf_outer2
        del cls.mf_inner_template2
        del cls.mol3
        del cls.mf_outer3
        del cls.mf_inner_template3

    def test_dmet_initialization(self):
        dmet_solver = DMET(
            mf_outer=self.mf_outer,
            mf_inner=self.mf_inner_template,
            fragments=self.fragments,
            threshold=1e-2
        )

        nao = self.mol.nao_nr()

        self.assertEqual(dmet_solver.nfrags, 2, "Number of fragments should be 2.")
        self.assertEqual(len(dmet_solver.frag_idx), 2, "Fragment indices list should have length 2.")

        self.assertEqual(dmet_solver.u_ao.shape, (nao, nao), "Correlation potential u_ao should be of shape (nao, nao).")
        self.assertTrue(isinstance(dmet_solver.u_ao, cp.ndarray), "Correlation potential should be a CuPy array.")
        self.assertEqual(dmet_solver.threshold, 1e-2, "Default threshold should be 1e-2.")

    def test_fragment_ao_indices(self):
        frag_idx = get_fragment_ao_indices(self.mol, [0, 1])
        self.assertTrue(isinstance(frag_idx, cp.ndarray))
        self.assertEqual(frag_idx.dtype, cp.int32)

    def test_density_matrix_decompose(self):
        mf = gpu_hf.RHF(self.mol)
        mf.conv_tol = 1e-14
        mf.kernel()

        mo_coeff = _as_cupy(mf.mo_coeff)
        mo_occ = _as_cupy(mf.mo_occ)
        C_occ = mo_coeff[:, mo_occ > 0]
        S = _as_cupy(mf.get_ovlp())
        D = _as_cupy(mf.make_rdm1())

        frag_idx = cp.arange(0, 2, dtype=cp.int32)
        env_idx = cp.arange(2, 4, dtype=cp.int32)

        frag_orb, bath_orb, core_orb, info = density_matrix_decompose(
            C_occ, S, frag_idx, env_idx, threshold=1e-2
        )

        # Eigenvalues should be in [0, 2]
        eigvals = info['eigenvalues']
        self.assertTrue(float(cp.all(eigvals >= -1e-10)), "Eigenvalues should be non-negative.")
        self.assertTrue(float(cp.all(eigvals <= 2.0 + 1e-10)), "Eigenvalues should not exceed 2.")

        # Check eigenvalue sum equals Tr(S_A_full @ D @ S_full_A @ S_A_inv)
        S_A = S[cp.ix_(frag_idx, frag_idx)]
        S_A_inv = cp.linalg.pinv(S_A)
        S_A_full = S[frag_idx, :]
        S_full_A = S[:, frag_idx]
        n_frag_electrons = float(cp.trace(S_A_full @ D @ S_full_A @ S_A_inv))
        
        self.assertAlmostEqual(float(cp.sum(eigvals)), n_frag_electrons, places=10,
                               msg="Sum of eigenvalues should equal fragment electron count.")

        # Verify orthonormality of returned orbitals where applicable
        if bath_orb.shape[1] > 0:
            # density_matrix_decompose returns bath/complement orbitals in the
            # full AO representation for the non-orthogonal implementation.
            bath_orth = bath_orb.T @ S @ bath_orb
            err = float(cp.abs(bath_orth - cp.eye(bath_orb.shape[1])).max())
            self.assertTrue(err < 1e-8, f"Bath orbitals not orthonormal, max error: {err}")

        if core_orb.shape[1] > 0:
            core_orth = core_orb.T @ S @ core_orb
            err = float(cp.abs(core_orth - cp.eye(core_orb.shape[1])).max())
            self.assertTrue(err < 1e-8, f"Core orbitals not orthonormal, max error: {err}")

        # frag_orb should be orthonormal in fragment S metric
        if frag_orb.shape[1] > 0:
            S_A = S[cp.ix_(frag_idx, frag_idx)]
            frag_orth = frag_orb.T @ S_A @ frag_orb
            err = float(cp.abs(frag_orth - cp.eye(frag_orb.shape[1])).max())
            self.assertTrue(err < 1e-8, f"Fragment orbitals not orthonormal, max error: {err}")

    def test_build_core_dm(self):
        mf = gpu_hf.RHF(self.mol)
        mf.conv_tol = 1e-14
        mf.kernel()

        C = _as_cupy(mf.mo_coeff)
        C_occ = C[:, mf.mo_occ > 0]
        S = _as_cupy(mf.get_ovlp())
        nao = self.mol.nao_nr()

        frag_idx = cp.arange(0, 2, dtype=cp.int32)
        env_idx = cp.arange(2, 4, dtype=cp.int32)

        frag_orb, bath_orb, core_orb, info = density_matrix_decompose(
            C_occ, S, frag_idx, env_idx, threshold=1e-2
        )

        dm_core = build_core_dm(env_idx, core_orb, nao, S)

        # Core DM should be symmetric
        err_sym = float(cp.abs(dm_core - dm_core.T).max())
        self.assertTrue(err_sym < 1e-10, f"Core DM not symmetric, max error: {err_sym}")

        # Core electron count should match info
        if core_orb.shape[1] > 0:
            n_core_elec = float(cp.trace(dm_core @ S))
            self.assertAlmostEqual(n_core_elec, info['n_core_electrons'], places=8,
                                   msg="Core electron count mismatch.")

    def test_orthogonalize(self):
        mf = gpu_hf.RHF(self.mol)
        mf.conv_tol = 1e-14
        mf.kernel()
        S = _as_cupy(mf.get_ovlp())

        # Create random vectors
        np.random.seed(42)
        n = S.shape[0]
        vecs_np = np.random.randn(n, 3)
        vecs = cp.asarray(vecs_np)

        C = _orthogonalize(vecs, S)
        CtSC = C.T @ S @ C
        err = float(cp.abs(CtSC - cp.eye(C.shape[1])).max())
        self.assertTrue(err < 1e-10, f"_orthogonalize failed, max error: {err}")

        # Test with zero-size input
        C0 = _orthogonalize(cp.zeros((n, 0)), S)
        self.assertEqual(C0.shape[1], 0)

    def test_dmet_execution_and_convergence(self):
        dmet_solver = DMET(
            mf_outer=self.mf_outer,
            mf_inner=self.mf_inner_template,
            fragments=self.fragments,
            threshold=1e-2,
            max_macro_iter=20,
            macro_tol=1e-3
        )

        e_tot = dmet_solver.kernel()

        self.mf_outer.mo_coeff = None
        e_tot_ref = self.mf_outer.kernel()

        # When outer and inner are the same (HF-in-HF), DMET should match HF energy
        self.assertTrue(np.abs(e_tot - e_tot_ref) < 1e-4,
                        f"DMET energy {e_tot} should be close to reference HF energy {e_tot_ref}, diff={np.abs(e_tot - e_tot_ref)}")

    def test_dmet_execution_ethane(self):
        dmet_solver = DMET(
            mf_outer=self.mf_outer2,
            mf_inner=self.mf_inner_template2,
            fragments=self.fragments2,
            threshold=1e-3,
            max_macro_iter=20,
            macro_tol=1e-3
        )

        e_tot = dmet_solver.kernel()

        self.mf_outer2.mo_coeff = None
        e_tot_ref = self.mf_outer2.kernel()

        self.assertTrue(np.abs(e_tot - e_tot_ref) < 1e-3,
                        f"DMET energy {e_tot} should be close to reference HF energy {e_tot_ref}, diff={np.abs(e_tot - e_tot_ref)}")

    def test_dmet_template_isolation(self):
        dmet_solver = DMET(
            mf_outer=self.mf_outer2,
            mf_inner=self.mf_inner_template2,
            fragments=self.fragments2,
            threshold=1e-2,
            max_macro_iter=3,
            macro_tol=1e-3
        )
        dmet_solver.kernel()

        self.mf_inner_template2.mo_coeff = None
        self.mf_inner_template2.kernel()

        self.assertTrue(self.mf_inner_template2.converged,
                        "The inner template was poisoned by DMET macro-loops and failed to converge!")

    def test_correlation_potential_symmetry(self):
        dmet_solver = DMET(
            mf_outer=self.mf_outer,
            mf_inner=self.mf_inner_template,
            fragments=self.fragments,
            threshold=1e-2,
            max_macro_iter=2
        )
        dmet_solver.kernel()

        u = dmet_solver.u_ao

        sym_err = float(cp.abs(u - u.T).max())
        self.assertTrue(sym_err < 1e-10, f"Correlation potential u_ao is not symmetric. Max err: {sym_err}")

    def test_multifragment_algebraic_and_conservation(self):
        dmet_solver = DMET(
            mf_outer=self.mf_outer3,
            mf_inner=self.mf_inner_template3,
            fragments=self.fragments3,
            threshold=1e-2,
            max_macro_iter=1
        )
        dmet_solver.kernel()

        S_ao = cp.asarray(self.mf_outer3.get_ovlp())
        n_total_elec = float(self.mol3.nelectron)

        for ifrag in range(dmet_solver.nfrags):
            B = dmet_solver.B[ifrag]
            D_core = dmet_solver.dm_core[ifrag]
            D_emb_high = cp.asarray(dmet_solver.mf_inner[ifrag].make_rdm1())

            # In non-orthogonal DMET, B^T S B is S_emb, NOT the identity matrix.
            # We verify the fragment block is S_AA and the bath block is I.
            S_emb = B.T @ S_ao @ B
            n_frag = len(dmet_solver.frag_idx[ifrag])
            
            S_AA = S_ao[cp.ix_(dmet_solver.frag_idx[ifrag], dmet_solver.frag_idx[ifrag])]
            max_frag_err = float(cp.abs(S_emb[:n_frag, :n_frag] - S_AA).max())
            self.assertTrue(max_frag_err < 1e-8,
                            f"Fragment {ifrag}: Fragment block of S_emb != S_AA. Max err: {max_frag_err}")

            n_bath = B.shape[1] - n_frag
            if n_bath > 0:
                max_bath_err = float(cp.abs(S_emb[n_frag:, n_frag:] - cp.eye(n_bath)).max())
                self.assertTrue(max_bath_err < 1e-8,
                                f"Fragment {ifrag}: Bath block of S_emb is not orthonormal. Max err: {max_bath_err}")

            # Check Core DM spatial isolation from the active space
            core_overlap = B.T @ S_ao @ D_core @ S_ao @ B
            max_overlap_err = float(cp.abs(core_overlap).max())
            self.assertTrue(max_overlap_err < dmet_solver.threshold,
                            f"Fragment {ifrag}: Core DM leaks into Active Space. Max err: {max_overlap_err}")

            # Check total electron conservation for this fragment representation
            D_emb_ao = B @ D_emb_high @ B.T
            D_total_ao = D_core + D_emb_ao
            n_elec_calc = float(cp.trace(D_total_ao @ S_ao))
            self.assertAlmostEqual(n_elec_calc, n_total_elec, places=6,
                                   msg=f"Fragment {ifrag}: Electron loss detected. {n_elec_calc} != {n_total_elec}")

    def test_threshold_electron_tolerance(self):
        """Test that threshold properly controls electron count error."""
        dmet_solver = DMET(
            mf_outer=self.mf_outer,
            mf_inner=self.mf_inner_template,
            fragments=self.fragments,
            threshold=1e-2,
            max_macro_iter=1
        )

        if not self.mf_outer.converged:
            self.mf_outer.kernel()

        # Use C_occ for the first argument, and keep D for the D_ao kwarg
        mo_coeff = _as_cupy(self.mf_outer.mo_coeff)
        mo_occ = _as_cupy(self.mf_outer.mo_occ)
        C_occ = mo_coeff[:, mo_occ > 0]
        
        D = _as_cupy(self.mf_outer.make_rdm1())
        S = _as_cupy(self.mf_outer.get_ovlp())

        for ifrag in range(dmet_solver.nfrags):
            frag_idx = dmet_solver.frag_idx[ifrag]
            env_idx = dmet_solver.env_idx[ifrag]
            
            # Pass C_occ as the first argument, not D
            frag_orb, bath_orb, core_orb, info = density_matrix_decompose(
                C_occ, S, frag_idx, env_idx, threshold=1e-2, D_ao=D
            )
            # Core cumulative electrons should be within threshold
            self.assertTrue(info['cum_e_core'] <= 1e-2 + 1e-10,
                            f"Core electron error cumulative {info['cum_e_core']} exceeds threshold.")
                            
            # Use 'n_pure_fragment_nos' instead of 'n_frag_orbitals'
            ideal_frag_e = 2.0 * info['n_pure_fragment_nos']
            frag_err_diff = abs(ideal_frag_e - info['cum_e_frag'])
            self.assertTrue(frag_err_diff <= 1e-2 + 1e-10,
                            f"Fragment electron deviation {frag_err_diff} exceeds threshold.")

if __name__ == '__main__':
    print("Full Tests for DMET (density matrix diagonalization, non-orthogonal AO)")
    unittest.main()
