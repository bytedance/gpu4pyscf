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


import numpy as np
import cupy as cp
from pyscf import gto
from gpu4pyscf.dft import rks
from gpu4pyscf.lib.cupy_helper import eigh
from gpu4pyscf.qmmm.embedding.embedding import _as_cupy
from gpu4pyscf.qmmm.embedding.embedding_dft_try1_ml import (
    OneStepRKS, SingleFragmentEmbedding_ML)


class DualBasisOneStepRKS(OneStepRKS):
    """
    One-step RKS accelerated by a small/large dual-basis subspace projection.

    This class behaves like :class:`OneStepRKS` (a non-iterative, ML-density
    driven RKS on the *large* basis ``mol``), but replaces the expensive
    large-basis diagonalization ``eig(F_L, S_L)`` with a diagonalization
    projected onto the small-basis molecular-orbital subspace.

    Parameters
    ----------
    mol : gto.Mole
        The *large* (target) basis molecule. All physics/energies are defined
        here, exactly as in the parent :class:`OneStepRKS`.
    mol_small : gto.Mole
        The *small* basis molecule. Must describe the same geometry / electron
        count as ``mol`` but with a cheaper (smaller) basis set. Used only to
        build the projection subspace.
    eval_density_func : callable
        External ML density evaluator, identical signature/return contract to
        :class:`OneStepRKS` (``func(mol, xc, grids)`` -> 7 elements).
    xc : str
        Exchange-correlation functional (applied on the large basis).
    small_basis_solver : callable, optional
        ``func(mol_small, xc) -> C_S`` returning the small-basis MO coefficient
        matrix (N_S x N_S). If ``None`` (default), a standard converged
        small-basis RKS SCF at the same ``xc`` is used, which is cheap because
        N_S is small.
    """

    def __init__(self, mol, mol_small, eval_density_func, xc='LDA,VWN',
                 small_basis_solver=None):
        super().__init__(mol, eval_density_func, xc=xc)
        self.mol_small = mol_small
        self.small_basis_solver = small_basis_solver

        # Cross-basis projection matrix V (N_L x N_S), built lazily.
        self._V = None
        # Cache for reporting/benchmarking the achieved subspace reduction.
        self._subspace_dim = None
        self._full_dim = None

    def _solve_small_basis(self):
        """
        Solve the small basis problem F_S C_S = S_S C_S E_S and return the
        full MO coefficient matrix C_S (N_S x N_S) as a cupy array.
        """
        if self.small_basis_solver is not None:
            C_S = self.small_basis_solver(self.mol_small, self.xc)
            return _as_cupy(C_S)

        mf_s = rks.RKS(self.mol_small, xc=self.xc)
        mf_s.verbose = 0
        mf_s.conv_tol = 1.0E-10
        mf_s.kernel()
        if not mf_s.converged:
            raise RuntimeError(
                "Small-basis SCF did not converge; the dual-basis subspace "
                "would be unreliable.")
        return _as_cupy(mf_s.mo_coeff)

    def _build_projection(self):
        """
        Build the cross-basis projection matrix V = S_L^{-1} S_LS C_S.

        S_LS = <phi^L_mu | phi^S_nu> is the cross-basis overlap (N_L x N_S),
        obtained from the analytic one-electron overlap integrals between the
        large and small basis molecules.
        """
        C_S = self._solve_small_basis()

        s_cross = gto.intor_cross('int1e_ovlp', self.mol, self.mol_small)
        S_LS = _as_cupy(np.asarray(s_cross))
        S_L = _as_cupy(self.get_ovlp())

        nocc = int(self.mol.nelectron) // 2
        n_small = C_S.shape[1]
        if n_small < nocc:
            raise ValueError(
                f"Small basis provides only {n_small} molecular orbitals, "
                f"which is fewer than the {nocc} occupied orbitals required. "
                "Use a larger 'small' basis.")

        V = cp.linalg.solve(S_L, S_LS @ C_S)

        self._V = V
        self._subspace_dim = int(n_small)
        self._full_dim = int(S_L.shape[0])
        return V

    def eig(self, h, s, overwrite=False, x=None):
        if self._V is None:
            self._build_projection()

        V = self._V
        F_L = _as_cupy(h)
        S_L = _as_cupy(s)

        # Project into the small-basis MO subspace.
        F_sub = V.T @ F_L @ V
        S_sub = V.T @ S_L @ V

        # Symmetrize to suppress round-off asymmetry before the generalized eig.
        F_sub = 0.5 * (F_sub + F_sub.T)
        S_sub = 0.5 * (S_sub + S_sub.T)

        e_sub, U = eigh(F_sub, S_sub)

        # Lift the subspace eigenvectors back to the full large basis.
        mo_coeff = V @ U
        return e_sub, mo_coeff


class SingleFragmentEmbedding_ML_DualBasis(SingleFragmentEmbedding_ML):
    """
    Single-fragment CAS-like DFT embedding driven by a dual-basis one-step RKS.

    This is a thin convenience wrapper around :class:`SingleFragmentEmbedding_ML`.
    The only requirement is that ``mf_outer`` be a :class:`DualBasisOneStepRKS`
    instance, so that the global low-level step avoids the full large-basis
    diagonalization. The high-level inner solver ``mf_inner`` is unchanged and
    still operates on the large basis embedded cluster.
    """

    def __init__(self, mf_outer, mf_inner, fragment, threshold=1e-5, verbose=None):
        if not isinstance(mf_outer, DualBasisOneStepRKS):
            raise TypeError(
                "mf_outer must be a DualBasisOneStepRKS for the dual-basis "
                "accelerated embedding.")
        super().__init__(mf_outer, mf_inner, fragment,
                         threshold=threshold, verbose=verbose)
