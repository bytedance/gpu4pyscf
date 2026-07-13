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


import copy
import numpy as np
import cupy as cp
from pyscf import gto
from pyscf import lib
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import tag_array, eigh
import pyscf.ao2mo


def _as_cupy(x):
    if isinstance(x, cp.ndarray):
        return x
    return cp.asarray(x)


def _symmetrize(m):
    return 0.5 * (m + m.T)


def get_fragment_ao_indices(mol, frag_atoms):
    aoslice = mol.aoslice_by_atom()
    indices = []
    for ia in frag_atoms:
        ia = int(ia)
        if ia < 0 or ia >= mol.natm:
            raise ValueError(f"Atom index {ia} is out of range [0, {mol.natm}).")
        p0, p1 = int(aoslice[ia, 2]), int(aoslice[ia, 3])
        indices.extend(range(p0, p1))
    indices = cp.asarray(sorted(indices), dtype=cp.int32)
    if indices.size == 0:
        raise ValueError("Fragment is empty: no atomic orbitals were selected.")
    return indices


def _orthogonalize(vecs, S, tol=1e-10):
    """
    Orthonormalize a set of vectors in the metric S (symmetric positive definite).
    Returns orthonormal vectors C such that C^T S C = I.
    Handles linearly dependent vectors by discarding near-zero eigenvalues.
    """
    if vecs.size == 0 or vecs.shape[1] == 0:
        n = S.shape[0]
        return cp.zeros((n, 0))
    n_vecs = vecs.shape[1]
    M = vecs.T @ S @ vecs
    M = _symmetrize(M)
    try:
        eigvals, eigvecs = eigh(M)
    except cp.linalg.LinAlgError:
        eigvals, eigvecs = eigh(M + 1e-14 * cp.eye(n_vecs))
    keep = eigvals > tol
    if not cp.any(keep):
        n = S.shape[0]
        return cp.zeros((n, 0))
    inv_sqrt = 1.0 / cp.sqrt(eigvals[keep])
    C = vecs @ eigvecs[:, keep] * inv_sqrt[None, :]
    return C


def density_matrix_decompose(C_occ, S_ao, frag_idx, env_idx, threshold=1e-2,
                             D_ao=None):
    """
    Decompose the occupied space by diagonalizing the non-orthogonal AO
    fragment projector in the occupied MO basis.

    Parameters
    ----------
    C_occ : (nao, nocc) ndarray
        Occupied molecular orbital coefficients.
    S_ao : (nao, nao) ndarray
        AO overlap matrix.
    frag_idx : (n_frag,) int ndarray
        AO indices belonging to the fragment.
    env_idx : (n_env,) int ndarray
        AO indices belonging to the environment.
    threshold : float
        Electron-number tolerance for initial fragment/bath/core labels.  The
        labels are diagnostic only; nonzero occupied directions are kept in the
        embedding space to preserve the HF-in-HF fixed point.
    D_ao : (nao, nao) ndarray, optional
        Spin-summed AO density matrix. If omitted, it is reconstructed from
        the occupied orbitals for backward compatibility.

    Returns
    -------
    frag_orb : (n_frag, n_f) ndarray
        Near-pure fragment occupied orbitals in fragment AO basis.
    bath_orb : (nao, n_b) ndarray
        Physical bath orbitals selected by the threshold labels in the full AO
        basis (columns), C_bath^T S C_bath = I.
    core_orb : (nao, n_c) ndarray
        Frozen core orbitals in the full AO basis (columns), C_core^T S C_core
        = I.
    info : dict
        Metadata including eigenvalues, electron counts, B matrix, dm_core, etc.
    """
    C_occ = _as_cupy(C_occ)
    S_ao = _as_cupy(S_ao)
    frag_idx = _as_cupy(frag_idx)
    env_idx = _as_cupy(env_idx)
    if D_ao is None:
        D_ao = 2.0 * (C_occ @ C_occ.T)
    else:
        D_ao = _as_cupy(D_ao)

    n_frag = int(frag_idx.size)
    nao = S_ao.shape[0]

    if n_frag == 0:
        raise ValueError("Fragment has no AOs.")

    S_A = S_ao[cp.ix_(frag_idx, frag_idx)]
    S_A = _symmetrize(S_A)
    S_full_A = S_ao[:, frag_idx]
    S_A_full = S_ao[frag_idx, :]
    S_A_inv = cp.linalg.pinv(S_A)

    # Occupied-space fragment projector in a NON-ORTHOGONALIZED AO basis:
    #   P_A = A (A^T S A)^-1 A^T S
    #   Q_ij = <psi_i|P_A|psi_j>
    # Its eigenvalues lambda are spatial occupations in [0, 1].
    Q_occ = C_occ.T @ S_full_A @ S_A_inv @ S_A_full @ C_occ
    Q_occ = _symmetrize(Q_occ)
    try:
        lambda_vals, occ_rot = eigh(Q_occ)
    except cp.linalg.LinAlgError:
        lambda_vals, occ_rot = eigh(Q_occ + 1e-14 * cp.eye(Q_occ.shape[0]))
    lambda_vals = cp.clip(lambda_vals, 0.0, 1.0)
    idx_sort = cp.argsort(lambda_vals)[::-1]
    lambda_vals = lambda_vals[idx_sort]
    occ_rot = occ_rot[:, idx_sort]
    eigvals = 2.0 * lambda_vals

    n_A = eigvals.size
    n_frag_electrons_theory = float(cp.sum(eigvals))

    # Select pure fragment/core guesses by cumulative electron-number error:
    # fragment leakage is sum(2 - eigval), core leakage is sum(eigval).
    elec_tol = float(threshold)
    frag_leak = cp.cumsum(2.0 - eigvals)
    n_frag_guess = int(cp.count_nonzero(frag_leak <= elec_tol))

    core_leak = cp.cumsum(eigvals[::-1])
    n_core_guess = int(cp.count_nonzero(core_leak <= elec_tol))
    if n_frag_guess + n_core_guess > n_A:
        n_core_guess = max(0, n_A - n_frag_guess)

    frag_guess_idx = list(range(n_frag_guess))
    core_guess_idx = list(range(n_A - n_core_guess, n_A)) if n_core_guess else []
    bath_guess_idx = list(range(n_frag_guess, n_A - n_core_guess))
    complement_guess_idx = frag_guess_idx

    cum_e_frag = float(cp.sum(eigvals[frag_guess_idx])) if frag_guess_idx else 0.0
    cum_e_core = float(cp.sum(eigvals[core_guess_idx])) if core_guess_idx else 0.0

    n_frag_sel = n_frag

    C_bar = C_occ @ occ_rot

    # The environment partner of an occupied orbital in a non-orthogonal AO
    # basis is not the raw env AO block.  It is the S-orthogonal complement to
    # the fragment subspace, (I - P_A) C, where P_A = A S_AA^-1 A^T S.
    frag_projected = cp.zeros_like(C_bar)
    frag_projected[frag_idx, :] = S_A_inv @ S_A_full @ C_bar
    env_complement = C_bar - frag_projected

    bath_orb = cp.zeros((nao, 0))
    if len(bath_guess_idx) > 0:
        bath_raw = env_complement[:, cp.asarray(bath_guess_idx, dtype=cp.int32)]
        bath_orb = _orthogonalize(bath_raw, S_ao)
    n_bath = bath_orb.shape[1] if bath_orb.size else 0

    # Pure-fragment guesses can still carry small S-orthogonal tails.  Keep
    # these tails as complement orbitals, but do not count them as bath.
    complement_orb = cp.zeros((nao, 0))
    if len(complement_guess_idx) > 0:
        complement_raw = env_complement[:, cp.asarray(complement_guess_idx, dtype=cp.int32)]
        if n_bath > 0:
            complement_raw = complement_raw - bath_orb @ (bath_orb.T @ S_ao @ complement_raw)
        complement_orb = _orthogonalize(complement_raw, S_ao)
    n_complement = complement_orb.shape[1] if complement_orb.size else 0

    if n_bath > 0 and n_complement > 0:
        embedded_env_orb = cp.concatenate((bath_orb, complement_orb), axis=1)
    elif n_bath > 0:
        embedded_env_orb = bath_orb
    else:
        embedded_env_orb = complement_orb
    n_emb_env = embedded_env_orb.shape[1] if embedded_env_orb.size else 0

    core_orb = cp.zeros((nao, 0))
    if len(core_guess_idx) > 0:
        core_raw = C_bar[:, cp.asarray(core_guess_idx, dtype=cp.int32)]
        core_orb = _orthogonalize(core_raw, S_ao)
    n_core = core_orb.shape[1] if core_orb.size else 0
    n_core_electrons = 2 * n_core

    V_frag = cp.zeros((n_frag, 0))
    if len(frag_guess_idx) > 0:
        V_frag = _orthogonalize(
            frag_projected[cp.ix_(frag_idx, cp.asarray(frag_guess_idx, dtype=cp.int32))],
            S_A
        )

    # Build embedding basis B strictly in AO representation
    n_emb = n_frag + n_emb_env
    B = cp.zeros((nao, n_emb), dtype=float)
    if n_frag > 0:
        B[cp.ix_(frag_idx, cp.arange(n_frag))] = cp.eye(n_frag)
    if n_emb_env > 0:
        B[:, cp.arange(n_emb_env) + n_frag] = embedded_env_orb

    if n_emb > 0:
        S_emb = _symmetrize(B.T @ S_ao @ B)
        S_emb_inv = cp.linalg.pinv(S_emb)
        dm_emb_full = S_emb_inv @ B.T @ S_ao @ D_ao @ S_ao @ B @ S_emb_inv
    else:
        dm_emb_full = cp.zeros((0, 0))

    dm_core_full = build_core_dm(env_idx, core_orb, nao, S_ao)
    dm_core_full = _symmetrize(dm_core_full)
    
    info = {
        'n_core_electrons': n_core_electrons,
        'eigenvalues': eigvals,
        'n_frag_orbitals': n_frag_sel,
        'n_pure_fragment_nos': len(frag_guess_idx),
        'n_zero_fragment_nos': len(core_guess_idx),
        'n_bath_orbitals': n_bath,
        'n_physical_bath_orbitals': len(bath_guess_idx),
        'n_complement_orbitals': n_complement,
        'n_embedded_env_orbitals': n_emb_env,
        'n_core_orbitals': n_core,
        'n_frag_electrons_theory': n_frag_electrons_theory,
        'cum_e_core': cum_e_core,
        'cum_e_frag': cum_e_frag,
        'B': B,
        'dm_core': dm_core_full,
        'dm_emb_init': dm_emb_full if n_emb > 0 else cp.zeros((0, 0)),
        'S_emb': S_emb if n_emb > 0 else cp.zeros((0, 0)),
    }
    return V_frag, embedded_env_orb, core_orb, info


def build_core_dm(env_idx, core_orb, nao, S_ao):
    """Build frozen-core density matrix from core orbitals."""
    env_idx = _as_cupy(env_idx)
    if core_orb.size == 0 or core_orb.shape[1] == 0:
        return cp.zeros((nao, nao), dtype=float)
    if core_orb.shape[0] == nao:
        C_core = core_orb
    else:
        C_core = cp.zeros((nao, core_orb.shape[1]), dtype=float)
        C_core[env_idx, :] = core_orb
    return 2.0 * (C_core @ C_core.T)


def transform_h1(h_ao, B):
    """Project a 1-electron operator from the full AO basis to the embedded basis."""
    return B.T @ h_ao @ B


def _build_embedded_mole(nemb, n_emb_electrons, spin=0, verbose=0, max_memory=4000):
    if n_emb_electrons < 0 or n_emb_electrons > 2 * nemb:
        raise ValueError(f"Invalid embedded electron count: {n_emb_electrons}")

    mol = gto.Mole()
    mol.verbose = verbose
    mol.max_memory = max_memory
    mol.atom = []
    mol.basis = {}
    mol.unit = 'Bohr'
    mol.spin = spin
    mol.nelectron = int(n_emb_electrons)
    mol.charge = 0
    mol.build(parse_arg=False, dump_input=False)

    nemb_int = int(nemb)
    def _nao_nr(self=mol, _n=nemb_int):
        return _n

    mol.nao_nr = _nao_nr
    mol.nao = nemb_int
    return mol


def _instantiate_inner_mf(mf_template, embedded_mol):
    cls = type(mf_template)
    try:
        new_mf = cls(embedded_mol)
    except TypeError:
        new_mf = copy.copy(mf_template)
        new_mf.mol = embedded_mol
        new_mf.mo_coeff = None
        new_mf.mo_energy = None
        new_mf.mo_occ = None
        new_mf.converged = False

    for attr in ('xc', 'conv_tol', 'conv_tol_grad', 'max_cycle',
                 'level_shift', 'damp', 'diis', 'verbose'):
        if hasattr(mf_template, attr):
            try:
                setattr(new_mf, attr, getattr(mf_template, attr))
            except Exception:
                pass

    if hasattr(mf_template, 'grids') and hasattr(new_mf, 'grids'):
        for g_attr in ('level', 'prune', 'atom_grid'):
            if hasattr(mf_template.grids, g_attr):
                try:
                    setattr(new_mf.grids, g_attr, getattr(mf_template.grids, g_attr))
                except Exception:
                    pass

    return new_mf


class DMET(lib.StreamObject):
    """
    Density Matrix Embedding Theory driver with macroscopic iteration,
    using non-orthogonal AO basis density matrix diagonalization
    for orbital selection (instead of SVD in orthogonalized basis).

    Parameters
    ----------
    mf_outer : SCF object
        Low-level mean-field on the full system.
    mf_inner : SCF/post-HF object
        High-level mean-field or post-HF template applied to the embedded cluster.
    fragments : list of lists of int
        List of fragments, where each fragment is a list of atom indices.
    threshold : float
        Electron-number tolerance (default 1e-2) used to classify core/fragment/bath
        orbitals based on natural occupation numbers of the fragment density matrix.
    max_macro_iter : int
        Maximum number of macroscopic iterations for correlation potential (u).
    macro_tol : float
        Convergence tolerance for the difference in fragment 1-RDMs.
    """

    def __init__(self, mf_outer, mf_inner, fragments,
                 threshold=1e-2, max_macro_iter=20, macro_tol=1e-4,
                 verbose=None):
        if mf_outer is None or mf_inner is None:
            raise ValueError("mf_outer and mf_inner are both required.")
        if not fragments:
            raise ValueError("Provide a list of fragments to define the DMET regions.")

        if verbose is None:
            verbose = mf_outer.verbose
        else:
            verbose = int(verbose)
        self.log = logger.new_logger(mf_outer, verbose)
        self.mf_outer = mf_outer.copy()
        self.mf_inner_template = mf_inner.copy()
        self.full_mol = mf_outer.mol
        self.threshold = float(threshold)
        self.max_macro_iter = max_macro_iter
        self.macro_tol = macro_tol

        self.fragments = [list(int(a) for a in frag) for frag in fragments]
        self.nfrags = len(self.fragments)

        nao = int(self.full_mol.nao_nr())
        all_idx = cp.arange(nao, dtype=cp.int32)

        self.frag_idx = []
        self.env_idx = []
        for frag_atoms in self.fragments:
            f_idx = get_fragment_ao_indices(self.full_mol, frag_atoms)
            self.frag_idx.append(f_idx)
            env_mask = cp.ones(nao, dtype=bool)
            env_mask[f_idx] = False
            self.env_idx.append(all_idx[env_mask])

        self.frag_orb = [None] * self.nfrags
        self.bath_orb = [None] * self.nfrags
        self.core_orb = [None] * self.nfrags
        self.eig_info = [None] * self.nfrags
        self.B = [None] * self.nfrags
        self.dm_core = [None] * self.nfrags
        self.v_core_ao = [None] * self.nfrags
        self.h_emb = [None] * self.nfrags
        self.e_core = [None] * self.nfrags
        self.mf_inner = [None] * self.nfrags
        self.dm_emb_init = [None] * self.nfrags
        self.e_inner = [None] * self.nfrags
        self.e_tot = None
        # Correlation potential in AO basis
        self.u_ao = cp.zeros((nao, nao))

    def build_bath(self, ifrag, mo_coeff, mo_occ, D_ao=None, S_ao=None):
        """
        Run density-matrix-based decomposition for a specific fragment.
        """
        if D_ao is None:
            D_ao = _as_cupy(self.mf_outer.make_rdm1())
        if S_ao is None:
            S_ao = _as_cupy(self.mf_outer.get_ovlp())

        mo_coeff = _as_cupy(mo_coeff)
        mo_occ = _as_cupy(mo_occ)
        C_occ = mo_coeff[:, mo_occ > 0]

        frag_orb, bath_orb, core_orb, info = density_matrix_decompose(
            C_occ, S_ao, self.frag_idx[ifrag], self.env_idx[ifrag],
            self.threshold, D_ao=D_ao)

        nao = S_ao.shape[0]
        n_frag_sel = info['n_frag_orbitals']
        n_bath = info['n_bath_orbitals']
        n_emb_env = info['n_embedded_env_orbitals']
        n_core = info['n_core_orbitals']

        B = info['B']
        dm_core_ao = info['dm_core']
        dm_emb_init = info['dm_emb_init']

        self.frag_orb[ifrag] = frag_orb
        self.bath_orb[ifrag] = bath_orb
        self.core_orb[ifrag] = core_orb
        self.eig_info[ifrag] = info
        self.B[ifrag] = B
        self.dm_core[ifrag] = dm_core_ao
        self.dm_emb_init[ifrag] = dm_emb_init

        n_emb = B.shape[1]
        if n_emb > 0:
            s_emb = info['S_emb']
            n_emb_electrons = float(cp.trace(dm_emb_init @ s_emb))
        else:
            n_emb_electrons = 0.0
        n_core_e = float(cp.trace(dm_core_ao @ S_ao))

        self.log.info(f"Fragment {ifrag} density matrix diagonalization eigenvalues:")
        self.log.info(f"    {info['eigenvalues']}")

        self.log.info(f"Fragment {ifrag} embedding basis partition:")
        self.log.info(f"    Number of Fragment AO Orbitals      : {n_frag_sel}")
        self.log.info(f"    Number of Pure Fragment Guesses     : {info['n_pure_fragment_nos']}")
        self.log.info(f"    Number of Bath Guesses              : {info['n_physical_bath_orbitals']}")
        self.log.info(f"    Number of Core Guesses              : {info['n_zero_fragment_nos']}")
        self.log.info(f"    Number of Complement Orbitals       : {info['n_complement_orbitals']}")
        self.log.info(f"    Number of Bath Orbitals             : {n_bath}")
        self.log.info(f"    Number of Embedded Env Orbitals     : {n_emb_env}")
        self.log.info(f"    Number of Frozen Core Orbitals      : {n_core}")
        self.log.info(f"    Embedded electrons         : {n_emb_electrons:.4f}")
        self.log.info(f"    Core electrons             : {n_core_e:.4f}")
        self.log.info(f"    Total Embedded Space       : {n_emb} / {nao} (full AO)")

        return self

    def build_embedded_hamiltonian(self, ifrag, hcore_orig, S_ao=None):
        """
        Construct h^A in the embedded basis A.
        Uses bare hcore_orig (without the correlation potential 'u').
        """
        mol = self.full_mol
        h_ao = _as_cupy(hcore_orig)
        if S_ao is None:
            S_ao = _as_cupy(self.mf_outer.get_ovlp())

        if self.eig_info[ifrag]['n_core_orbitals'] > 0:
            v_core_ao = _as_cupy(self.mf_outer.get_veff(mol, self.dm_core[ifrag]))
        else:
            v_core_ao = cp.zeros_like(h_ao)

        self.v_core_ao[ifrag] = v_core_ao

        h_emb = transform_h1(h_ao + v_core_ao, self.B[ifrag])

        if self.eig_info[ifrag]['n_core_orbitals'] > 0:
            e_core = (cp.einsum('ij,ji->', self.dm_core[ifrag], h_ao)
                      + 0.5 * cp.einsum('ij,ji->', self.dm_core[ifrag], v_core_ao))
        else:
            e_core = 0.0

        self.h_emb[ifrag] = h_emb
        self.e_core[ifrag] = float(e_core)
        return self

    def _build_inner_mf(self, ifrag, dm_full_ao):
        nemb = self.B[ifrag].shape[1]
        n_total_electrons = float(self.full_mol.nelectron)
        s_ao = _as_cupy(self.mf_outer.get_ovlp())

        n_total_electrons = int(self.full_mol.nelectron)
        n_core_e = int(self.eig_info[ifrag]['n_core_electrons'])
        
        n_emb_electrons = max(0, min(n_total_electrons - n_core_e, 2 * nemb))

        emb_mol = _build_embedded_mole(
            nemb=nemb,
            n_emb_electrons=n_emb_electrons,
            spin=int(getattr(self.full_mol, 'spin', 0)),
            verbose=0,
            max_memory=int(getattr(self.full_mol, 'max_memory', 4000)),
        )

        mf_inner = _instantiate_inner_mf(self.mf_inner_template, emb_mol)

        h_emb = self.h_emb[ifrag]
        
        # Provide the actual non-orthogonal metric to the inner solver
        s_emb = _symmetrize(self.B[ifrag].T @ s_ao @ self.B[ifrag])
        ovlp = s_emb

        e_nuc = float(self.full_mol.energy_nuc())
        mf_inner.get_hcore = lambda *args, **kwargs: h_emb
        mf_inner.get_ovlp = lambda *args, **kwargs: ovlp
        mf_inner.energy_nuc = lambda *args, **kwargs: e_nuc + self.e_core[ifrag]

        B_mat = self.B[ifrag]
        v_core_ao = self.v_core_ao[ifrag]

        def _get_veff(mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
            if dm is None:
                dm = mf_inner.make_rdm1()
            dm_cp = _as_cupy(dm)

            dm_ao = B_mat @ dm_cp @ B_mat.T
            dm_full_ao_inner = self.dm_core[ifrag] + dm_ao

            v_eff_full = self.mf_inner_template.get_veff(self.full_mol, dm_full_ao_inner, hermi=hermi)
            v_eff_active = _as_cupy(v_eff_full) - v_core_ao

            if dm_cp.ndim == 2:
                v_eff_emb = B_mat.T @ v_eff_active @ B_mat
            else:
                v_eff_emb = cp.einsum('pi,xpq,qj->xij', B_mat, v_eff_active, B_mat)

            ecoul = getattr(v_eff_full, 'ecoul', 0.0)
            exc = getattr(v_eff_full, 'exc', 0.0)
            vj_full = getattr(v_eff_full, 'vj', None)
            if vj_full is not None:
                vj_emb = B_mat.T @ _as_cupy(vj_full) @ B_mat
            else:
                vj_emb = cp.zeros_like(v_eff_emb)

            vk_full = getattr(v_eff_full, 'vk', None)
            if vk_full is not None:
                vk_emb = B_mat.T @ _as_cupy(vk_full) @ B_mat
            else:
                vk_emb = cp.zeros_like(v_eff_emb)

            v_eff_emb = tag_array(v_eff_emb, ecoul=ecoul, exc=exc, vj=vj_emb, vk=vk_emb)
            return v_eff_emb

        mf_inner.get_veff = _get_veff

        dm_emb_init = self.dm_emb_init[ifrag]
        if dm_emb_init is None or dm_emb_init.size == 0:
            sB = s_ao @ self.B[ifrag]
            S_emb_inv = cp.linalg.pinv(s_emb)
            dm_emb_init = S_emb_inv @ sB.T @ _as_cupy(dm_full_ao) @ sB @ S_emb_inv

        s_emb = _symmetrize(self.B[ifrag].T @ s_ao @ self.B[ifrag])
        trace = float(cp.trace(dm_emb_init @ s_emb))
        if trace > 0.5 and n_emb_electrons > 0 and abs(trace - n_emb_electrons) > 1e-8:
            dm_emb_init = dm_emb_init * (n_emb_electrons / trace)
        self.dm_emb_init[ifrag] = dm_emb_init

        self.mf_inner[ifrag] = mf_inner
        return mf_inner

    def solve_embedded(self, ifrag):
        e_inner = self.mf_inner[ifrag].kernel(dm0=self.dm_emb_init[ifrag])
        if isinstance(e_inner, tuple):
            e_inner = float(self.mf_inner[ifrag].e_tot)
        else:
            e_inner = float(e_inner)
        self.e_inner[ifrag] = e_inner
        return e_inner

    def kernel(self):
        orig_outer_get_hcore = self.mf_outer.get_hcore
        hcore_orig = _as_cupy(self.mf_outer.get_hcore())
        s_ao = _as_cupy(self.mf_outer.get_ovlp())

        for macro_iter in range(self.max_macro_iter):
            self.log.info(f"Macro Iter {macro_iter}")

            u_ao = self.u_ao
            self.mf_outer.get_hcore = lambda *args, **kwargs: cp.asnumpy(hcore_orig + u_ao)
            self.mf_outer.mo_coeff = None
            self.mf_outer.kernel()

            mo_coeff = _as_cupy(self.mf_outer.mo_coeff)
            mo_occ = _as_cupy(self.mf_outer.mo_occ)
            dm_full_ao = _as_cupy(self.mf_outer.make_rdm1())

            e_tot = 0.0
            dm_inners = []

            for ifrag in range(self.nfrags):
                self.build_bath(ifrag, mo_coeff, mo_occ, D_ao=dm_full_ao, S_ao=s_ao)
                self.build_embedded_hamiltonian(ifrag, hcore_orig, S_ao=s_ao)
                mf_inner = self._build_inner_mf(ifrag, dm_full_ao)
                self.solve_embedded(ifrag)
                if not self.mf_inner[ifrag].converged:
                    raise RuntimeError(
                        f"Embedded high-level SCF did not converge for fragment {ifrag}; "
                        "do not use this density for delta energy."
                    )

                dm_emb = _as_cupy(mf_inner.make_rdm1())

                B = self.B[ifrag]
                dm_inner_active_ao = B @ dm_emb @ B.T
                dm_inner_full_ao = self.dm_core[ifrag] + dm_inner_active_ao
                dm_inners.append(dm_inner_full_ao)

                is_mean_field = hasattr(self.mf_inner_template, 'get_veff')
                if not is_mean_field:
                    raise NotImplementedError("Non-mean-field solver not implemented, needs thorough testing...")
                else:
                    self.log.info("using mean-field solver")
                    v_eff_full = _as_cupy(
                        self.mf_inner_template.get_veff(self.full_mol, dm_inner_full_ao)
                    )
                    h_eff_ao = hcore_orig + 0.5 * v_eff_full
                    idx = self.frag_idx[ifrag]
                    idx_mesh = cp.ix_(idx, cp.arange(h_eff_ao.shape[0]))
                    idx_mesh_t = cp.ix_(cp.arange(h_eff_ao.shape[0]), idx)
                    # Partition the mean-field energy in the original AO
                    # representation.  Row slicing in the embedded basis is
                    # only valid for an orthonormal fragment/bath basis.
                    e_frag_elec = 0.5 * (
                        cp.einsum('ij,ji->', h_eff_ao[idx_mesh], dm_inner_full_ao[idx_mesh_t])
                        + cp.einsum('ij,ji->', h_eff_ao[idx_mesh_t], dm_inner_full_ao[idx_mesh])
                    )

                e_frag_nuc = 0.0
                coords = self.full_mol.atom_coords()
                charges = self.full_mol.atom_charges()
                frag_atoms = self.fragments[ifrag]
                for i in frag_atoms:
                    for j in range(self.full_mol.natm):
                        if i == j:
                            continue
                        r = np.linalg.norm(coords[i] - coords[j])
                        e_frag_nuc += 0.5 * charges[i] * charges[j] / r

                self.log.info(f"Fragment {ifrag} Electronic Energy: {float(e_frag_elec):.8f} | Nuclear Energy: {e_frag_nuc:.8f}")
                e_tot += float(e_frag_elec) + e_frag_nuc

            error = 0.0
            for ifrag in range(self.nfrags):
                idx = self.frag_idx[ifrag]
                idx_mesh = cp.ix_(idx, idx)

                dm_high = dm_inners[ifrag]
                dm_low = dm_full_ao

                # Compare the fragment-projected density operator in the same
                # non-orthogonal metric used to build the bath.
                diff_cov = s_ao[idx, :] @ (dm_high - dm_low) @ s_ao[:, idx]
                error += float(cp.linalg.norm(diff_cov))

                self.u_ao[idx_mesh] -= 0.5 * diff_cov
                self.u_ao = _symmetrize(self.u_ao)

            self.log.note(f"Macro Iter {macro_iter + 1:2d} | E_DMET = {e_tot:.8f} | max(dD) = {error:.6e}")
            self.e_tot = e_tot
            if error < self.macro_tol:
                self.log.note("DMET macroscopic iterations converged.")
                break

        self.mf_outer.get_hcore = orig_outer_get_hcore
        self.mf_outer.mo_coeff = None
        self.mf_outer.mo_energy = None
        self.mf_outer.mo_occ = None

        return self.e_tot

    def __call__(self):
        return self.kernel()
