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

import numpy as np
import cupy as cp


def to_numpy(x):
    """Return a host (numpy) copy of *x* regardless of the backend."""
    if x is None:
        return None
    if isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return np.asarray(x)


def to_float(x):
    """Coerce a 0-d array / python scalar to a plain python float."""
    return float(to_numpy(x))


def as_backend(x, like):
    """Cast *x* to the same backend (cupy/numpy) as *like*."""
    if isinstance(like, cp.ndarray):
        return cp.asarray(x)
    return np.asarray(to_numpy(x))


def project_dm_emb_to_ao(dm_emb, B):
    B = as_backend(B, dm_emb)
    nao, nemb = B.shape
    assert dm_emb.shape == (nemb, nemb), (
        f"dm_emb must be (N_emb, N_emb)=({nemb},{nemb}), got {tuple(dm_emb.shape)}")
    dm_ao = B @ dm_emb @ B.T
    assert dm_ao.shape == (nao, nao), (
        f"projected dm must be (N_AO, N_AO)=({nao},{nao}), got {tuple(dm_ao.shape)}")
    return dm_ao


def project_mo_emb_to_ao(mo_emb, B):
    B = as_backend(B, mo_emb)
    nao, nemb = B.shape
    assert mo_emb.shape[0] == nemb, (
        f"mo_emb leading dim must be N_emb={nemb}, got {mo_emb.shape[0]}")
    mo_ao = B @ mo_emb
    assert mo_ao.shape[0] == nao
    return mo_ao


def project_dm_ao_to_emb(dm_ao, B, s_ao):
    B = as_backend(B, dm_ao)
    s_ao = as_backend(s_ao, dm_ao)
    sB = s_ao @ B
    dm_emb = sB.T @ dm_ao @ sB
    nemb = B.shape[1]
    assert dm_emb.shape == (nemb, nemb)
    return dm_emb


def assert_full_ao(dm, nao, name="density matrix"):
    """Hard guard used before any global-property evaluation."""
    shape = tuple(dm.shape)
    assert shape == (nao, nao), (
        f"{name} must be the full AO space (N_AO, N_AO)=({nao},{nao}); got "
        f"{shape}. Did you forget to project the embedding-basis quantity with B?")
    return True


def hybrid_exchange_coeff(mf):
    ni = getattr(mf, "_numint", None)
    xc = getattr(mf, "xc", None)
    if ni is not None and xc is not None:
        spin = getattr(getattr(mf, "mol", None), "spin", 0)
        return float(ni.hybrid_coeff(xc, spin=spin))
    # Plain Hartree-Fock object: 100% exact exchange, no DFT XC.
    return 1.0


def semilocal_exc(mf, dm_ao):
    ni = getattr(mf, "_numint", None)
    if ni is None:
        return 0.0
    grids = mf.grids
    if grids.coords is None:
        grids.build()
    dm = as_backend(dm_ao, cp.zeros(1))
    # nr_rks returns (nelec, exc, vxc); we only need the energy.
    _, exc, _ = ni.nr_rks(mf.mol, grids, mf.xc, dm)
    return to_float(exc)


def exact_exchange_energy(mf, dm_ao):
    K = get_k_matrix(mf, dm_ao)
    dm = as_backend(dm_ao, K)
    return -0.25 * to_float((dm * K).sum())


def get_k_matrix(mf, dm_ao):
    mol = mf.mol
    try:
        K = mf.get_k(mol, as_backend(dm_ao, cp.zeros(1)))
    except TypeError:
        K = mf.get_k(mol, dm_ao)
    return K


def delta_exc_shift_active(mf_low, mf_high, dm_act_ref_ao):
    r"""Local functional-upgrade shift on the active reference density.

    .. math::

        \Delta E_{xc\_shift}^{act} =
            \big(E_{xc}^{high}[\rho_{act}^{ref}] - E_{xc}^{low}[\rho_{act}^{ref}]\big)
            - \tfrac12 (c_x^{high} - c_x^{low})\,
              \mathrm{Tr}\!\big[D_{act}^{ref} K(D_{act}^{ref})\big]
    """
    exc_high = semilocal_exc(mf_high, dm_act_ref_ao)
    exc_low = semilocal_exc(mf_low, dm_act_ref_ao)
    d_semilocal = exc_high - exc_low

    cx_high = hybrid_exchange_coeff(mf_high)
    cx_low = hybrid_exchange_coeff(mf_low)
    d_cx = cx_high - cx_low

    if abs(d_cx) < 1e-15:
        d_exchange = 0.0
    else:
        # exact_exchange_energy already carries the -1/4 Tr[DK] factor.
        d_exchange = d_cx * exact_exchange_energy(mf_high, dm_act_ref_ao)

    return float(d_semilocal + d_exchange)


def shifted_reference_energy(mf_low, mf_high, emb, ifrag=0):
    r"""Dynamic reference energy that absorbs the environment-XC mismatch.

    .. math::

        E_{ref}^{shifted} = E_{global}^{low}[D_{conv}^{low}]
                          + \Delta E_{xc\_shift}^{act}[D_{act}^{ref}]
    """
    if not getattr(mf_low, "converged", False):
        mf_low.kernel()
    e_global_low = float(mf_low.e_tot)

    dm_low_ao = mf_low.make_rdm1()
    s_ao = mf_low.get_ovlp()
    B = emb.B[ifrag]

    dm_act_ref = project_dm_ao_to_emb(dm_low_ao, B, s_ao)   # B^T S D S B (N_emb)
    dm_act_ref_ao = project_dm_emb_to_ao(dm_act_ref, B)     # back to N_AO

    delta = delta_exc_shift_active(mf_low, mf_high, dm_act_ref_ao)
    return {
        "e_global_low": e_global_low,
        "delta_xc_shift": float(delta),
        "e_ref_shifted": float(e_global_low + delta),
    }


def high_level_nonscf_energy(mf_low, mf_high):
    r"""High-level total energy evaluated *non-self-consistently* on the
    low-level converged density.

    .. math::  E_{high}^{nonSCF} = E_{high}[D_{conv}^{low}]

    This is the analytic value that :func:`shifted_reference_energy` must equal
    when the embedding active space spans the entire molecule (full-dimension
    exactness test).
    """
    if not getattr(mf_low, "converged", False):
        mf_low.kernel()
    dm_low = mf_low.make_rdm1()
    h1e = mf_high.get_hcore()
    vhf = mf_high.get_veff(mf_high.mol, dm_low)
    e_elec = mf_high.energy_elec(dm_low, h1e, vhf)[0]
    return float(e_elec + mf_high.energy_nuc())


# ---------------------------------------------------------------------------
# Local orbital energies (HOMO / LUMO of the embedded cluster)
# ---------------------------------------------------------------------------
def core_homo_lumo(mf_inner):
    """HOMO / LUMO (and gap) of the embedded cluster solver, in Hartree.
    """
    mo_energy = to_numpy(mf_inner.mo_energy).ravel()
    mo_occ = to_numpy(mf_inner.mo_occ).ravel()
    occ = np.where(mo_occ > 1e-8)[0]
    vir = np.where(mo_occ <= 1e-8)[0]
    if occ.size == 0 or vir.size == 0:
        return {"homo": None, "lumo": None, "gap": None}
    homo = float(mo_energy[occ].max())
    lumo = float(mo_energy[vir].min())
    return {"homo": homo, "lumo": lumo, "gap": float(lumo - homo)}


# ---------------------------------------------------------------------------
# Local excited states: TDA via explicit assembly of the A matrix
# ---------------------------------------------------------------------------
def build_tda_amatrix(mf_outer, mf_inner, mo_coeff_ao, mo_energy, mo_occ, singlet=True):
    r"""Assemble the TDA *A* matrix in the AO-projected MO basis and diagonalise.

    Instead of relying on gen_response or explicitly building the O(N^4) ERI
    tensor (which causes OOM), we batch the transition density matrices and pass 
    them to mf_outer.get_jk. The DFT XC responses for the active subspace are 
    computed block-by-block on the grid using the INNER functional's parameters.
    """
    nao = int(mf_outer.mol.nao_nr())
    mo_coeff_ao = as_backend(mo_coeff_ao, cp.zeros(1))
    assert mo_coeff_ao.shape[0] == nao, (
        f"mo_coeff must be AO-projected (leading dim N_AO={nao}); got "
        f"{mo_coeff_ao.shape[0]}. Project with B first (C_AO = B @ C_emb).")

    mo_energy_h = to_numpy(mo_energy).ravel()
    mo_occ_h = to_numpy(mo_occ).ravel()
    occ_idx = np.where(mo_occ_h > 1e-8)[0]
    vir_idx = np.where(mo_occ_h <= 1e-8)[0]
    nocc, nvir = occ_idx.size, vir_idx.size
    if nocc == 0 or nvir == 0:
        return {"excitation_energies": [], "a_matrix": np.zeros((0, 0)),
                "nocc": int(nocc), "nvir": int(nvir)}

    C_occ = mo_coeff_ao[:, occ_idx]
    C_vir = mo_coeff_ao[:, vir_idx]

    e_ia = mo_energy_h[vir_idx] - mo_energy_h[occ_idx, None]
    a = cp.diag(e_ia.ravel()).reshape(nocc, nvir, nocc, nvir)

    mol = mf_outer.mol
    from gpu4pyscf import scf

    def add_hf_(a_mat, hyb=1.0):
        n_ex = nocc * nvir
        batch_size = 128  # Safe batch size to prevent OOM
        
        # Batch over all (j, b) transition pairs
        for p0 in range(0, n_ex, batch_size):
            p1 = min(n_ex, p0 + batch_size)
            n_batch = p1 - p0
            
            dms = cp.empty((n_batch, nao, nao))
            for k in range(n_batch):
                idx = p0 + k
                j = idx // nvir
                b = idx % nvir
                # Transition density matrix P[j,b] = C_j * C_b^T
                dms[k] = cp.outer(C_occ[:, j], C_vir[:, b])
                
            # Utilize global optimized JK builder for integration
            vj, vk = mf_outer.get_jk(mol, dms, hermi=0)
            
            if singlet:
                v_resp = 2.0 * vj
            else:
                v_resp = cp.zeros_like(vj)
                
            if hyb != 0:
                v_resp -= hyb * vk
                
            # Contract V_resp with C_occ and C_vir to get A matrix slice
            tmp = cp.tensordot(v_resp, C_vir, axes=([2], [0]))  
            tmp2 = cp.tensordot(tmp, C_occ, axes=([1], [0]))    
            batch_A = tmp2.transpose(0, 2, 1)                   
            
            for k in range(n_batch):
                idx = p0 + k
                j = idx // nvir
                b = idx % nvir
                a_mat[:, :, j, b] += batch_A[k]

    if isinstance(mf_inner, scf.hf.KohnShamDFT):
        grids = mf_outer.grids  # Keep global outer grids for spatial integration
        ni = mf_inner._numint   # USE INNER functional evaluator
        xc = mf_inner.xc        # USE INNER xc name
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(xc, mol.spin)

        # 1. Add exact exchange / Coulomb components using INNER hyb
        add_hf_(a, hyb)

        if omega != 0:
            raise NotImplementedError('RSH functional is not fully implemented in this block.')

        xctype = ni._xc_type(xc)
        opt = getattr(ni, 'gdftopt', None)
        if opt is None:
            ni.build(mol, grids.coords)
            opt = ni.gdftopt
        _sorted_mol = opt._sorted_mol

        # Use the inner (active) orbitals to evaluate the background density for fxc
        mo_coeff_global = cp.asarray(mo_coeff_ao)
        mo_occ_global = cp.asarray(mo_occ)
        
        mo_coeff_global_sort = opt.sort_orbitals(mo_coeff_global, axis=[0])
        C_occ_sort = opt.sort_orbitals(C_occ, axis=[0])
        C_vir_sort = opt.sort_orbitals(C_vir, axis=[0])

        # 2. Add DFT grid-based exchange-correlation response using INNER xc
        if xctype == 'LDA':
            ao_deriv = 0
            for ao, mask, weight, coords in ni.block_loop(_sorted_mol, grids, nao, ao_deriv):
                rho = ni.eval_rho2(_sorted_mol, ao, mo_coeff_global_sort[mask], mo_occ_global, mask, xctype, with_lapl=False)
                if singlet or singlet is None:
                    fxc = ni.eval_xc_eff(xc, rho, deriv=2, xctype=xctype)[2]
                    wfxc = fxc[0,0] * weight
                else:
                    fxc = ni.eval_xc_eff(xc, cp.stack((rho, rho))*0.5, deriv=2, xctype=xctype)[2]
                    wfxc = (fxc[0,0,0,0] - fxc[1,0,0,0]) * 0.5 * weight

                rho_o = cp.einsum('pr,pi->ri', ao, C_occ_sort[mask])
                rho_v = cp.einsum('pr,pi->ri', ao, C_vir_sort[mask])
                rho_ov = cp.einsum('ri,ra->ria', rho_o, rho_v)
                w_ov = cp.einsum('ria,r->ria', rho_ov, wfxc)
                iajb = cp.einsum('ria,rjb->iajb', rho_ov, w_ov) * 2
                a += iajb

        elif xctype == 'GGA':
            ao_deriv = 1
            for ao, mask, weight, coords in ni.block_loop(_sorted_mol, grids, nao, ao_deriv):
                rho = ni.eval_rho2(_sorted_mol, ao, mo_coeff_global_sort[mask], mo_occ_global, mask, xctype, with_lapl=False)
                if singlet or singlet is None:
                    fxc = ni.eval_xc_eff(xc, rho, deriv=2, xctype=xctype)[2]
                    wfxc = fxc * weight
                else:
                    fxc = ni.eval_xc_eff(xc, cp.stack((rho, rho))*0.5, deriv=2, xctype=xctype)[2]
                    wfxc = (fxc[0,:,0,:] - fxc[1,:,0,:]) * 0.5 * weight

                rho_o = cp.einsum('xpr,pi->xri', ao, C_occ_sort[mask])
                rho_v = cp.einsum('xpr,pi->xri', ao, C_vir_sort[mask])
                rho_ov = cp.einsum('xri,ra->xria', rho_o, rho_v[0])
                rho_ov[1:4] += cp.einsum('ri,xra->xria', rho_o[0], rho_v[1:4])
                
                w_ov = cp.einsum('xyr,xria->yria', wfxc, rho_ov)
                iajb = cp.einsum('xria,xrjb->iajb', w_ov, rho_ov) * 2
                a += iajb

        elif xctype == 'MGGA':
            ao_deriv = 1
            for ao, mask, weight, coords in ni.block_loop(_sorted_mol, grids, nao, ao_deriv):
                rho = ni.eval_rho2(_sorted_mol, ao, mo_coeff_global_sort[mask], mo_occ_global, mask, xctype, with_lapl=False)
                if singlet or singlet is None:
                    fxc = ni.eval_xc_eff(xc, rho, deriv=2, xctype=xctype)[2]
                    wfxc = fxc * weight
                else:
                    fxc = ni.eval_xc_eff(xc, cp.stack((rho, rho))*0.5, deriv=2, xctype=xctype)[2]
                    wfxc = (fxc[0,:,0,:] - fxc[1,:,0,:]) * 0.5 * weight

                rho_o = cp.einsum('xpr,pi->xri', ao, C_occ_sort[mask])
                rho_v = cp.einsum('xpr,pi->xri', ao, C_vir_sort[mask])
                rho_ov = cp.einsum('xri,ra->xria', rho_o, rho_v[0])
                rho_ov[1:4] += cp.einsum('ri,xra->xria', rho_o[0], rho_v[1:4])
                tau_ov = cp.einsum('xri,xra->ria', rho_o[1:4], rho_v[1:4]) * 0.5
                rho_ov = cp.vstack([rho_ov, tau_ov[cp.newaxis]])
                
                w_ov = cp.einsum('xyr,xria->yria', wfxc, rho_ov)
                iajb = cp.einsum('xria,xrjb->iajb', w_ov, rho_ov) * 2
                a += iajb

    else:
        add_hf_(a, hyb=1.0)

    A_mat = a.reshape(nocc * nvir, nocc * nvir)
    A_mat = 0.5 * (A_mat + A_mat.T)
    A_mat = to_numpy(A_mat)

    w, v = np.linalg.eigh(A_mat)
    return {"excitation_energies": [float(x) for x in np.sort(w)],
            "eigenvectors": v,
            "a_matrix": A_mat, "nocc": int(nocc), "nvir": int(nvir)}


def embedded_tda(emb, mf_outer, ifrag=0, singlet=True, nstates=5):
    """Convenience wrapper: project the cluster MOs to AO then run :func:`build_tda_amatrix`.

    Returns the lowest ``nstates`` TDA excitation energies (Hartree and eV).
    """
    mf_inner = emb.mf_inner[ifrag]
    B = emb.B[ifrag]
    mo_coeff_ao = project_mo_emb_to_ao(mf_inner.mo_coeff, B)
    # Passed mf_inner to ensure correct functional response parameters
    res = build_tda_amatrix(mf_outer, mf_inner, mo_coeff_ao, mf_inner.mo_energy,
                            mf_inner.mo_occ, singlet=singlet)
    energies = res["excitation_energies"][:nstates]
    HARTREE2EV = 27.211386245988
    return {
        "excitation_energies_au": energies,
        "excitation_energies_ev": [e * HARTREE2EV for e in energies],
        "eigenvectors": res["eigenvectors"].tolist(),
        "nocc": res["nocc"],
        "nvir": res["nvir"],
    }


# ---------------------------------------------------------------------------
# Population analysis (Mulliken) on the AO-projected density
# ---------------------------------------------------------------------------
def mulliken_charges(mol, dm_ao, s_ao, atom_ids=None):
    r"""Mulliken atomic charges from an AO-basis density matrix.

    .. math::  q_A = Z_A - \sum_{\mu \in A} (D S)_{\mu\mu}

    Parameters
    ----------
    mol : the full molecule.
    dm_ao : (N_AO, N_AO) array  -- MUST already be in the AO basis.
    s_ao : (N_AO, N_AO) array   -- AO overlap.
    atom_ids : optional list of atom indices to report (default: all atoms).

    Returns
    -------
    dict mapping atom index -> charge (float).
    """
    nao = int(mol.nao_nr())
    assert_full_ao(dm_ao, nao, "Mulliken density matrix")
    dm = to_numpy(dm_ao)
    s = to_numpy(s_ao)
    pop = np.einsum('ij,ji->i', dm, s)       # per-AO gross population

    aoslice = mol.aoslice_by_atom()
    if atom_ids is None:
        atom_ids = list(range(mol.natm))

    charges = {}
    for ia in atom_ids:
        ia = int(ia)
        p0, p1 = int(aoslice[ia, 2]), int(aoslice[ia, 3])
        elec_pop = float(pop[p0:p1].sum())
        z = float(mol.atom_charge(ia))
        charges[ia] = z - elec_pop
    return charges


# ---------------------------------------------------------------------------
# Density-difference cube export
# ---------------------------------------------------------------------------
def density_difference_cube(mol, dm_embedding_ao, dm_global_high_ao, outfile,
                            nx=60, ny=60, nz=60):
    r"""Write a Gaussian cube of ``rho_embedding - rho_global_high``.

    Both density matrices must already be in the *full AO basis*
    (N_AO x N_AO); the embedding density therefore has to be projected with
    :func:`project_dm_emb_to_ao` (plus its frozen core) *before* calling this.

    Uses pyscf's ``cubegen`` on host (numpy) arrays.
    """
    from pyscf.tools import cubegen
    nao = int(mol.nao_nr())
    assert_full_ao(dm_embedding_ao, nao, "embedding density (cube)")
    assert_full_ao(dm_global_high_ao, nao, "global-high density (cube)")
    dm_diff = to_numpy(dm_embedding_ao) - to_numpy(dm_global_high_ao)
    cubegen.density(mol, outfile, dm_diff, nx=nx, ny=ny, nz=nz)
    return outfile


def full_ao_embedding_density(emb, ifrag=0):
    """Total embedding density in the AO basis: ``D_core + B D_emb B^T``.

    Returns an (N_AO, N_AO) array suitable for cube / Mulliken analysis.
    """
    mf_inner = emb.mf_inner[ifrag]
    B = emb.B[ifrag]
    dm_emb = mf_inner.make_rdm1()
    dm_active_ao = project_dm_emb_to_ao(dm_emb, B)
    dm_core = emb.dm_core[ifrag]
    dm_core = as_backend(dm_core, dm_active_ao)
    return dm_active_ao + dm_core

