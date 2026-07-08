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
from pyscf import lib
from gpu4pyscf.dft import rks
from gpu4pyscf.lib.cupy_helper import tag_array
from gpu4pyscf.qmmm.embedding.embedding_dmet import DMET, _as_cupy
from gpu4pyscf.qmmm.embedding.embedding_dmet_dft_try1 import SingleFragmentEmbedding


class OneStepRKS(rks.RKS):
    """
    One-step RKS class based on machine learning (ML) predicted density.

    This class bypasses traditional SCF iterations. Instead, it relies entirely
    on an external ML density evaluation function to construct the global effective
    potential and calculate the double counting energy.
    """
    def __init__(self, mol, eval_density_func, xc='LDA,VWN'):
        super().__init__(mol)
        self.xc = xc
        self.max_cycle = 1

        # eval_density_func is the external ML interface.
        # Signature: def func(mol, xc, grids)
        # Returns 7 elements: vj, vk, vxc, e_j, e_k, e_xc, int_rho_vxc
        self.eval_density_func = eval_density_func

        self._v_eff_global = None
        self._e_dc_global = None
        self._use_harris_veff = False

    def _get_harris_veff(self, mol=None):
        if mol is None:
            mol = self.mol

        if self._v_eff_global is not None:
            return self._v_eff_global

        if self.grids.coords is None:
            self.grids.build()

        vj, vk, vxc, e_j, e_k, e_xc, int_rho_vxc = self.eval_density_func(
            mol, self.xc, self.grids)

        v_eff_ao = _as_cupy(vj) + _as_cupy(vxc)
        if vk is not None:
            v_eff_ao -= _as_cupy(vk)
            e_k = float(e_k)
        else:
            e_k = 0.0

        # double counting energy
        e_dc = float(e_j) - e_k + float(int_rho_vxc) - float(e_xc)

        vk_array = _as_cupy(vk) if vk is not None else cp.zeros_like(v_eff_ao)
        v_eff_ao = tag_array(v_eff_ao, ecoul=float(e_j) - e_k, exc=float(e_xc), vj=_as_cupy(vj), vk=vk_array)

        self._v_eff_global = v_eff_ao
        self._e_dc_global = e_dc
        return self._v_eff_global

    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        # Use ML evaluation ONLY during the global SCF step.
        # For standard embedding steps, fallback to the native exact DFT evaluation.
        if getattr(self, '_use_harris_veff', False):
            return self._get_harris_veff(mol)
        return super().get_veff(mol, dm, dm_last, vhf_last, hermi)

    def kernel(self, dm0=None, **kwargs):
        if self.max_cycle != 1:
            lib.logger.warn(self, "OneStepRKS is a non-iterative method. "
                                  f"Overriding max_cycle from {self.max_cycle} to 1.")
            self.max_cycle = 1

        # Temporarily enable Harris ML potential for the global 1-step evaluation
        self._use_harris_veff = True

        # Instance-level override: forcefully bypass DensityFitMixin's get_veff
        # so it strictly hits our ML evaluator during the global step
        self.get_veff = lambda mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1: self._get_harris_veff(mol)

        try:
            e_tot = super().kernel(dm0=dm0, **kwargs)
        finally:
            self._use_harris_veff = False
            if 'get_veff' in self.__dict__:
                del self.__dict__['get_veff']

        self.converged = True
        return e_tot

    def energy_elec(self, dm=None, h1e=None, vhf=None):
        """
        E_elec = Tr[D * (h + Veff)] - E_DC
        """
        if getattr(self, '_use_harris_veff', False):
            if dm is None:
                dm = self.make_rdm1()
            if h1e is None:
                h1e = self.get_hcore()
            if vhf is None:
                vhf = self._get_harris_veff(self.mol)

            dm_cp = _as_cupy(dm)
            h1e_cp = _as_cupy(h1e)
            vhf_cp = _as_cupy(vhf)

            fock = h1e_cp + vhf_cp
            e_band = float(cp.sum(dm_cp * fock))

            # Fallback to ensure _e_dc_global is safely initialized
            if self._e_dc_global is None:
                self._get_harris_veff(self.mol)

            e_elec = e_band - self._e_dc_global
            return e_elec, self._e_dc_global
        else:
            return super().energy_elec(dm, h1e, vhf)


class SingleFragmentEmbedding_ML(SingleFragmentEmbedding):
    """
    Single-Fragment subspace variational embedding utilizing ML density
    with density-matrix-diagonalization based DMET in non-orthogonal AO basis.

    This class performs DMET bond-breaking via density matrix diagonalization,
    and evaluates the local embedded energies using a CAS-like variational approach
    without ONIOM correction, explicitly handling non-linear DFT exchange-correlation
    components.
    """
    def __init__(self, mf_outer, mf_inner, fragment, threshold=1e-2, verbose=None):
        super().__init__(mf_outer, mf_inner, fragment,
                         threshold=threshold, verbose=verbose)
        self.fragment = self.fragments[0]

    def kernel(self):

        if not self.mf_outer.converged:
            self.mf_outer.kernel()

        e_global_low = self.mf_outer.e_tot
        self.log.note(f"Global Low-Level E (ML input) = {e_global_low:.8f}")

        mo_coeff = _as_cupy(self.mf_outer.mo_coeff)
        mo_occ = _as_cupy(self.mf_outer.mo_occ)
        dm_full_ao_low = _as_cupy(self.mf_outer.make_rdm1())
        hcore_orig = _as_cupy(self.mf_outer.get_hcore())
        s_ao = _as_cupy(self.mf_outer.get_ovlp())

        ifrag = 0

        self.build_bath(ifrag, mo_coeff, mo_occ, D_ao=dm_full_ao_low, S_ao=s_ao)
        self.build_embedded_hamiltonian(ifrag, hcore_orig, S_ao=s_ao)

        self.log.info("Running high-level inner DFT in embedding space...")
        mf_inner = self._build_inner_mf(ifrag, dm_full_ao_low)

        # Patch for gpu4pyscf inner MF losing _numint
        if getattr(mf_inner, '_numint', None) is None:
            import copy
            if hasattr(self, 'mf_inner_template') and getattr(self.mf_inner_template, '_numint', None) is not None:
                mf_inner._numint = copy.copy(self.mf_inner_template._numint)
            elif getattr(self.mf_outer, '_numint', None) is not None:
                mf_inner._numint = copy.copy(self.mf_outer._numint)
            else:
                from gpu4pyscf.dft import numint
                mf_inner._numint = numint.NumInt()

        # Ensure it's correctly assigned back to the list so solve_embedded uses the patched object
        self.mf_inner[ifrag] = mf_inner

        B_mat = self.B[ifrag]
        dm_core_mat = self.dm_core[ifrag]
        h_eval_bare_mat = B_mat.T @ hcore_orig @ B_mat

        # Add the missing core 1-electron energy
        e1_core = float(cp.sum(dm_core_mat * hcore_orig))

        e_nuc_full = float(self.full_mol.energy_nuc())
        mf_inner.energy_nuc = lambda *args, **kwargs: e_nuc_full

        # Override get_veff for strictly CAS-like Hybrid XC potential evaluation
        def custom_hybrid_get_veff(mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
            if dm is None:
                dm = mf_inner.make_rdm1()
            dm_cp = _as_cupy(dm)

            dm_act_ao = B_mat @ dm_cp @ B_mat.T
            dm_full_ao = dm_core_mat + dm_act_ao

            direct_scf_bak = getattr(self.mf_outer, 'direct_scf', True)
            self.mf_outer.direct_scf = False
            v_low_full = self.mf_outer.get_veff(self.full_mol, dm_full_ao)
            v_low_act = self.mf_outer.get_veff(self.full_mol, dm_act_ao)
            self.mf_outer.direct_scf = direct_scf_bak

            direct_scf_bak_high = getattr(self.mf_inner_template, 'direct_scf', True)
            self.mf_inner_template.direct_scf = False
            v_high_act = self.mf_inner_template.get_veff(self.full_mol, dm_act_ao)
            self.mf_inner_template.direct_scf = direct_scf_bak_high

            # Hybrid effective potential construction
            v_eff_active = _as_cupy(v_low_full) + _as_cupy(v_high_act) - _as_cupy(v_low_act) - self.v_core_ao[ifrag]

            if dm_cp.ndim == 2:
                v_eff_emb = B_mat.T @ v_eff_active @ B_mat
            else:
                v_eff_emb = cp.einsum('pi,xpq,qj->xij', B_mat, v_eff_active, B_mat)

            ecoul = float(getattr(v_low_full, 'ecoul', 0.0))
            exc = float(getattr(v_low_full, 'exc', 0.0)) + float(getattr(v_high_act, 'exc', 0.0)) - float(getattr(v_low_act, 'exc', 0.0))

            return tag_array(v_eff_emb, ecoul=ecoul, exc=exc)

        mf_inner.get_veff = custom_hybrid_get_veff

        # Override energy_elec to print the true full system energy using CAS-like formulation
        def custom_energy_elec(dm=None, h1e=None, vhf=None):
            if dm is None:
                dm = mf_inner.make_rdm1()

            # Use the 'vhf' that is passed in, which has been evaluated by our custom_hybrid_get_veff!
            if vhf is None:
                vhf = mf_inner.get_veff(mf_inner.mol, dm)

            dm_cp = _as_cupy(dm)

            # e1: Active space single-electron energy + Core single-electron energy
            e1_active = float(cp.sum(dm_cp * h_eval_bare_mat))
            e1 = e1_active + e1_core

            # e2: Directly use the exact hybrid 2e energy from our custom get_veff!
            ecoul_full = float(getattr(vhf, 'ecoul', 0.0))
            exc_hybrid = float(getattr(vhf, 'exc', 0.0))
            e2 = ecoul_full + exc_hybrid

            # Update scf_summary for meaningful debugging output relative to the core
            mf_inner.scf_summary['e1'] = e1
            mf_inner.scf_summary['coul'] = ecoul_full
            mf_inner.scf_summary['exc'] = exc_hybrid

            return e1 + e2, e2

        mf_inner.energy_elec = custom_energy_elec

        self.solve_embedded(ifrag)
        if not self.mf_inner[ifrag].converged:
            raise RuntimeError(
                f"Embedded high-level SCF did not converge for fragment {ifrag}; "
                "do not use this density for final energy evaluation."
            )

        dm_emb_high = _as_cupy(mf_inner.make_rdm1())
        # The low-level embedded evaluation is entirely bypassed (No ONIOM correction)

        B = self.B[ifrag]
        dm_core = self.dm_core[ifrag]
        is_mean_field = hasattr(self.mf_inner_template, 'get_veff')

        if is_mean_field:
            h_eval_bare = B.T @ hcore_orig @ B

            # Evaluate High-Level energy using our explicitly defined Hybrid function
            e_high = self._evaluate_embedded_energy(dm_emb_high, h_eval_bare, B, dm_core)
        else:
            raise NotImplementedError("WFT evaluation is not implemented for this class.")

        # In CAS-like physics, total energy is directly the strict subspace variational minimum
        self.e_tot = float(e_high)

        self.log.note("-" * 40)
        self.log.note(f"Final Total Embedded E (CAS-DFT) : {self.e_tot:.8f}")
        self.log.note("-" * 40)

        return self.e_tot
