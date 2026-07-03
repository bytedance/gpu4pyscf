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

import cupy as cp
import numpy as np
import pyscf.ao2mo
from gpu4pyscf.lib.cupy_helper import tag_array
from gpu4pyscf.qmmm.embedding.embedding import DMET, lowdin_orth, _as_cupy


class SingleFragmentEmbedding(DMET):
    """
    Single-Fragment subspace variational embedding for DFT (CAS-like).
    
    This class performs a single-shot,
    single-fragment exact subspace energy evaluation.
    """
    
    def __init__(self, mf_outer, mf_inner, fragment, threshold=1e-5, verbose=None):
        """
        Parameters
       -------
        mf_outer : SCF object
            Low-level mean-field on the full system (e.g., PBE).
        mf_inner : SCF/DFT/post-HF object
            High-level template applied to the embedded cluster (e.g., B3LYP).
        fragment : list of int
            A single list of atom indices defining the QM region.
        threshold : float
            Eigenvalue cutoff used to classify environment orbitals.
        """
        fragments = [fragment]
        
        super().__init__(mf_outer, mf_inner, fragments,
                         threshold=threshold, max_macro_iter=1, verbose=verbose)
        
        self.fragment = self.fragments[0]
    
    def _evaluate_embedded_energy(self, dm_emb, h_eval_bare, B, dm_core):
        """
        Evaluate the CAS-DFT energy using the hybrid XC formulation:
        E_tot = Tr(D_tot * h) + 1/2 Tr(D_tot * J_tot) + E_xc^low[tot] + E_xc^high[act] - E_xc^low[act]
        """
        e_h_active = float(cp.sum(dm_emb * h_eval_bare))
        
        dm_act_ao = B @ dm_emb @ B.T
        dm_full_ao = dm_core + dm_act_ao
        
        # Low-level evaluation on full density
        direct_scf_bak = getattr(self.mf_outer, 'direct_scf', True)
        self.mf_outer.direct_scf = False
        v_low_full = self.mf_outer.get_veff(self.full_mol, dm_full_ao)
        
        # Low-level evaluation on active density
        v_low_act = self.mf_outer.get_veff(self.full_mol, dm_act_ao)
        self.mf_outer.direct_scf = direct_scf_bak
        
        # High-level evaluation on active density
        direct_scf_bak_high = getattr(self.mf_inner_template, 'direct_scf', True)
        self.mf_inner_template.direct_scf = False
        v_high_act = self.mf_inner_template.get_veff(self.full_mol, dm_act_ao)
        self.mf_inner_template.direct_scf = direct_scf_bak_high

        # Assemble the hybrid 2e energy (Coulomb is exact from full density)
        e_coul_full = float(getattr(v_low_full, 'ecoul', 0.0))
        e_xc_hybrid = float(getattr(v_low_full, 'exc', 0.0)) + float(getattr(v_high_act, 'exc', 0.0)) - float(getattr(v_low_act, 'exc', 0.0))
        e_2e_full = e_coul_full + e_xc_hybrid
        
        hcore_orig = _as_cupy(self.mf_outer.get_hcore())
        e_1e_core = float(cp.sum(dm_core * hcore_orig))
        
        e_nuc = float(self.full_mol.energy_nuc())
        return e_nuc + e_1e_core + e_h_active + e_2e_full

    def kernel(self):
        if not self.mf_outer.converged:
            self.mf_outer.kernel()
            
        e_global_low = self.mf_outer.e_tot
        mo_coeff = _as_cupy(self.mf_outer.mo_coeff)
        mo_occ = _as_cupy(self.mf_outer.mo_occ)
        dm_full_ao_low = _as_cupy(self.mf_outer.make_rdm1())
        
        hcore_orig = _as_cupy(self.mf_outer.get_hcore())
        s_ao = _as_cupy(self.mf_outer.get_ovlp())
        X, X_inv = lowdin_orth(s_ao)

        ifrag = 0
        
        self.build_bath(ifrag, mo_coeff, mo_occ, X_inv, X)
        self.build_embedded_hamiltonian(ifrag, hcore_orig)
        
        # Build and Run Inner embedded solver
        mf_inner = self._build_inner_mf(ifrag, dm_full_ao_low)

        # Patch for gpu4pyscf inner MF losing _numint (Replaced previous flawed logic)
        if getattr(mf_inner, '_numint', None) is None:
            from gpu4pyscf.dft import numint
            mf_inner._numint = numint.NumInt()
            
        # Ensure it's correctly assigned back to the list so solve_embedded uses the patched object
        self.mf_inner[ifrag] = mf_inner
        
        B_mat = self.B[ifrag]
        dm_core_mat = self.dm_core[ifrag]
        h_eval_bare_mat = B_mat.T @ hcore_orig @ B_mat

        # Add the missing core 1-electron energy (kinetic + nuclear attraction from the frozen core)
        e1_core = float(cp.sum(dm_core_mat * hcore_orig))
        
        e_nuc_full = float(self.full_mol.energy_nuc())
        mf_inner.energy_nuc = lambda *args, **kwargs: e_nuc_full
        
        def custom_hybrid_get_veff(mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
            if dm is None: dm = mf_inner.make_rdm1()
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
            # Note: self.v_core_ao[ifrag] (low-level core Veff) is subtracted to isolate the active field
            # This is because the core Veff is already included in the full h_emb, refering to embedding.py
            v_eff_active = _as_cupy(v_low_full) + _as_cupy(v_high_act) - _as_cupy(v_low_act) - self.v_core_ao[ifrag]
            
            if dm_cp.ndim == 2:
                v_eff_emb = B_mat.T @ v_eff_active @ B_mat
            else:
                v_eff_emb = cp.einsum('pi,xpq,qj->xij', B_mat, v_eff_active, B_mat)
            
            ecoul = float(getattr(v_low_full, 'ecoul', 0.0))
            exc = float(getattr(v_low_full, 'exc', 0.0)) + float(getattr(v_high_act, 'exc', 0.0)) - float(getattr(v_low_act, 'exc', 0.0))
            
            return tag_array(v_eff_emb, ecoul=ecoul, exc=exc)

        mf_inner.get_veff = custom_hybrid_get_veff
        
        def custom_energy_elec(dm=None, h1e=None, vhf=None):
            if dm is None: dm = mf_inner.make_rdm1()
            if vhf is None: vhf = mf_inner.get_veff(mf_inner.mol, dm)
            
            dm_cp = _as_cupy(dm)
            
            # e1: Active space single-electron energy + Core single-electron energy
            e1_active = float(cp.sum(dm_cp * h_eval_bare_mat))
            e1 = e1_active + e1_core
            
            # e2: Full system 2e energy minus core 2e energy
            ecoul_full = float(getattr(vhf, 'ecoul', 0.0))
            exc_hybrid = float(getattr(vhf, 'exc', 0.0))
            e2 = ecoul_full + exc_hybrid
            
            # Update scf_summary for meaningful PySCF debugging output
            mf_inner.scf_summary['e1'] = e1
            mf_inner.scf_summary['coul'] = ecoul_full
            mf_inner.scf_summary['exc'] = exc_hybrid
            
            return e1 + e2, e2
            
        mf_inner.energy_elec = custom_energy_elec
        
        self.log.info("Running high-level inner solver...")
        self.solve_embedded(ifrag)
        if not self.mf_inner[ifrag].converged:
            raise RuntimeError(
                f"Embedded high-level SCF did not converge for fragment {ifrag}; "
                "do not use this density for delta energy."
            )
        
        dm_emb_high = _as_cupy(mf_inner.make_rdm1())
        
        B = self.B[ifrag]
        dm_core = self.dm_core[ifrag]
        is_mean_field = hasattr(self.mf_inner_template, 'get_veff')
        
        if is_mean_field:
            h_eval_bare = B.T @ hcore_orig @ B
            
            # Evaluate High-Level energy using hybrid XC formulation
            e_high = self._evaluate_embedded_energy(
                dm_emb_high, h_eval_bare, B, dm_core
            )
        else:
            raise NotImplementedError("WFT evaluation is not implemented for this class.")
        
        self.log.note(f"Global Low-Level E : {e_global_low:.8f}")
        
        self.e_tot = float(e_high)
        self.log.note(f"Total Embedded E (hybrid XC) : {self.e_tot:.8f}")
        
        return self.e_tot