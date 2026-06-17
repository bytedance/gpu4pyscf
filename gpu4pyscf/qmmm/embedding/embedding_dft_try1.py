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
    Single-Fragment strict subspace variational embedding for DFT (CAS-like).
    
    This class performs a single-shot,
    single-fragment exact subspace energy evaluation WITHOUT ONIOM correction.
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
        Evaluate the exact CAS-DFT energy using the hybrid XC formulation:
        E_tot = Tr(D_tot * h) + 1/2 Tr(D_tot * J_tot) + E_xc^low[tot] + E_xc^high[act] - E_xc^low[act]
        """
        e_h_active = float(cp.sum(dm_emb * h_eval_bare))
        
        dm_act_ao = B @ dm_emb @ B.T
        dm_full_ao = dm_core + dm_act_ao
        
        # 1. Low-level evaluation on full density
        direct_scf_bak = getattr(self.mf_outer, 'direct_scf', True)
        self.mf_outer.direct_scf = False
        v_low_full = self.mf_outer.get_veff(self.full_mol, dm_full_ao)
        
        # 2. Low-level evaluation on active density
        v_low_act = self.mf_outer.get_veff(self.full_mol, dm_act_ao)
        self.mf_outer.direct_scf = direct_scf_bak
        
        # 3. High-level evaluation on active density
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
        
        hcore_orig = _as_cupy