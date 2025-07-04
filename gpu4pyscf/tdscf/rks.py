# Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
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
from gpu4pyscf.tdscf._lr_eig import eigh as lr_eigh
from gpu4pyscf.dft.rks import KohnShamDFT
from gpu4pyscf.lib.cupy_helper import contract, tag_array, transpose_sum
from gpu4pyscf.lib import logger
from gpu4pyscf.tdscf import rhf as tdhf_gpu
from gpu4pyscf import dft

__all__ = [
    'TDA', 'TDDFT', 'TDRKS', 'CasidaTDDFT', 'TDDFTNoHybrid',
]

class TDA(tdhf_gpu.TDA):
    def nuc_grad_method(self):
        if getattr(self._scf, 'with_df', None):
            from gpu4pyscf.df.grad import tdrks
            return tdrks.Gradients(self)
        else:
            from gpu4pyscf.grad import tdrks
            return tdrks.Gradients(self)
    
    def NAC(self):
        if getattr(self._scf, 'with_df', None):
            raise NotImplementedError("density fitting NAC is not supported.")
        else:
            from gpu4pyscf.nac import tdrks
            return tdrks.NAC(self)

class TDDFT(tdhf_gpu.TDHF):
    def nuc_grad_method(self):
        if getattr(self._scf, 'with_df', None):
            from gpu4pyscf.df.grad import tdrks
            return tdrks.Gradients(self)
        else:
            from gpu4pyscf.grad import tdrks
            return tdrks.Gradients(self)

    def NAC(self):
        if getattr(self._scf, 'with_df', None):
            raise NotImplementedError("density fitting NAC is not supported.")
        else:
            from gpu4pyscf.nac import tdrks
            return tdrks.NAC(self)
TDRKS = TDDFT

class CasidaTDDFT(TDDFT):
    '''Solve the Casida TDDFT formula (A-B)(A+B)(X+Y) = (X+Y)w^2
    '''

    init_guess = TDA.init_guess
    get_precond = TDA.get_precond

    def gen_vind(self, mf=None):
        if mf is None:
            mf = self._scf
        singlet = self.singlet
        mo_coeff = mf.mo_coeff
        assert mo_coeff.dtype == cp.double
        mo_energy = mf.mo_energy
        mo_occ = mf.mo_occ
        occidx = mo_occ == 2
        viridx = mo_occ == 0
        orbv = mo_coeff[:,viridx]
        orbo = mo_coeff[:,occidx]

        e_ia = mo_energy[viridx] - mo_energy[occidx,None]
        d_ia = e_ia ** .5
        ed_ia = e_ia * d_ia
        hdiag = e_ia.ravel() ** 2
        vresp = self.gen_response(singlet=singlet, hermi=1)
        nocc, nvir = e_ia.shape

        def vind(zs):
            zs = cp.asarray(zs).reshape(-1,nocc,nvir)
            # *2 for double occupancy
            mo1 = contract('xov,pv->xpo', zs*(d_ia*2), orbv)
            dms = contract('xpo,qo->xpq', mo1, orbo)
            # +cc for A+B and K_{ai,jb} in A == K_{ai,bj} in B
            dms = transpose_sum(dms)
            dms = tag_array(dms, mo1=mo1, occ_coeff=orbo)
            v1ao = vresp(dms)
            v1mo = contract('xpq,qo->xpo', v1ao, orbo)
            v1mo = contract('xpo,pv->xov', v1mo, orbv)
            v1mo += zs * ed_ia
            v1mo *= d_ia
            return v1mo.reshape(v1mo.shape[0],-1)

        return vind, hdiag

    def kernel(self, x0=None, nstates=None):
        '''TDDFT diagonalization solver
        '''
        log = logger.new_logger(self)
        cpu0 = log.init_timer()
        mf = self._scf
        if mf._numint.libxc.is_hybrid_xc(mf.xc):
            raise RuntimeError('%s cannot be used with hybrid functional'
                               % self.__class__)
        self.check_sanity()
        self.dump_flags()
        if nstates is None:
            nstates = self.nstates
        else:
            self.nstates = nstates

        vind, hdiag = self.gen_vind(self._scf)
        precond = self.get_precond(hdiag)

        def pickeig(w, v, nroots, envs):
            idx = cp.where(w > self.positive_eig_threshold)[0]
            return w[idx], v[:,idx], idx

        x0sym = None
        if x0 is None:
            x0 = self.init_guess()

        self.converged, w2, x1 = lr_eigh(
            vind, x0, precond, tol_residual=self.conv_tol, lindep=self.lindep,
            nroots=nstates, x0sym=x0sym, pick=pickeig, max_cycle=self.max_cycle,
            max_memory=self.max_memory, verbose=log)

        mo_energy = self._scf.mo_energy
        mo_occ = self._scf.mo_occ
        occidx = mo_occ == 2
        viridx = mo_occ == 0
        e_ia = mo_energy[viridx] - mo_energy[occidx,None]
        e_ia = e_ia**.5
        if isinstance(e_ia, cp.ndarray):
            e_ia = e_ia.get()

        def norm_xy(w, z):
            zp = e_ia * z.reshape(e_ia.shape)
            zm = w/e_ia * z.reshape(e_ia.shape)
            x = (zp + zm) * .5
            y = (zp - zm) * .5
            norm = lib.norm(x)**2 - lib.norm(y)**2
            norm = abs(.5/norm)**.5  # normalize to 0.5 for alpha spin
            return (x*norm, y*norm)

        idx = np.where(w2 > self.positive_eig_threshold)[0]
        self.e = w2[idx]**.5
        self.xy = [norm_xy(self.e[i], x1[i]) for i in idx]
        log.timer('TDDFT', *cpu0)
        self._finalize()
        return self.e, self.xy

TDDFTNoHybrid = CasidaTDDFT

def tddft(mf):
    '''Driver to create TDDFT or CasidaTDDFT object'''
    if mf._numint.libxc.is_hybrid_xc(mf.xc):
        return TDDFT(mf)
    else:
        return CasidaTDDFT(mf)

dft.rks.RKS.TDA           = lib.class_as_method(TDA)
dft.rks.RKS.TDHF          = None
#dft.rks.RKS.TDDFT         = lib.class_as_method(TDDFT)
dft.rks.RKS.TDDFTNoHybrid = lib.class_as_method(TDDFTNoHybrid)
dft.rks.RKS.CasidaTDDFT   = lib.class_as_method(CasidaTDDFT)
dft.rks.RKS.TDDFT         = tddft
