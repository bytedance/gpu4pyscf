# Copyright 2026 The PySCF Developers. All Rights Reserved.
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


from pyscf import __config__
from pyscf import gto, lib
from pyscf import scf as cpu_scf
from pyscf.ao2mo.incore import _conc_mos
from gpu4pyscf import scf
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import asarray, sandwich_dot, contract, pack_tril, block_diag
import numpy as np
import cupy as cp

DMET_ORTH_METHOD = getattr(__config__, 'dmet_orth_method', 'lowdin')
DMET_BATH_THRESHOLD = getattr(__config__, 'dmet_bath_threshold', 1e-10)

def lowdin(s):
    e, v = cp.linalg.eigh(s)
    idx = e > 1e-15
    caolo = cp.dot(v[:,idx]/cp.sqrt(e[idx]), v[:,idx].conj().T)
    cloao = cp.dot(caolo, s)
    return caolo, cloao

class DMETBase(lib.StreamObject):
    '''
    Single-impurity One-shot Density Matrix Embedding Theory (DMET)

    Args:
        mf: SCF object
            Low-level spin-restricted mean-field object of the global system.
        imp: int or string or a list of strings
            The label of the impurity orbtials
        orth: string
            The orthogonalization method used to define the impurity orbitals.  Default value is 'lowdin'.
        threshold: float
            Threshold for the eigenvalues of the environmental density matrix to determine the number of bath orbitals.

    Attributes:
        verbose : int
            Print level.  Default value equals to :class:`Mole.verbose`.
        max_memory : float or int
            Allowed memory in MB.  Default value equals to :class:`Mole.max_memory`.
        nimp : int
            Number of impurity orbitals.
        nenv : int
            Number of environment orbitals.
    
    Saved results

    '''

    _keys = {
        'mf', 'mol', 'orth', 'threshold', 'verbose', 'max_memory', 'nimp', 'imp', 'nenv', '_es_df', 'with_df'
    }

    def __init__(self, mf, imp=0, orth=DMET_ORTH_METHOD, threshold=DMET_BATH_THRESHOLD):
        self.check_low_level(mf)

        mol = mf.mol
        self.mol = mol
        self.mf = mf
        self.orth = orth
        self.threshold = threshold
        self.verbose = mol.verbose
        self.stdout = mol.stdout
        self.max_memory = mol.max_memory
        if isinstance(imp, int):
            if imp < 0 or imp >= mol.natm:
                raise ValueError(f'Atom {imp} is out of range')
            aoslice = mol.aoslice_by_atom()[imp]
            self.imp = asarray(cp.arange(aoslice[2], aoslice[3]))
        elif isinstance(imp, str):
            self.imp = asarray(mol.search_ao_label(imp))
        else:
            self.imp = []
            for label in imp:
                if isinstance(label, int):
                    if label < 0 or label >= mol.natm:
                        raise ValueError(f'Atom {label} is out of range')
                    aoslice = mol.aoslice_by_atom()[label]
                    self.imp.extend(np.arange(aoslice[2], aoslice[3]))
                else:
                    self.imp.extend(mol.search_ao_label(label))
            self.imp = asarray(self.imp)
        self.nimp = len(self.imp)
        self.nenv = mol.nao_nr() - self.nimp

        self._es_df = False

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('low-level SCF = %s', 'ROHF' if isinstance(self.mf, scf.rohf.ROHF) else 'RHF')
        log.info('orth = %s', self.orth)
        log.info('threshold = %s', self.threshold)
        log.info('nimp = %s', self.nimp)
        log.info('nenv = %s', self.nenv)
        log.info('max_memory %d MB (current use %d MB)',
                 self.max_memory, lib.current_memory()[0])

    def check_sanity(self):
        super().check_sanity()
        if self.nimp == 0:
            raise ValueError('No impurity orbitals')
        return self

    def check_low_level(self, mf):
        if not isinstance(mf, (scf.hf.RHF, scf.rohf.ROHF)):
            raise NotImplementedError('Currently only RHF and ROHF are supported')
        if getattr(mf, 'with_df', None) is not None:
            self.with_df = mf.with_df
        else:
            raise NotImplementedError
        return self

    def energy_nuc(self):
        return self.mf.energy_nuc()

    def get_orth(self, mf=None, orth=None, dm=None):
        raise NotImplementedError

    def get_es_h1e(self, mf=None, core_orb=None, es_orb=None):
        if mf is None: mf = self.mf
        if core_orb is None: core_orb = self.core_orb
        if es_orb is None: es_orb = self.es_orb

        hcore = mf.get_hcore()
        if core_orb.size > 0:
            if isinstance(mf, scf.rohf.ROHF):
                core_dm = cp.dot(core_orb, core_orb.T)
                core_dm = asarray([core_dm, core_dm])
                veff = mf.get_veff(dm=core_dm)[0]
            else:
                core_dm = cp.dot(core_orb, core_orb.T)*2
                veff = mf.get_veff(dm=core_dm)
            h1e = hcore + veff
        else:
            h1e = hcore
        es_h1e = sandwich_dot(h1e, es_orb)
        return es_h1e.get()

    def get_es_cderi(self, mf=None, es_orb=None):
        if mf is None: mf = self.mf
        if es_orb is None: es_orb = self.es_orb

        with_df = mf.with_df
        naux = with_df.naux
        eosym, neo_pair, moij, ijslice = _conc_mos(es_orb.get(), es_orb.get(), compact=True)
        es_cderi = np.zeros([naux, neo_pair])
        blksize = int(with_df.get_blksize() * 0.8)
        q1 = 0
        for cderi_block, buf_cderi in with_df.loop(blksize):
            buf_cderi = None
            nblock = cderi_block.shape[0]
            q0, q1 = q1, q1 + nblock
            tmp = contract('Lij,ip->Lpj', cderi_block, es_orb)
            tmp = contract('Lpj,jq->Lpq', tmp, es_orb)
            if eosym == 's2':
                tmp = pack_tril(tmp)
            else:
                tmp = tmp.reshape(nblock, -1)
            es_cderi[q0:q1] += tmp.get()
            tmp = None
        return es_cderi
    
    def get_es_s1e(self, mf=None, es_orb=None):
        if mf is None: mf = self.mf
        if es_orb is None: es_orb = self.es_orb
        s1e = mf.get_ovlp()
        es_s1e = sandwich_dot(s1e, es_orb)
        return es_s1e.get()

    def energy_core(self, mf=None, core_orb=None):
        if mf is None: mf = self.mf
        if core_orb is None: core_orb = self.core_orb

        if core_orb.size > 0:
            if isinstance(mf, scf.rohf.ROHF):
                core_dm = cp.dot(core_orb, core_orb.T)
                core_dm = asarray([core_dm, core_dm])
            else:
                core_dm = cp.dot(core_orb, core_orb.T)*2
            e_core = mf.energy_elec(dm=core_dm)[0]
        else:
            e_core = 0
        return e_core

    def get_es_dm(self, C_proj, dm=None):
        if dm is None: dm = self.mf.make_rdm1()

        if isinstance(self.mf, scf.rohf.ROHF):
            es_dm = cp.zeros((2, self.neo, self.neo))
            dma, dmb = dm
            es_dm[0] = sandwich_dot(dma, C_proj)
            es_dm[1] = sandwich_dot(dmb, C_proj)
        else:
            es_dm = sandwich_dot(dm, C_proj)
        return es_dm.get()

    def build_embedded_orbitals(self):
        raise NotImplementedError

    def build_embedded_hamiltonian(self, mf, es_orb=None, core_orb=None):
        if mf is None: mf = self.mf
        if es_orb is None: es_orb = self.es_orb
        if core_orb is None: core_orb = self.core_orb

        log = logger.new_logger(self, self.verbose)
        t0 = log.init_timer()

        self.es_h1e = self.get_es_h1e(mf, core_orb, es_orb)
        t0 = log.timer_debug1('embedded h1e', *t0)

        self.es_cderi = self.get_es_cderi(mf, es_orb)
        t0 = log.timer_debug1('embedded cderi', *t0)

        self.es_s1e = self.get_es_s1e(mf, es_orb)
        t0 = log.timer_debug1('embedded s1e', *t0)

        return self

    def kernel(self):
        self.check_sanity()
        self.dump_flags()
        mf = self.mf

        log = logger.new_logger(self, self.verbose)
        t0 = t1 = log.init_timer()
        emb_orb, es_dm = self.build_embedded_orbitals(mf, self.imp, self.threshold)
        self.emb_orb = emb_orb
        log.info('')
        log.info('number of impurity orbitals = %s', self.nimp)
        log.info('number of bath orbitals = %s', self.nbath)
        log.info('number of embedded cluster orbitals = %s', self.neo)
        log.info('number of frozen core orbitals = %s', self.ncore)
        log.info('number of frozen virtual orbitals = %s', self.nvir)
        log.info('number of frozen orbitals = %s', self.ncore + self.nvir)
        t1 = log.timer('embedded orbitals', *t1)

        es_orb = emb_orb[:, :self.neo]
        core_orb = emb_orb[:, self.neo: self.neo+self.ncore]
        self.build_embedded_hamiltonian(mf, es_orb, core_orb)
        t1 = log.timer('embedded Hamiltonian', *t1)

        es_mol = gto.Mole()
        es_mol.verbose = 0
        es_mol.incore_anyway = True
        es_mol.nelectron = self.mol.nelectron - 2*self.ncore
        es_mol.spin = self.mol.spin
        es_mol.build()

        if isinstance(self.mf, scf.rohf.ROHF):
            es_mf = cpu_scf.ROHF(es_mol)
        else:
            es_mf = cpu_scf.RHF(es_mol)
        es_mf.get_hcore = lambda *args: self.es_h1e
        es_mf.get_ovlp = lambda *args: self.es_s1e
        es_mf.mo_energy = np.zeros(self.neo)
        
        if mf.level_shift is None:
            es_mf.level_shift = 0
        else:
            es_mf.level_shift = mf.level_shift
        es_mf.conv_check = mf.conv_check
        es_mf.conv_tol = mf.conv_tol
        es_mf.conv_tol_grad = mf.conv_tol_grad
        es_mf.max_cycle = mf.max_cycle
        es_mf.verbose = mf.verbose
        if self._es_df:
            es_mf = es_mf.density_fit()
            es_mf.with_df._cderi = self.es_cderi
        else:
            es_mf._eri = self.es_cderi.T @ self.es_cderi
            self.es_cderi = None
        es_mf.kernel(es_dm)
        self.es_mf = es_mf
        t1 = log.timer('embedded mean-field', *t1)

        e_es = es_mf.e_tot
        e_core = self.energy_core(mf, core_orb)
        e_nuc = self.energy_nuc()
        t1 = log.timer_debug1('core energy', *t1)

        e_tot = e_es + e_core + e_nuc
        self.e_es = e_es
        self.e_core = e_core
        self.e_nuc = e_nuc
        self.e_tot = e_tot

        t0 = log.timer('DMET', *t0)
        self._finalize()
        return es_mf

    def _finalize(self):
        log = logger.Logger(self.stdout, self.verbose)

        log.info('')
        log.info('E(es-mf) = %#.15g  E(core) = %#.15g  E(nuc) = %#.12g', self.e_es, self.e_core, self.e_nuc)
        log.info('E = %#.15g  E - E(mf) = %#.8g', self.e_tot, self.e_tot - self.mf.e_tot)
        return self

    def density_fit(self):
        self._es_df = True
        return self

    def make_global_mf(self):
        raise NotImplementedError

class DMET(DMETBase):
    def get_orth(self, mf=None, orth=None, dm=None):
        if mf is None: mf = self.mf
        if orth is None: orth = self.orth
        if dm is None: dm = mf.make_rdm1()
        if dm.ndim == 3:
            dm = dm[0] + dm[1]

        s = mf.get_ovlp()
        if orth.lower() == 'lowdin':
            caolo, cloao = lowdin(s)
        elif orth.lower() == 'iplo':
            # Impurity-preserved orthogonalization
            # J. Chem. Theory Comput. 2026, 22, 10126–10136.
            caolo, cloao = lowdin(s)
            imp = self.imp
            mask_env = cp.ones(self.mol.nao, dtype=bool)
            mask_env[imp] = False

            Q1 = cloao[:, imp]
            Q1, _ = cp.linalg.qr(Q1)
            P = cp.eye(*cloao.shape) - Q1 @ Q1.T.conj()
            B = cp.dot(P, cloao[:, mask_env])
            U, S, Vh = cp.linalg.svd(B, full_matrices=False)

            Q = cp.zeros(cloao.shape)
            Q[:, imp] = Q1
            Q[:, mask_env] = U[:, :self.nenv]
            cloao = cp.dot(Q.T.conj(), cloao)
            caolo = cp.dot(caolo, Q)
        else:
            raise NotImplementedError

        ldm = sandwich_dot(dm, cloao.T)
        return ldm, caolo, cloao

    def build_embedded_orbitals(self, mf=None, imp=None, threshold=None):
        if mf is None: mf = self.mf
        if imp is None: imp = self.imp
        if threshold is None: threshold = self.threshold

        nao = self.mol.nao
        env = asarray([x for x in range(nao) if x not in imp])
        dm = mf.make_rdm1()
        if dm.ndim == 3:
            dm = dm[0] + dm[1]
        ldm, caolo, cloao = self.get_orth(mf, self.orth, dm)
        ldm_env = ldm[env,:][:,env]

        occ_env, orb_env = cp.linalg.eigh(ldm_env)

        nimp = self.nimp
        nvir = int(cp.sum(occ_env < threshold))
        nbath = int(cp.sum((occ_env >= threshold) & (occ_env <= 2-threshold)))
        ncore = int(cp.sum(occ_env > 2-threshold))
        neo = nimp + nbath
        self.neo = neo
        self.nbath = nbath
        self.ncore = ncore
        self.nvir = nvir

        vir_idx = cp.nonzero(occ_env < threshold)[0]
        bath_idx = cp.nonzero((occ_env >= threshold) & (occ_env <= 2-threshold))[0]
        core_idx = cp.nonzero(occ_env > 2-threshold)[0]
        orb_env = cp.hstack((orb_env[:, bath_idx], orb_env[:, core_idx], orb_env[:, vir_idx]))
        cloes = block_diag([cp.eye(nimp), orb_env])
        rearange_idx = cp.argsort(cp.concatenate((imp, env)))
        cloes = cloes[rearange_idx, :]

        cloeo = cloes[:, :neo]
        emb_orb = cp.dot(caolo, cloes)

        C_proj = cp.dot(cloao.T, cloeo)
        es_dm = self.get_es_dm(C_proj)
        return emb_orb, es_dm

    def make_global_mf(self):
        emb_orb = self.emb_orb
        es_orb = emb_orb[:, :self.neo]
        core_orb = emb_orb[:, self.neo: self.neo+self.ncore]
        vir_orb = emb_orb[:, self.neo+self.ncore:]
        global_mf = self.mf.copy()
        global_mf.mo_coeff = cp.hstack((core_orb, es_orb @ asarray(self.es_mf.mo_coeff), vir_orb))
        global_mf.mo_occ = cp.hstack((2*cp.ones(self.ncore), asarray(self.es_mf.mo_occ), cp.zeros(self.nvir)))
        return global_mf

LODMET = DMET

class AODMET(DMET):
    def get_orth(self, mf=None, orth=None, dm=None):
        if mf is None: mf = self.mf
        if orth is None: orth = self.orth
        if dm is None: dm = mf.make_rdm1()
        if dm.ndim == 3:
            dm = dm[0] + dm[1]

        mol = mf.mol
        s = mf.get_ovlp()
        imp = self.imp
        env = asarray([x for x in range(self.mol.nao) if x not in imp])
        s_env = s[env,:][:,env]
        caolo = cp.eye(mol.nao)
        cloao = cp.eye(mol.nao)
        if orth.lower() == 'lowdin':
            caolo = cp.eye(mol.nao)
            cloao = cp.eye(mol.nao)
            s_env = s[env,:][:,env]
            caolo_env, cloao_env = lowdin(s_env)
        elif orth.lower() == 'cholesky':
            cloao_env = cp.linalg.cholesky(s_env).T
            caolo_env = cp.linalg.inv(cloao_env)
        else:
            raise NotImplementedError
        caolo[cp.ix_(env,env)] = caolo_env
        cloao[cp.ix_(env,env)] = cloao_env

        ldm = sandwich_dot(dm, cloao.T)
        return ldm, caolo, cloao

    def build_embedded_orbitals(self, mf=None, imp=None, threshold=None):
        if mf is None: mf = self.mf
        if imp is None: imp = self.imp
        if threshold is None: threshold = self.threshold

        nao = self.mol.nao
        env = asarray([x for x in range(nao) if x not in imp])
        dm = mf.make_rdm1()
        if dm.ndim == 3:
            dm = dm[0] + dm[1]
        ldm, caolo, cloao = self.get_orth(mf, self.orth, dm)
        ldm_env = ldm[env,:][:,env]

        occ_env, orb_env = cp.linalg.eigh(ldm_env)

        nimp = self.nimp
        nvir = int(cp.sum(occ_env < threshold))
        nbath = int(cp.sum(((occ_env >= threshold) & (occ_env <= 2-threshold)) | (occ_env >= 2+threshold)))
        ncore = int(cp.sum((occ_env > 2-threshold) & (occ_env < 2+threshold)))
        neo = nimp + nbath
        self.neo = neo
        self.nbath = nbath
        self.ncore = ncore
        self.nvir = nvir

        vir_idx = cp.nonzero(occ_env < threshold)[0]
        bath_idx = cp.nonzero(((occ_env >= threshold) & (occ_env <= 2-threshold)) | (occ_env >= 2+threshold))[0]
        core_idx = cp.nonzero((occ_env > 2-threshold) & (occ_env < 2+threshold))[0]
        orb_env = cp.hstack((orb_env[:, bath_idx], orb_env[:, core_idx], orb_env[:, vir_idx]))
        cloes = block_diag([cp.eye(nimp), orb_env])
        rearange_idx = cp.argsort(cp.concatenate((imp, env)))
        cloes = cloes[rearange_idx, :]

        cloeo = cloes[:, :neo]
        emb_orb = cp.dot(caolo, cloes)

        C_proj = cp.dot(cloao.T, cloeo)
        es_dm = self.get_es_dm(C_proj)
        return emb_orb, es_dm

    def make_global_mf(self):
        emb_orb = self.emb_orb
        es_orb = emb_orb[:, :self.neo]
        core_orb = emb_orb[:, self.neo: self.neo+self.ncore]
        vir_orb = emb_orb[:, self.neo+self.ncore:]

        s = self.mf.get_ovlp()
        occ_orb = cp.hstack([core_orb, es_orb @ asarray(self.es_mf.mo_coeff)])
        proj = occ_orb.T @ s @ vir_orb
        vir_proj = vir_orb - occ_orb @ proj
        s_vir_proj = sandwich_dot(s, vir_proj)
        vir_orth = vir_proj @ lowdin(s_vir_proj)[0]

        global_mf = self.mf.copy()
        global_mf.mo_coeff = cp.hstack((occ_orb, vir_orth))
        global_mf.mo_occ = cp.hstack((2*cp.ones(self.ncore), asarray(self.es_mf.mo_occ), cp.zeros(self.nvir)))
        return global_mf