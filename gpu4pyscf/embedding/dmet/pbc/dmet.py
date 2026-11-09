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


from pyscf import lib
from pyscf.ao2mo.incore import _conc_mos
from gpu4pyscf.pbc import scf as pbcscf
from gpu4pyscf.embedding.dmet import dmet as mol_dmet
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import asarray, sandwich_dot, contract, pack_tril
import numpy as np
import cupy as cp

class DMET(mol_dmet.DMET):
    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('low-level SCF = %s', 'ROHF' if isinstance(self.mf, pbcscf.rohf.ROHF) else 'RHF')
        log.info('orth = %s', self.orth)
        log.info('threshold = %s', self.threshold)
        log.info('nimp = %s', self.nimp)
        log.info('nenv = %s', self.nenv)
        log.info('max_memory %d MB (current use %d MB)',
                 self.max_memory, lib.current_memory()[0])

    def check_low_level(self, mf):
        if isinstance(mf, (pbcscf.KRHF, pbcscf.KUHF, pbcscf.KGHF)):
            raise NotImplementedError('Currently only Gamma-point DMET is supported')
        if not isinstance(mf, (pbcscf.hf.RHF, pbcscf.rohf.ROHF)):
            raise NotImplementedError('Currently only RHF and ROHF are supported')
        if mf.exxdiv is not None:
            raise NotImplementedError(f'exxdiv {mf.exxdiv} is not supported in DMET')
        if getattr(mf, 'with_df', None) is not None:
            self.with_df = mf.with_df
        else:
            raise NotImplementedError
        return self

    def get_es_h1e(self, mf=None, core_orb=None, es_orb=None):
        if mf is None: mf = self.mf
        if core_orb is None: core_orb = self.core_orb
        if es_orb is None: es_orb = self.es_orb

        hcore = mf.get_hcore()
        if core_orb.size > 0:
            if isinstance(mf, pbcscf.rohf.ROHF):
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
        naux = with_df.get_naoaux()
        eosym, neo_pair, moij, ijslice = _conc_mos(es_orb.get(), es_orb.get(), compact=True)
        es_cderi = np.zeros([naux, neo_pair])
        blksize = int(with_df.get_blksize() * 0.8)
        q1 = 0
        for cderi_block, buf_cderi, sign in with_df.loop_gamma_point(blksize):
            buf_cderi = sign = None
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

LODMET = DMET

class AODMET(mol_dmet.AODMET, DMET):
    pass