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

'''
Restricted open-shell Hartree-Fock for periodic systems at a single k-point

See Also:
    pyscf/pbc/scf/khf.py : Hartree-Fock for periodic systems with k-point sampling
'''

import numpy as np
import cupy as cp
from pyscf import lib
from pyscf.scf import rohf as rohf_cpu
from gpu4pyscf.lib import logger, utils
from gpu4pyscf.scf import rohf as mol_rohf
from gpu4pyscf.pbc.scf import hf as pbchf
from gpu4pyscf.pbc.scf import uhf as pbcuhf
from gpu4pyscf.lib.cupy_helper import tag_array
from pyscf import __config__



class ROHF(pbchf.RHF):
    '''ROHF class for PBCs at a single point (default: gamma point).
    '''

    get_init_guess = pbcuhf.UHF.get_init_guess
    init_guess_by_chkfile = pbcuhf.UHF.init_guess_by_chkfile
    init_guess_by_minao  = mol_rohf.ROHF.init_guess_by_minao
    init_guess_by_atom   = mol_rohf.ROHF.init_guess_by_atom
    init_guess_by_huckel = mol_rohf.ROHF.init_guess_by_huckel
    init_guess_by_mod_huckel = mol_rohf.ROHF.init_guess_by_mod_huckel
    eig = mol_rohf.ROHF.eig
    get_fock = mol_rohf.ROHF.get_fock
    get_occ = mol_rohf.ROHF.get_occ
    get_grad = mol_rohf.ROHF.get_grad
    make_rdm1 = mol_rohf.ROHF.make_rdm1
    energy_elec = mol_rohf.ROHF.energy_elec
    analyze = mol_rohf.ROHF.analyze
    canonicalize = mol_rohf.ROHF.canonicalize
    spin_square = mol_rohf.ROHF.spin_square
    stability = mol_rohf.ROHF.stability
    dip_moment = pbchf.SCF.dip_moment
    gen_response = pbcuhf.UHF.gen_response

    def __init__(self, cell, kpt=None,
                 exxdiv=getattr(__config__, 'pbc_scf_SCF_exxdiv', 'ewald')):
        pbchf.SCF.__init__(self, cell, kpt, exxdiv)
        self.nelec = None

    @property
    def nelec(self):
        if self._nelec is not None:
            return self._nelec
        else:
            cell = self.cell
            ne = cell.nelectron
            nalpha = (ne + cell.spin) // 2
            nbeta = nalpha - cell.spin
            if nalpha + nbeta != ne:
                raise RuntimeError('Electron number %d and spin %d are not consistent\n'
                                   'Note cell.spin = 2S = Nalpha - Nbeta, not 2S+1' %
                                   (ne, self.spin))
            return nalpha, nbeta
    @nelec.setter
    def nelec(self, x):
        self._nelec = x

    def dump_flags(self, verbose=None):
        pbchf.SCF.dump_flags(self, verbose)
        logger.info(self, 'number of electrons per cell  '
                    'alpha = %d beta = %d', *self.nelec)
        return self

    def get_veff(self, cell=None, dm=None, dm_last=None, vhf_last=None, hermi=1,
                 kpt=None, kpts_band=None):
        if cell is None: cell = self.cell
        if kpt is None: kpt = self.kpt
        if dm is None: dm = self.make_rdm1()
        dm = mol_rohf._to_uhf_dm(dm)
        return pbcuhf.UHF.get_veff(self, cell, dm, dm_last, vhf_last, hermi, kpt, kpts_band)

    def get_bands(self, kpts_band, cell=None, dm=None, kpt=None):
        '''Get energy bands at the given (arbitrary) 'band' k-points.

        Returns:
            mo_energy : (nmo,) ndarray or a list of (nmo,) ndarray
                Bands energies E_n(k)
            mo_coeff : (nao, nmo) ndarray or a list of (nao,nmo) ndarray
                Band orbitals psi_n(k)
        '''
        raise NotImplementedError

    def init_guess_by_1e(self, cell=None):
        if cell is None: cell = self.cell
        if cell.dimension < 3:
            logger.warn(self, 'Hcore initial guess is not recommended in '
                        'the SCF of low-dimensional systems.')
        return mol_rohf.ROHF.init_guess_by_1e(self, cell)

    def to_ks(self, xc='HF'):
        '''Convert to RKS object.
        '''
        raise NotImplementedError
        from gpu4pyscf.pbc import dft
        return self._transfer_attrs_(dft.ROKS(self.cell, self.kpt, xc=xc))

    def to_cpu(self):
        mf = rohf_cpu.ROHF(self.cell)
        with lib.temporary_env(self, _numint=None):
            utils.to_cpu(self, out=mf)
        return mf
