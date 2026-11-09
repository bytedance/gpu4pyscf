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

import numpy as np
from pyscf.pbc import gto
from pyscf import mcscf
from gpu4pyscf.pbc.scf import rohf
from gpu4pyscf.embedding.dmet.pbc import dmet

atom = '''
    Co        0.00000000        0.00000000        0.00000000
    Mg        0.00000000        2.10700011        2.10700011
    Mg        2.10700011        0.00000000        2.10700011
    Mg        2.10700011        2.10700011        0.00000000
    O         2.10700011        2.10700011        2.10700011
    O         2.10700011        0.00000000        0.00000000
    O         0.00000000        2.10700011        0.00000000
    O         0.00000000        0.00000000        2.10700011
'''

cell = gto.Cell(
    atom = atom,
    a = np.eye(3)*4.214,
    unit = 'A',
    basis = 'gth-dzvp-molopt-sr',
    pseudo = ' gth-hf-rev',
    spin = 3,
    verbose = 4,
)
mf = rohf.ROHF(cell, exxdiv=None).density_fit()
mf.init_guess = 'atom'
mf.level_shift = 0.2
mf.conv_tol = 1e-8
mf.max_cycle = 100
mf.kernel()

dmet_GPU = dmet.DMET(mf, 0, orth='lowdin')                           # Select the orbitals on Co after Lowdin orthogonalization as impurity
dmet_GPU = dmet.DMET(mf, 'Co.*', orth='lowdin')                      # See also pyscf.gto.mole.search_ao_label
# dmet_GPU = dmet.DMET(mf, ['Co.*', 'S.*'], orth='lowdin')
es_mf = dmet_GPU.kernel()

print('E diff = ', dmet_GPU.e_tot - mf.e_tot)

es_cas = mcscf.CASCI(es_mf, 5, 7)                                    # High-level solver
es_cas.verbose = 4
es_cas.kernel()

print('E dmet-casci = ', es_cas.e_tot + dmet_GPU.e_core + dmet_GPU.e_nuc)

# AO-DMET
dmet_GPU = dmet.AODMET(mf, 'Co.*')
es_mf = dmet_GPU.kernel()

print('E diff = ', dmet_GPU.e_tot - mf.e_tot)

es_cas = mcscf.CASCI(es_mf, 5, 7)
es_cas.verbose = 4
es_cas.kernel()

print('E aodmet-casci = ', es_cas.e_tot + dmet_GPU.e_core + dmet_GPU.e_nuc)