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

from pyscf import gto, mcscf
from gpu4pyscf.scf import rohf
from gpu4pyscf.embedding.dmet import dmet

def get_mol(dihedral):
    mol = gto.M(atom = '''
               Co             
               S                  1            2.30186590
               S                  1            2.30186590    2            109.47122060
               S                  1            2.30186590    3            109.47122065    2            -120.00000001                  0
               S                  1            2.30186590    4            109.47122060    3            120.00000001                   0
               H                  2            1.30714645    1            109.47121982    4            '''+str(-60-dihedral)+'''      0
               H                  4            1.30714645    1            109.47121982    3            '''+str(60+dihedral)+'''       0
               H                  5            1.30714645    1            109.47121982    4            '''+str(-180+dihedral)+'''     0
               H                  3            1.30714645    1            109.47121982    4            '''+str(60-dihedral)+'''       0
    ''',
    basis='def2svp', spin=3, charge=-2, verbose=4)
    return mol

mol = get_mol(0)
mf = rohf.ROHF(mol).density_fit()
mf.init_guess = 'atom'
mf.level_shift = 0.2
mf.kernel()

dmet_GPU = dmet.DMET(mf, 0, orth='lowdin', threshold=1e-12)          # Select the orbitals on Co after Lowdin orthogonalization as impurity
dmet_GPU = dmet.DMET(mf, 'Co.*', orth='lowdin', threshold=1e-12)     # See also pyscf.gto.mole.search_ao_label
# dmet_GPU = dmet.DMET(mf, ['Co.*', 'S.*'], orth='lowdin')
es_mf = dmet_GPU.kernel()

print('E diff = ', dmet_GPU.e_tot - mf.e_tot)

es_cas = mcscf.CASCI(es_mf, 5, 7)                                    # High-level solver
es_cas.verbose = 4
es_cas.kernel()

print('E dmet-casci = ', es_cas.e_tot + dmet_GPU.e_core + dmet_GPU.e_nuc)

# AO-DMET
dmet_GPU = dmet.AODMET(mf, 'Co.*', threshold=1e-12)
es_mf = dmet_GPU.kernel()

print('E diff = ', dmet_GPU.e_tot - mf.e_tot)

es_cas = mcscf.CASCI(es_mf, 5, 7)
es_cas.verbose = 4
es_cas.kernel()

print('E aodmet-casci = ', es_cas.e_tot + dmet_GPU.e_core + dmet_GPU.e_nuc)