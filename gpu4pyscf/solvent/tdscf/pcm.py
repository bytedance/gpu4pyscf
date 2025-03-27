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

'''
TD of PCM family solvent model
'''

import numpy
import cupy
import ctypes
from cupyx import scipy
from pyscf import lib
from pyscf import gto
from pyscf.grad import rhf as rhf_grad
from gpu4pyscf.gto import int3c1e
from gpu4pyscf.solvent.pcm import PI, switch_h, libsolvent
from gpu4pyscf.gto.int3c1e_ip import int1e_grids_ip1, int1e_grids_ip2
from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf.lib import logger
from pyscf import lib as pyscf_lib

libdft = lib.load_library('libdft')

def make_tda_object(tda_method):
    '''For td_method in vacuum, add td of solvent pcmobj'''
    name = (tda_method._scf.with_solvent.__class__.__name__
            + tda_method.__class__.__name__)
    return lib.set_class(WithSolventTDA(tda_method),
                         (WithSolventTDA, tda_method.__class__), name)

class WithSolventTDA:
    from gpu4pyscf.lib.utils import to_gpu, device

    def __init__(self, grad_method):
        self.__dict__.update(grad_method.__dict__)

    def undo_solvent(self):
        cls = self.__class__
        name_mixin = self.base.with_solvent.__class__.__name__
        obj = lib.view(self, lib.drop_class(cls, WithSolventTDA, name_mixin))
        return obj
    
    to_cpu = NotImplemented