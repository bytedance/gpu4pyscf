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

from functools import reduce, partial
import numpy as np
import cupy as cp
from pyscf import lib
from gpu4pyscf.lib import logger
from gpu4pyscf.scf import ucphf
from gpu4pyscf.dft import numint
from gpu4pyscf.df import int3c2e
from gpu4pyscf.lib.cupy_helper import contract, add_sparse
from gpu4pyscf.grad import rks as rks_grad
from gpu4pyscf.grad import rhf as rhf_grad
from gpu4pyscf.grad import tdrhf as tdrhf_grad
from gpu4pyscf.tdscf._uhf_resp_sf import cache_xc_kernel_sf, mcfun_eval_xc_adapter_sf
from gpu4pyscf.grad import tdrks


def grad_elec(td_grad, x_y, atmlst=None, max_memory=2000, verbose=logger.INFO):
    ''' Spin flip TDDFT gradient in UKS framework. Note: This function supports
    both TDA or TDDFT results.

    Parameters
    ----------
    Args:
        td_grad : sftda.TDA_SF object.

    Returns:
        The gradient of excited states: Ei^{\\xi} = E0^{\\xi} + wi^{\\xi}
    This function is based on https://github.com/pyscf/pyscf-forge/blob/master/pyscf/grad/tduks_sf.py
    '''
    log = logger.new_logger(td_grad, verbose)
    time0 = logger.process_clock(), logger.perf_counter()

    mol = td_grad.mol
    mf = td_grad.base._scf

    mo_coeff = mf.mo_coeff
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    if not isinstance(mo_occ, cp.ndarray):
        mo_occ = cp.asarray(mo_occ)
    if not isinstance(mo_energy, cp.ndarray):
        mo_energy = cp.asarray(mo_energy)
    if not isinstance(mo_coeff, cp.ndarray):
        mo_coeff = cp.asarray(mo_coeff)
    
    occidxa = cp.where(mo_occ[0]>0)[0]
    occidxb = cp.where(mo_occ[1]>0)[0]
    viridxa = cp.where(mo_occ[0]==0)[0]
    viridxb = cp.where(mo_occ[1]==0)[0]
    nocca = len(occidxa)
    noccb = len(occidxb)
    nvira = len(viridxa)
    nvirb = len(viridxb)
    orboa = mo_coeff[0][:,occidxa]
    orbob = mo_coeff[1][:,occidxb]
    orbva = mo_coeff[0][:,viridxa]
    orbvb = mo_coeff[1][:,viridxb]
    nao = mo_coeff[0].shape[0]

    nmoa = nocca + nvira
    nmob = noccb + nvirb

    if td_grad.base.extype ==  0: # spin-flip-up
        # x_ab, a means vira, b means occb
        x_ab, y_ab = x_y
        if not isinstance(x_ab, cp.ndarray):
            x_ab = cp.asarray(x_ab)
        xpy_ab = x_ab.T
        xmy_ab = x_ab.T

        dvv_a = cp.einsum('ai,bi->ab', xpy_ab, xpy_ab) + cp.einsum('ai,bi->ab', xmy_ab, xmy_ab) # T^{ab \alpha \beta}*2
        doo_b =-cp.einsum('ai,aj->ij', xpy_ab, xpy_ab) - cp.einsum('ai,aj->ij', xmy_ab, xmy_ab) # T^{ij \alpha \beta}*2

        dmxpy_ab = reduce(cp.dot, (orbva, xpy_ab, orbob.T)) # ua ai iv -> uv -> (X+Y)_{uv \alpha \beta}
        dmxmy_ab = reduce(cp.dot, (orbva, xmy_ab, orbob.T)) # ua ai iv -> uv -> (X-Y)_{uv \alpha \beta}

        dmzoo_b = reduce(cp.dot, (orbob, doo_b, orbob.T)) # \sum_{\sigma ab} 2*Tij \sigma C_{iu} C_{iu}
        dmzoo_a = reduce(cp.dot, (orbva, dvv_a, orbva.T))

        ni = mf._numint
        ni.libxc.test_deriv_order(mf.xc, 3, raise_error=True)
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)

        # used by mcfun.
        rho0, vxc, fxc = ni.cache_xc_kernel(mf.mol, mf.grids, mf.xc,
                                        mo_coeff, mo_occ, spin=1)

        f1vo, f1oo, vxc1, k1ao = \
                _contract_xc_kernel(td_grad, mf.xc, (dmxpy_ab, dmxmy_ab),
                                    (dmzoo_a,dmzoo_b), True, True, max_memory, td_grad.base.extype)
        k1ao_xpy, k1ao_xmy = k1ao

        # f1vo, (2,2,4,nao,nao), (X+Y) and (X-Y) with fxc_sf
        # f1oo, (2,4,nao,nao), 2T with fxc_sc
        # vxc1, ao with v1^{\sigma}
        # k1ao_xpy，(2,2,4,nao,nao), (X+Y)(X+Y) and (X-Y)(X-Y) with gxc

        if abs(hyb) > 1e-10:
            dm = (dmzoo_a, dmxpy_ab.T, -dmxmy_ab.T,
                  dmzoo_b, dmxpy_ab, dmxmy_ab)
            vj, vk = mf.get_jk(mol, dm, hermi=0)
            vk *= hyb
            if abs(omega) > 1e-10:
                vk += mf.get_k(mol, dm, hermi=0, omega=omega) * (alpha-hyb)
            vj = vj.reshape(2,3,nao,nao)
            vk = vk.reshape(2,3,nao,nao)

            veff0doo = vj[0,0]+vj[1,0] - vk[:,0]+ f1oo[:,0]
            veff0doo[0] += (k1ao_xpy[0,0,0] + k1ao_xpy[0,1,0] + k1ao_xpy[1,0,0] + k1ao_xpy[1,1,0]
                           +k1ao_xmy[0,0,0] + k1ao_xmy[0,1,0] + k1ao_xmy[1,0,0] + k1ao_xmy[1,1,0])
            veff0doo[1] += (k1ao_xpy[0,0,0] + k1ao_xpy[0,1,0] - k1ao_xpy[1,0,0] - k1ao_xpy[1,1,0]
                           +k1ao_xmy[0,0,0] + k1ao_xmy[0,1,0] - k1ao_xmy[1,0,0] - k1ao_xmy[1,1,0])

            wvoa = reduce(cp.dot, (orbva.T, veff0doo[0], orboa)) *2
            wvob = reduce(cp.dot, (orbvb.T, veff0doo[1], orbob)) *2

            veff = - vk[:,1] + f1vo[0,:,0]
            veff0mop_ab = reduce(cp.dot, (mo_coeff[0].T, veff[1], mo_coeff[1]))

            wvob += cp.einsum('ca,ci->ai', veff0mop_ab[nocca:,noccb:], xpy_ab) *2
            wvoa -= cp.einsum('il,al->ai', veff0mop_ab[:nocca,:noccb], xpy_ab) *2

            veff = -vk[:,2] + f1vo[1,:,0]
            veff0mom_ab = reduce(cp.dot, (mo_coeff[0].T, veff[1], mo_coeff[1]))

            wvob += cp.einsum('ca,ci->ai', veff0mom_ab[nocca:,noccb:], xmy_ab) *2
            wvoa -= cp.einsum('il,al->ai', veff0mom_ab[:nocca,:noccb], xmy_ab) *2

        else:
            dm = (dmzoo_a, dmxpy_ab.T, -dmxmy_ab.T,
                  dmzoo_b, dmxpy_ab, dmxmy_ab)
            vj = mf.get_j(mol, dm, hermi=0).reshape(2,3,nao,nao)

            veff0doo = vj[0,0]+vj[1,0] + f1oo[:,0]
            veff0doo[0] += (k1ao_xpy[0,0,0] + k1ao_xpy[0,1,0] + k1ao_xpy[1,0,0] + k1ao_xpy[1,1,0]
                           +k1ao_xmy[0,0,0] + k1ao_xmy[0,1,0] + k1ao_xmy[1,0,0] + k1ao_xmy[1,1,0])
            veff0doo[1] += (k1ao_xpy[0,0,0] + k1ao_xpy[0,1,0] - k1ao_xpy[1,0,0] - k1ao_xpy[1,1,0]
                           +k1ao_xmy[0,0,0] + k1ao_xmy[0,1,0] - k1ao_xmy[1,0,0] - k1ao_xmy[1,1,0])

            wvoa = reduce(cp.dot, (orbva.T, veff0doo[0], orboa)) *2
            wvob = reduce(cp.dot, (orbvb.T, veff0doo[1], orbob)) *2

            veff = f1vo[0,:,0]
            veff0mop_ab = reduce(cp.dot, (mo_coeff[0].T, veff[1], mo_coeff[1]))

            wvob += cp.einsum('ca,ci->ai', veff0mop_ab[nocca:,noccb:], xpy_ab) *2
            wvoa -= cp.einsum('il,al->ai', veff0mop_ab[:nocca,:noccb], xpy_ab) *2

            veff = f1vo[1,:,0]
            veff0mom_ab = reduce(cp.dot, (mo_coeff[0].T, veff[1], mo_coeff[1]))

            wvob += cp.einsum('ca,ci->ai', veff0mom_ab[nocca:,noccb:], xmy_ab) *2
            wvoa -= cp.einsum('il,al->ai', veff0mom_ab[:nocca,:noccb], xmy_ab) *2
    elif td_grad.base.extype == 1: # spin-flip-down
        # x_ab, a means vira, b means occb
        x_ba, y_ba = x_y
        if not isinstance(x_ba, cp.ndarray):
            x_ba = cp.asarray(x_ba)
        xpy_ba = x_ba.T
        xmy_ba = x_ba.T

        dvv_b = cp.einsum('ai,bi->ab', xpy_ba, xpy_ba) + cp.einsum('ai,bi->ab', xmy_ba, xmy_ba) # T^{ab \beta \alpha}*2
        doo_a =-cp.einsum('ai,aj->ij', xpy_ba, xpy_ba) - cp.einsum('ai,aj->ij', xmy_ba, xmy_ba) # T^{ij \beta \alpha}*2

        dmxpy_ba = reduce(cp.dot, (orbvb, xpy_ba, orboa.T)) # ua ai iv -> uv -> (X+Y)_{uv \beta \alpha}
        dmxmy_ba = reduce(cp.dot, (orbvb, xmy_ba, orboa.T)) # ua ai iv -> uv -> (X-Y)_{uv \beta \alpha}

        dmzoo_a = reduce(cp.dot, (orboa, doo_a, orboa.T)) # \sum_{\sigma ab} 2*Tab \sigma C_{au} C_{bu}
        dmzoo_b = reduce(cp.dot, (orbvb, dvv_b, orbvb.T))

        ni = mf._numint
        ni.libxc.test_deriv_order(mf.xc, 3, raise_error=True)
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)

        # used by mcfun.
        rho0, vxc, fxc = ni.cache_xc_kernel(mf.mol, mf.grids, mf.xc,
                                        mo_coeff, mo_occ, spin=1)

        f1vo, f1oo, vxc1, k1ao = \
                _contract_xc_kernel(td_grad, mf.xc, (dmxpy_ba, dmxmy_ba),
                                    (dmzoo_a,dmzoo_b), True, True, max_memory, td_grad.base.extype)
        k1ao_xpy, k1ao_xmy = k1ao

        # f1vo, (2,2,4,nao,nao), (X+Y) and (X-Y) with fxc_sf
        # f1oo, (2,4,nao,nao), 2T with fxc_sc
        # vxc1, ao with v1^{\sigma}
        # k1ao_xpy，(2,2,4,nao,nao), (X+Y)(X+Y) and (X-Y)(X-Y) with gxc

        if abs(hyb) > 1e-10:
            dm = (dmzoo_a, dmxpy_ba, dmxmy_ba,
                  dmzoo_b, dmxpy_ba.T, -dmxmy_ba.T)
            vj, vk = mf.get_jk(mol, dm, hermi=0)
            if not isinstance(vj, cp.ndarray):
                vj = cp.asarray(vj)
            if not isinstance(vk, cp.ndarray):
                vk = cp.asarray(vk)
            vk *= hyb
            if abs(omega) > 1e-10:
                vk += mf.get_k(mol, dm, hermi=0, omega=omega) * (alpha-hyb)
            vj = vj.reshape(2,3,nao,nao)
            vk = vk.reshape(2,3,nao,nao)

            veff0doo = vj[0,0]+vj[1,0] - vk[:,0]+ f1oo[:,0]
            veff0doo[0] += (k1ao_xpy[0,0,0] + k1ao_xpy[0,1,0] + k1ao_xpy[1,0,0] + k1ao_xpy[1,1,0]
                           +k1ao_xmy[0,0,0] + k1ao_xmy[0,1,0] + k1ao_xmy[1,0,0] + k1ao_xmy[1,1,0])
            veff0doo[1] += (k1ao_xpy[0,0,0] + k1ao_xpy[0,1,0] - k1ao_xpy[1,0,0] - k1ao_xpy[1,1,0]
                           +k1ao_xmy[0,0,0] + k1ao_xmy[0,1,0] - k1ao_xmy[1,0,0] - k1ao_xmy[1,1,0])

            wvoa = reduce(cp.dot, (orbva.T, veff0doo[0], orboa)) *2
            wvob = reduce(cp.dot, (orbvb.T, veff0doo[1], orbob)) *2

            veff = - vk[:,1] + f1vo[0,:,0]
            veff0mop_ba = reduce(cp.dot, (mo_coeff[1].T, veff[0], mo_coeff[0]))

            wvoa += cp.einsum('ca,ci->ai', veff0mop_ba[noccb:,nocca:], xpy_ba) *2
            wvob -= cp.einsum('il,al->ai', veff0mop_ba[:noccb,:nocca], xpy_ba) *2

            veff = -vk[:,2] + f1vo[1,:,0]
            veff0mom_ba = reduce(cp.dot, (mo_coeff[1].T, veff[0], mo_coeff[0]))

            wvoa += cp.einsum('ca,ci->ai', veff0mom_ba[noccb:,nocca:], xmy_ba) *2
            wvob -= cp.einsum('il,al->ai', veff0mom_ba[:noccb,:nocca], xmy_ba) *2

        else:
            dm = (dmzoo_a, dmxpy_ba, dmxmy_ba,
                  dmzoo_b, dmxpy_ba.T, -dmxmy_ba.T)
            vj = mf.get_j(mol, dm, hermi=0).reshape(2,3,nao,nao)

            veff0doo = vj[0,0]+vj[1,0] + f1oo[:,0]
            veff0doo[0] += (k1ao_xpy[0,0,0] + k1ao_xpy[0,1,0] + k1ao_xpy[1,0,0] + k1ao_xpy[1,1,0]
                           +k1ao_xmy[0,0,0] + k1ao_xmy[0,1,0] + k1ao_xmy[1,0,0] + k1ao_xmy[1,1,0])
            veff0doo[1] += (k1ao_xpy[0,0,0] + k1ao_xpy[0,1,0] - k1ao_xpy[1,0,0] - k1ao_xpy[1,1,0]
                           +k1ao_xmy[0,0,0] + k1ao_xmy[0,1,0] - k1ao_xmy[1,0,0] - k1ao_xmy[1,1,0])

            wvoa = reduce(cp.dot, (orbva.T, veff0doo[0], orboa)) *2
            wvob = reduce(cp.dot, (orbvb.T, veff0doo[1], orbob)) *2

            veff = f1vo[0,:,0]
            veff0mop_ba = reduce(cp.dot, (mo_coeff[1].T, veff[0], mo_coeff[0]))

            wvoa += cp.einsum('ca,ci->ai', veff0mop_ba[noccb:,nocca:], xpy_ba) *2
            wvob -= cp.einsum('il,al->ai', veff0mop_ba[:noccb,:nocca], xpy_ba) *2

            veff = f1vo[1,:,0]
            veff0mom_ba = reduce(cp.dot, (mo_coeff[1].T, veff[0], mo_coeff[0]))

            wvoa += cp.einsum('ca,ci->ai', veff0mom_ba[noccb:,nocca:], xmy_ba) *2
            wvob -= cp.einsum('il,al->ai', veff0mom_ba[:noccb,:nocca], xmy_ba) *2
    else:
        raise RuntimeError("Only spin-flip UHF/UKS is supported")

    vresp = mf.gen_response(hermi=1)

    def fvind(x):
        dm1 = cp.empty((2,nao,nao))
        x_a = x[0,:nvira*nocca].reshape(nvira,nocca)
        x_b = x[0,nvira*nocca:].reshape(nvirb,noccb)
        dm_a = reduce(cp.dot, (orbva, x_a, orboa.T))
        dm_b = reduce(cp.dot, (orbvb, x_b, orbob.T))
        dm1[0] = (dm_a + dm_a.T).real
        dm1[1] = (dm_b + dm_b.T).real

        v1 = vresp(dm1)
        v1a = reduce(cp.dot, (orbva.T, v1[0], orboa))
        v1b = reduce(cp.dot, (orbvb.T, v1[1], orbob))
        return cp.hstack((v1a.ravel(), v1b.ravel()))

    z1a, z1b = ucphf.solve(fvind, mo_energy, mo_occ, (wvoa,wvob),
                           max_cycle=td_grad.cphf_max_cycle,
                           tol=td_grad.cphf_conv_tol)[0]

    time1 = log.timer('Z-vector using UCPHF solver', *time0)

    z1ao = cp.zeros((2,nao,nao))
    z1ao[0] += reduce(cp.dot, (orbva, z1a, orboa.T))
    z1ao[1] += reduce(cp.dot, (orbvb, z1b, orbob.T))

    veff = vresp((z1ao+z1ao.transpose(0,2,1))*0.5)

    im0a = cp.zeros((nmoa,nmoa))
    im0b = cp.zeros((nmob,nmob))
    if td_grad.base.extype == 0:
        im0a[:nocca,:nocca] = reduce(cp.dot, (orboa.T, veff0doo[0]+veff[0], orboa)) *.5
        im0b[:noccb,:noccb] = reduce(cp.dot, (orbob.T, veff0doo[1]+veff[1], orbob)) *.5
        im0b[:noccb,:noccb] += cp.einsum('aj,ai->ij', veff0mop_ab[nocca:,:noccb], xpy_ab) *0.5
        im0b[:noccb,:noccb] += cp.einsum('aj,ai->ij', veff0mom_ab[nocca:,:noccb], xmy_ab) *0.5

        im0a[nocca:,nocca:]  = cp.einsum('bi,ai->ab', veff0mop_ab[nocca:,:noccb], xpy_ab) *0.5
        im0a[nocca:,nocca:] += cp.einsum('bi,ai->ab', veff0mom_ab[nocca:,:noccb], xmy_ab) *0.5

        im0a[nocca:,:nocca]  = cp.einsum('il,al->ai', veff0mop_ab[:nocca,:noccb], xpy_ab)
        im0a[nocca:,:nocca] += cp.einsum('il,al->ai', veff0mom_ab[:nocca,:noccb], xmy_ab)
    elif td_grad.base.extype == 1:
        im0a[:nocca,:nocca] = reduce(cp.dot, (orboa.T, veff0doo[0]+veff[0], orboa)) *.5
        im0b[:noccb,:noccb] = reduce(cp.dot, (orbob.T, veff0doo[1]+veff[1], orbob)) *.5
        im0a[:nocca,:nocca] += cp.einsum('aj,ai->ij', veff0mop_ba[noccb:,:nocca], xpy_ba) *0.5
        im0a[:nocca,:nocca] += cp.einsum('aj,ai->ij', veff0mom_ba[noccb:,:nocca], xmy_ba) *0.5

        im0b[noccb:,noccb:]  = cp.einsum('bi,ai->ab', veff0mop_ba[noccb:,:nocca], xpy_ba) *0.5
        im0b[noccb:,noccb:] += cp.einsum('bi,ai->ab', veff0mom_ba[noccb:,:nocca], xmy_ba) *0.5

        im0b[noccb:,:noccb]  = cp.einsum('il,al->ai', veff0mop_ba[:noccb,:nocca], xpy_ba)
        im0b[noccb:,:noccb] += cp.einsum('il,al->ai', veff0mom_ba[:noccb,:nocca], xmy_ba)

    zeta_a = (mo_energy[0][:,None] + mo_energy[0]) * .5
    zeta_b = (mo_energy[1][:,None] + mo_energy[1]) * .5
    zeta_a[nocca:,:nocca] = mo_energy[0][:nocca]
    zeta_b[noccb:,:noccb] = mo_energy[1][:noccb]
    zeta_a[:nocca,nocca:] = mo_energy[0][nocca:]
    zeta_b[:noccb,noccb:] = mo_energy[1][noccb:]

    dm1a = cp.zeros((nmoa,nmoa))
    dm1b = cp.zeros((nmob,nmob))
    if td_grad.base.extype == 0:
        dm1b[:noccb,:noccb] = doo_b * .5
        dm1a[nocca:,nocca:] = dvv_a * .5
    elif td_grad.base.extype == 1:
        dm1a[:nocca,:nocca] = doo_a * .5
        dm1b[noccb:,noccb:] = dvv_b * .5

    dm1a[nocca:,:nocca] = z1a *.5
    dm1b[noccb:,:noccb] = z1b *.5

    dm1a[:nocca,:nocca] += cp.eye(nocca) # for ground state
    dm1b[:noccb,:noccb] += cp.eye(noccb)

    im0a = reduce(cp.dot, (mo_coeff[0], im0a+zeta_a*dm1a, mo_coeff[0].T))
    im0b = reduce(cp.dot, (mo_coeff[1], im0b+zeta_b*dm1b, mo_coeff[1].T))
    im0 = im0a + im0b

    dmz1dooa = z1ao[0] + dmzoo_a
    dmz1doob = z1ao[1] + dmzoo_b
    oo0a = reduce(cp.dot, (orboa, orboa.T))
    oo0b = reduce(cp.dot, (orbob, orbob.T))

    mf_grad = mf.nuc_grad_method().to_cpu()
    h1 = cp.asarray(mf_grad.get_hcore(mol))  # without 1/r like terms
    s1 = cp.asarray(mf_grad.get_ovlp(mol))
    dh_ground = contract("xij,ij->xi", h1, oo0a + oo0b)
    dh_td = contract("xij,ij->xi", h1, (dmz1dooa + dmz1doob) * 0.25 + (dmz1dooa + dmz1doob).T * 0.25)
    ds = contract("xij,ij->xi", s1, (im0 + im0.T) * 0.5)

    dh1e_ground = int3c2e.get_dh1e(mol, oo0a + oo0b)  # 1/r like terms
    if mol.has_ecp():
        dh1e_ground += rhf_grad.get_dh1e_ecp(mol, oo0a + oo0b)  # 1/r like terms
    dh1e_td = int3c2e.get_dh1e(mol, (dmz1dooa + dmz1doob) * 0.25 + (dmz1dooa + dmz1doob).T * 0.25)  # 1/r like terms
    if mol.has_ecp():
        dh1e_td += rhf_grad.get_dh1e_ecp(
            mol, (dmz1dooa + dmz1doob) * 0.25 + (dmz1dooa + dmz1doob).T * 0.25)  # 1/r like terms
    if atmlst is None:
        atmlst = range(mol.natm)
    dvhf_all = 0
    j_factor = 1.0
    k_factor = 0.0
    with_k = ni.libxc.is_hybrid_xc(mf.xc)
    if with_k:
        k_factor = hyb
    dvhf = td_grad.get_veff(mol, cp.stack(((dmz1dooa + dmz1dooa.T) * 0.25 + oo0a,
                                           (dmz1doob + dmz1doob.T) * 0.25 + oo0b,)), j_factor, k_factor)
    dvhf_all += dvhf
    dvhf = td_grad.get_veff(mol, cp.stack(((dmz1dooa + dmz1dooa.T), (dmz1doob + dmz1doob.T))) * 0.25,
        j_factor, k_factor)
    dvhf_all -= dvhf
    if td_grad.base.extype == 0:
        dvhf = td_grad.get_veff(mol, cp.stack(((dmxpy_ab + dmxpy_ab.T), (dmxpy_ab + dmxpy_ab.T))) * 0.5,
            0.0, k_factor)
        dvhf_all += dvhf * 1
        dvhf = td_grad.get_veff(mol, cp.stack(((dmxmy_ab - dmxmy_ab.T), (dmxmy_ab - dmxmy_ab.T))) * 0.5,
                j_factor=0.0, k_factor=k_factor, hermi=2)
        dvhf_all += dvhf * 1

    elif td_grad.base.extype == 1:
        dvhf = td_grad.get_veff(mol, cp.stack(((dmxpy_ba + dmxpy_ba.T), (dmxpy_ba.T + dmxpy_ba))) * 0.5,
            0.0, k_factor)
        dvhf_all += dvhf * 1
        dvhf = td_grad.get_veff(mol, cp.stack(((dmxmy_ba - dmxmy_ba.T), (-dmxmy_ba.T + dmxmy_ba))) * 0.5,
                j_factor=0.0, k_factor=k_factor, hermi=2)
        dvhf_all += dvhf * 1

    fxcz1 = _contract_xc_kernel_z(td_grad, mf.xc, z1ao, max_memory)
    veff1 = cp.zeros((2,4,3,nao,nao))
    veff1[:,0] += vxc1[:,1:]
    veff1[:,1] += (f1oo[:,1:] + fxcz1[:,1:])*2
    veff1[0,1] += (k1ao_xpy[0,0,1:] + k1ao_xpy[0,1,1:] + k1ao_xpy[1,0,1:] + k1ao_xpy[1,1,1:]
                  +k1ao_xmy[0,0,1:] + k1ao_xmy[0,1,1:] + k1ao_xmy[1,0,1:] + k1ao_xmy[1,1,1:])*2
    veff1[1,1] += (k1ao_xpy[0,0,1:] + k1ao_xpy[0,1,1:] - k1ao_xpy[1,0,1:] - k1ao_xpy[1,1,1:]
                  +k1ao_xmy[0,0,1:] + k1ao_xmy[0,1,1:] - k1ao_xmy[1,0,1:] - k1ao_xmy[1,1,1:])*2

    veff1[:,2] += f1vo[0,:,1:]
    veff1[:,3] += f1vo[1,:,1:]
    veff1a, veff1b = veff1
    time1 = log.timer('2e AO integral derivatives', *time1)

    if atmlst is None:
        atmlst = range(mol.natm)
    offsetdic = mol.offset_nr_by_atom()
    de = cp.zeros((len(atmlst),3))
    delec = 2.0 * (dh_ground + dh_td - ds)
    aoslices = mol.aoslice_by_atom()
    delec = cp.asarray([cp.sum(delec[:, p0:p1], axis=1) for p0, p1 in aoslices[:, 2:]])

    deveff0 = cp.asarray(
    [contract("xpq,pq->x", veff1a[0,:,p0:p1], oo0a[p0:p1] + dmz1dooa[p0:p1] * 0.5)
     for p0, p1 in aoslices[:, 2:]])
    deveff0 += cp.asarray(
        [contract("xpq,pq->x", veff1b[0,:,p0:p1], oo0b[p0:p1] + dmz1doob[p0:p1] * 0.5)
        for p0, p1 in aoslices[:, 2:]])
    deveff0 += cp.asarray(
        [contract("xpq,qp->x", veff1a[0,:,p0:p1], oo0a[:,p0:p1] + dmz1dooa[:,p0:p1] * 0.5)
        for p0, p1 in aoslices[:, 2:]])
    deveff0 += cp.asarray(
        [contract("xpq,qp->x", veff1b[0,:,p0:p1], oo0b[:,p0:p1] + dmz1doob[:,p0:p1] * 0.5)
        for p0, p1 in aoslices[:, 2:]])

    deveff1 = cp.asarray(
        [contract("xpq,pq->x", veff1a[1,:,p0:p1], oo0a[p0:p1] * 0.5)
        for p0, p1 in aoslices[:, 2:]])
    deveff1 += cp.asarray(
        [contract("xpq,pq->x", veff1b[1,:,p0:p1], oo0b[p0:p1] * 0.5)
        for p0, p1 in aoslices[:, 2:]])

    if td_grad.base.extype == 0:
        deveff2 = cp.asarray(
            [contract('xpq,pq->x', veff1b[2,:,p0:p1], dmxpy_ab[p0:p1,:])
            for p0, p1 in aoslices[:, 2:]])
        deveff2 += cp.asarray(
            [contract('xqp,pq->x', veff1b[2,:,p0:p1], dmxpy_ab[:,p0:p1])  
            for p0, p1 in aoslices[:, 2:]])
        deveff3 = cp.asarray(
            [contract('xpq,pq->x', veff1b[3,:,p0:p1], dmxmy_ab[p0:p1,:])
            for p0, p1 in aoslices[:, 2:]])
        deveff3 += cp.asarray(
            [contract('xqp,pq->x', veff1b[3,:,p0:p1], dmxmy_ab[:,p0:p1])  
            for p0, p1 in aoslices[:, 2:]])
    elif td_grad.base.extype == 1:
        deveff2 = cp.asarray(
            [contract('xpq,pq->x', veff1a[2,:,p0:p1], dmxpy_ba[p0:p1,:])
            for p0, p1 in aoslices[:, 2:]])
        deveff2 += cp.asarray(
            [contract('xqp,pq->x', veff1a[2,:,p0:p1], dmxpy_ba[:,p0:p1])  
            for p0, p1 in aoslices[:, 2:]])
        deveff3 = cp.asarray(
            [contract('xpq,pq->x', veff1a[3,:,p0:p1], dmxmy_ba[p0:p1,:])
            for p0, p1 in aoslices[:, 2:]])
        deveff3 += cp.asarray(
            [contract('xqp,pq->x', veff1a[3,:,p0:p1], dmxmy_ba[:,p0:p1])  
            for p0, p1 in aoslices[:, 2:]])

    de += 2.0 * dvhf_all + delec + dh1e_ground + dh1e_td + deveff0 + deveff1 + deveff2 + deveff3
    log.timer('TDUKS nuclear gradients', *time0)
    return de.get()

def _contract_xc_kernel(td_grad, xc_code, dmvo, dmoo=None, with_vxc=True,
                        with_kxc=True, max_memory=2000, extype=0):
    mol = td_grad.mol
    mf = td_grad.base._scf
    grids = mf.grids

    ni = mf._numint
    xctype = ni._xc_type(xc_code)

    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    nao = mo_coeff[0].shape[0]
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    opt = getattr(ni, "gdftopt", None)
    if opt is None:
        ni.build(mol, grids.coords)
        opt = ni.gdftopt
    _sorted_mol = opt._sorted_mol
    mo_coeff = opt.sort_orbitals(mo_coeff, axis=[1])

    f1vo = cp.zeros((2,2,4,nao,nao))
    deriv = 2

    if dmoo is not None:
        f1oo = cp.zeros((2,4,nao,nao))
        dmoo0 = opt.sort_orbitals(dmoo[0], axis=[0, 1])
        dmoo1 = opt.sort_orbitals(dmoo[1], axis=[0, 1])
    else:
        f1oo = None
    if with_vxc:
        v1ao = cp.zeros((2,4,nao,nao))
    else:
        v1ao = None
    if with_kxc:
        k1ao_xpy = cp.zeros((2,2,4,nao,nao))
        k1ao_xmy = cp.zeros((2,2,4,nao,nao))
        deriv = 3
    else:
        k1ao_xpy = k1ao_xmy = None

    dmvo0 = opt.sort_orbitals(dmvo[0], axis=[0, 1])
    dmvo1 = opt.sort_orbitals(dmvo[1], axis=[0, 1])

    # # create a mc object to use mcfun.
    # nimc = numint2c.NumInt2C()
    # nimc.collinear = 'mcol'
    # nimc.collinear_samples=td_grad.base.collinear_samples

    # calculate the derivatives.
    # fxc_sf,kxc_sf = cache_xc_kernel_sf(ni,mol,mf.grids,mf.xc,mo_coeff,mo_occ,
        # td_grad.base.collinear_samples,deriv=3)[2:]
    p0,p1=0,0 # the two parameters are used for counts the batch of grids.

    if xctype == "LDA":
        fmat_, ao_deriv = tdrks._lda_eval_mat_, 1
    elif xctype == "GGA":
        fmat_, ao_deriv = _gga_eval_mat_, 2
    elif xctype == "MGGA":
        fmat_, ao_deriv = tdrks._mgga_eval_mat_, 2

    if xctype == 'LDA':
        for ao, mask, weight, coords \
                in ni.block_loop(_sorted_mol, grids, nao, ao_deriv):
            p0 = p1
            p1+= weight.shape[0]

            mo_coeff_mask_a = mo_coeff[0, mask]
            mo_coeff_mask_b = mo_coeff[1, mask]
            dmvo0_mask = dmvo0[mask[:, None], mask]
            dmvo1_mask = dmvo1[mask[:, None], mask]
            with_lapl = False
            # fxc_sf, kxc_sf = cache_xc_kernel_sf(ni,mol,mf.grids,mf.xc,mo_coeff,mo_occ,
                # td_grad.base.collinear_samples,deriv=3)[2:]
            rhoa_slice = ni.eval_rho2(_sorted_mol, ao, mo_coeff_mask_a,
                                mo_occ[0], None, xctype, with_lapl)
            rhob_slice = ni.eval_rho2(_sorted_mol, ao, mo_coeff_mask_b,
                                mo_occ[1], None, xctype, with_lapl)
            rho_ab = (rhoa_slice, rhob_slice)
            rho_z = cp.array([rho_ab[0]+rho_ab[1],
                            rho_ab[0]-rho_ab[1]])
            eval_xc_eff = mcfun_eval_xc_adapter_sf(ni, xc_code, td_grad.base.collinear_samples)
            fxc_sf, kxc_sf = eval_xc_eff(xc_code, rho_z, deriv=3, xctype=xctype)[2:4]
            s_s = fxc_sf * weight


            if extype == 0:
                rho1_ab = ni.eval_rho(_sorted_mol, ao[0], dmvo0_mask, mask, xctype)
                # s_s*2 because of \sigma_x \sigma_x + \sigma_y \sigma_y
                fmat_(_sorted_mol, f1vo[0][1], ao, rho1_ab*s_s*2, mask, shls_slice, ao_loc)
                fmat_(_sorted_mol, f1vo[0][0], ao, rho1_ab*s_s*2, mask, shls_slice, ao_loc)
                # lda_sum_(f1vo[0][1], ao, rho1_ab*s_s*2, mask)
                # lda_sum_(f1vo[0][0], ao, rho1_ab*s_s*2, mask)

                if with_kxc:
                    s_s_n = kxc_sf[:,:,0] * weight
                    s_s_s = kxc_sf[:,:,1] * weight
                    fmat_(_sorted_mol, k1ao_xpy[0][0], ao, s_s_n*2*rho1_ab*rho1_ab, mask, shls_slice, ao_loc)
                    fmat_(_sorted_mol, k1ao_xpy[1][0], ao, s_s_s*2*rho1_ab*rho1_ab, mask, shls_slice, ao_loc)

                rho1_ab = ni.eval_rho(_sorted_mol, ao[0], dmvo1_mask, mask, xctype)

                # py attention to the order of f1vo[1][1] and f1vo[1][0]
                fmat_(_sorted_mol, f1vo[1][1], ao, rho1_ab*s_s*2, mask, shls_slice, ao_loc)
                fmat_(_sorted_mol, f1vo[1][0], ao, -rho1_ab*s_s*2, mask, shls_slice, ao_loc)

                if with_kxc:
                    # Note the "-"
                    fmat_(_sorted_mol, k1ao_xmy[0][0], ao, s_s_n*2*rho1_ab*rho1_ab, mask, shls_slice, ao_loc)
                    fmat_(_sorted_mol, k1ao_xmy[1][0], ao, s_s_s*2*rho1_ab*rho1_ab, mask, shls_slice, ao_loc)
            elif extype == 1:
                rho1_ba = ni.eval_rho(_sorted_mol, ao[0], dmvo0_mask, mask, xctype)
                # s_s*2 because of \sigma_x \sigma_x + \sigma_y \sigma_y
                fmat_(_sorted_mol, f1vo[0][1], ao, rho1_ba*s_s*2, mask, shls_slice, ao_loc)
                fmat_(_sorted_mol, f1vo[0][0], ao, rho1_ba*s_s*2, mask, shls_slice, ao_loc)

                if with_kxc:
                    s_s_n = kxc_sf[:,:,0] * weight
                    s_s_s = kxc_sf[:,:,1] * weight
                    fmat_(_sorted_mol, k1ao_xpy[0][1], ao, s_s_n*2*rho1_ba*rho1_ba, mask, shls_slice, ao_loc)
                    fmat_(_sorted_mol, k1ao_xpy[1][1], ao, s_s_s*2*rho1_ba*rho1_ba, mask, shls_slice, ao_loc)

                rho1_ba = ni.eval_rho(_sorted_mol, ao[0], dmvo1_mask, mask, xctype)

                # py attention to the order of f1vo[1][1] and f1vo[1][0]
                fmat_(_sorted_mol, f1vo[1][1], ao, -rho1_ba*s_s*2, mask, shls_slice, ao_loc)
                fmat_(_sorted_mol, f1vo[1][0], ao, rho1_ba*s_s*2, mask, shls_slice, ao_loc)

                if with_kxc:
                    # Note the "-"
                    fmat_(_sorted_mol, k1ao_xmy[0][1], ao, s_s_n*2*rho1_ba*rho1_ba, mask, shls_slice, ao_loc)
                    fmat_(_sorted_mol, k1ao_xmy[1][1], ao, s_s_s*2*rho1_ba*rho1_ba, mask, shls_slice, ao_loc)

            rho = (ni.eval_rho2(_sorted_mol, ao[0],mo_coeff_mask_a, mo_occ[0], mask, xctype),
                ni.eval_rho2(_sorted_mol, ao[0], mo_coeff_mask_b, mo_occ[1], mask, xctype))
            vxc, fxc, kxc = ni.eval_xc_eff(xc_code, rho, deriv=deriv, spin=1)[1:]
            u_u, u_d, d_d = fxc[0].T * weight
            if dmoo is not None:
                dmoo0_mask = dmoo0[mask[:, None], mask]
                dmoo1_mask = dmoo1[mask[:, None], mask]
                rho2a = ni.eval_rho(_sorted_mol, ao[0], dmoo0_mask, mask, xctype, hermi=1)
                rho2b = ni.eval_rho(_sorted_mol, ao[0], dmoo1_mask, mask, xctype, hermi=1)
                fmat_(_sorted_mol, f1oo[0], ao, u_u*rho2a+u_d*rho2b, mask, shls_slice, ao_loc)
                fmat_(_sorted_mol, f1oo[1], ao, u_d*rho2a+d_d*rho2b, mask, shls_slice, ao_loc)
            if with_vxc:
                vrho = vxc[0].T * weight
                fmat_(_sorted_mol, v1ao[0], ao, vrho[0], mask, shls_slice, ao_loc)
                fmat_(_sorted_mol, v1ao[1], ao, vrho[1], mask, shls_slice, ao_loc)

    elif xctype == 'GGA':
        for ao, mask, weight, coords \
                in ni.block_loop(_sorted_mol, grids, nao, ao_deriv, max_memory):
            p0 = p1
            p1+= weight.shape[0]

            mo_coeff_mask_a = mo_coeff[0, mask]
            mo_coeff_mask_b = mo_coeff[1, mask]
            dmvo0_mask = dmvo0[mask[:, None], mask]
            dmvo1_mask = dmvo1[mask[:, None], mask]

            with_lapl = False
            # fxc_sf, kxc_sf = cache_xc_kernel_sf(ni,mol,mf.grids,mf.xc,mo_coeff,mo_occ,
                # td_grad.base.collinear_samples,deriv=3)[2:]
            rhoa_slice = ni.eval_rho2(_sorted_mol, ao, mo_coeff_mask_a,
                                mo_occ[0], None, xctype, with_lapl)
            rhob_slice = ni.eval_rho2(_sorted_mol, ao, mo_coeff_mask_b,
                                mo_occ[1], None, xctype, with_lapl)
            rho_ab = (rhoa_slice, rhob_slice)
            rho_z = cp.array([rho_ab[0]+rho_ab[1],
                            rho_ab[0]-rho_ab[1]])
            eval_xc_eff = mcfun_eval_xc_adapter_sf(ni, xc_code, td_grad.base.collinear_samples)
            fxc_sf, kxc_sf = eval_xc_eff(xc_code, rho_z, deriv=3, xctype=xctype)[2:4]

            if extype == 0:
                rho1_ab = ni.eval_rho(_sorted_mol, ao, dmvo0_mask, mask, xctype, hermi=0, with_lapl=False)

                wv_sf = uks_sf_gga_wv1(rho1_ab,fxc_sf,weight, extype)
                fmat_(_sorted_mol, f1vo[0][1], ao, wv_sf, mask, shls_slice, ao_loc)
                fmat_(_sorted_mol, f1vo[0][0], ao, wv_sf, mask, shls_slice, ao_loc)

                if with_kxc:
                    gv_sf = uks_sf_gga_wv2_p(rho1_ab,kxc_sf,weight, extype)
                    fmat_(_sorted_mol, k1ao_xpy[0][0], ao, gv_sf[0], mask, shls_slice, ao_loc)
                    fmat_(_sorted_mol, k1ao_xpy[1][0], ao, gv_sf[1], mask, shls_slice, ao_loc)

                rho1_ab = ni.eval_rho(_sorted_mol, ao, dmvo1_mask, mask, xctype, hermi=0, with_lapl=False)

                wv_sf = uks_sf_gga_wv1(rho1_ab,fxc_sf,weight, extype)
                fmat_(_sorted_mol, f1vo[1][1], ao, wv_sf, mask, shls_slice, ao_loc)
                fmat_(_sorted_mol, f1vo[1][0], ao, -wv_sf, mask, shls_slice, ao_loc)

                if with_kxc:
                    gv_sf = uks_sf_gga_wv2_m(rho1_ab,kxc_sf,weight, extype)
                    fmat_(_sorted_mol, k1ao_xmy[0][0], ao, gv_sf[0], mask, shls_slice, ao_loc)
                    fmat_(_sorted_mol, k1ao_xmy[1][0], ao, gv_sf[1], mask, shls_slice, ao_loc)
            elif extype == 1:
                rho1_ba = ni.eval_rho(_sorted_mol, ao, dmvo0_mask, mask, xctype, hermi=0, with_lapl=False)
                wv_sf = uks_sf_gga_wv1(rho1_ba,fxc_sf,weight, extype)
                fmat_(_sorted_mol, f1vo[0][1], ao, wv_sf, mask, shls_slice, ao_loc)
                fmat_(_sorted_mol, f1vo[0][0], ao, wv_sf, mask, shls_slice, ao_loc)

                if with_kxc:
                    gv_sf = uks_sf_gga_wv2_p(rho1_ba,kxc_sf,weight, extype)
                    fmat_(_sorted_mol, k1ao_xpy[0][1], ao, gv_sf[0], mask, shls_slice, ao_loc)
                    fmat_(_sorted_mol, k1ao_xpy[1][1], ao, gv_sf[1], mask, shls_slice, ao_loc)

                rho1_ba = ni.eval_rho(_sorted_mol, ao, dmvo1_mask, mask, xctype, hermi=0, with_lapl=False)

                wv_sf = uks_sf_gga_wv1(rho1_ba,fxc_sf,weight, extype)
                fmat_(_sorted_mol, f1vo[1][1], ao, -wv_sf, mask, shls_slice, ao_loc)
                fmat_(_sorted_mol, f1vo[1][0], ao, wv_sf, mask, shls_slice, ao_loc)

                if with_kxc:
                    gv_sf = uks_sf_gga_wv2_m(rho1_ba,kxc_sf,weight, extype)
                    fmat_(_sorted_mol, k1ao_xmy[0][1], ao, gv_sf[0], mask, shls_slice, ao_loc)
                    fmat_(_sorted_mol, k1ao_xmy[1][1], ao, gv_sf[1], mask, shls_slice, ao_loc)

            rho = cp.stack([ni.eval_rho2(_sorted_mol, ao, mo_coeff_mask_a, mo_occ[0], mask, xctype),
                            ni.eval_rho2(_sorted_mol, ao, mo_coeff_mask_b, mo_occ[1], mask, xctype)])
            vxc, fxc, kxc = ni.eval_xc_eff(xc_code, rho, deriv=deriv, spin=1)[1:]
            if dmoo is not None:
                dmoo0_mask = dmoo0[mask[:, None], mask]
                dmoo1_mask = dmoo1[mask[:, None], mask]
                rho2 = cp.stack([ni.eval_rho(_sorted_mol, ao, dmoo0_mask, mask, xctype, hermi=1),
                                 ni.eval_rho(_sorted_mol, ao, dmoo1_mask, mask, xctype, hermi=1)])
                # wv = numint._uks_gga_wv1(rho, rho2, vxc, fxc, weight)
                tmp = contract("axg,axbyg->byg", rho2, fxc)
                wv = contract("byg,g->byg", tmp, weight)
                wv[:,0] *= 0.5
                fmat_(_sorted_mol, f1oo[0], ao, wv[0], mask, shls_slice, ao_loc)
                fmat_(_sorted_mol, f1oo[1], ao, wv[1], mask, shls_slice, ao_loc)
            if with_vxc:
                # wv = numint._uks_gga_wv0(rho, vxc, weight)
                wv = vxc * weight
                wv[:,0] *= 0.5
                fmat_(_sorted_mol, v1ao[0], ao, wv[0], mask, shls_slice, ao_loc)
                fmat_(_sorted_mol, v1ao[1], ao, wv[1], mask, shls_slice, ao_loc)

    elif xctype == 'MGGA':
        raise NotImplementedError('MGGA not implemented')

    else:
        raise NotImplementedError(f'td-uks for functional {xc_code}')

    f1vo[:,:,1:] *= -1
    f1vo = opt.unsort_orbitals(f1vo, axis=[3, 4])
    if f1oo is not None: 
        f1oo[:,1:] *= -1
        f1oo = opt.unsort_orbitals(f1oo, axis=[2, 3])
    if v1ao is not None: 
        v1ao[:,1:] *= -1
        v1ao = opt.unsort_orbitals(v1ao, axis=[2, 3])
    if with_kxc:
        k1ao_xpy[:,:,1:] *= -1
        k1ao_xmy[:,:,1:] *= -1
        k1ao_xpy = opt.unsort_orbitals(k1ao_xpy, axis=[3, 4])
        k1ao_xmy = opt.unsort_orbitals(k1ao_xmy, axis=[3, 4])

    return f1vo, f1oo, v1ao, (k1ao_xpy,k1ao_xmy)

def _gga_eval_mat_(mol, vmat, ao, wv, mask, shls_slice, ao_loc):
    # wv[0] *= 0.5  # *.5 because vmat + vmat.T at the end
    aow = numint._scale_ao(ao[:4], wv[:4])
    tmp = numint._dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
    vtmp = tmp + tmp.T
    add_sparse(vmat[0], vtmp, mask)
    wv = cp.asarray(wv, order="C")
    vtmp = rks_grad._gga_grad_sum_(ao, wv)
    add_sparse(vmat[1:], vtmp, mask)
    return vmat

def uks_sf_gga_wv1(rho1, fxc_sf, weight, extype):
    # fxc_sf with a shape (4,4,ngrid), 4 means I, \nabla_x,y,z.
    if extype == 0:
        ngrid = weight.shape[-1]
        wv_ab = cp.empty((4,ngrid))
        wv_ab = cp.einsum('yp,xyp->xp',  rho1,fxc_sf)

        # Don't forget (sigma_x sigma_x + sigma_y sigma_y) needs *2 for kernel term.
        wv_ab[1:] *=2.0
        return wv_ab*weight
    elif extype == 1:
        ngrid = weight.shape[-1]
        wv_ba = cp.empty((4,ngrid))
        wv_ba = cp.einsum('yp,xyp->xp',  rho1,fxc_sf)

        # Don't forget (sigma_x sigma_x + sigma_y sigma_y) needs *2 for kernel term.
        wv_ba[1:] *=2.0
        return wv_ba*weight
    # rho1_ab,rho1_ba = rho1
    # ngrid = weight.shape[-1]
    # wv_ab, wv_ba = cp.empty((2,4,ngrid))
    # wv_ab = cp.einsum('yp,xyp->xp',  rho1_ab,fxc_sf)
    # wv_ba = cp.einsum('yp,xyp->xp',  rho1_ba,fxc_sf)
    # # wv_ab[0] = wv_ab[0] *2 *.5 # *2 bacause of kernel, *0.5 for the (x + x.T)*0.5
    # # wv_ba[0] = wv_ba[0] *2 *.5

    # # Don't forget (sigma_x sigma_x + sigma_y sigma_y) needs *2 for kernel term.
    # wv_ab[1:] *=2.0
    # wv_ba[1:] *=2.0
    # return wv_ab*weight, wv_ba*weight

def uks_sf_gga_wv2_p(rho1, kxc_sf, weight, extype):
    # kxc_sf with a shape (4,4,2,4,ngrid), 4 means I,\nabla_x,y,z,
    # 0: n, \nabla_x,y,z n;  1: s, \nabla_x,y,z s.
    if extype == 0:
        ngrid = weight.shape[-1]
        gv_ab = cp.empty((2,4,ngrid))
        # Note *2 and *0.5 like in function uks_sf_gga_wv1
        gv_ab = cp.einsum('xp,yp,xyvzp->vzp', rho1, rho1, kxc_sf, optimize=True)

        gv_ab[0,1:] *=2.0
        gv_ab[1,1:] *=2.0
        return gv_ab*weight
    elif extype == 1:
        ngrid = weight.shape[-1]
        gv_ba = cp.empty((2,4,ngrid))
        # Note *2 and *0.5 like in function uks_sf_gga_wv1
        gv_ba = cp.einsum('xp,yp,xyvzp->vzp', rho1, rho1, kxc_sf, optimize=True)

        gv_ba[0,1:] *=2.0
        gv_ba[1,1:] *=2.0
        return gv_ba*weight
    # rho1_ab,rho1_ba = rho1
    # ngrid = weight.shape[-1]
    # gv_ab, gv_ba = cp.empty((2,2,4,ngrid))
    # # Note *2 and *0.5 like in function uks_sf_gga_wv1
    # gv_ab = cp.einsum('xp,yp,xyvzp->vzp', rho1_ab, rho1_ab+rho1_ba, kxc_sf, optimize=True)
    # gv_ba = cp.einsum('xp,yp,xyvzp->vzp', rho1_ba, rho1_ba+rho1_ab, kxc_sf, optimize=True)

    # gv_ab[0,1:] *=2.0
    # gv_ab[1,1:] *=2.0
    # gv_ba[0,1:] *=2.0
    # gv_ba[1,1:] *=2.0
    # return gv_ab*weight, gv_ba*weight

def uks_sf_gga_wv2_m(rho1, kxc_sf,weight, extype):
    if extype == 0:
        ngrid = weight.shape[-1]
        gv_ab = cp.empty((2,5,ngrid))
        # Note *2 and *0.5 like in function uks_sf_mgga_wv1
        gv_ab = cp.einsum('xp,yp,xyvzp->vzp', rho1, rho1, kxc_sf , optimize=True)

        gv_ab[:,1:] *=2.0
        return gv_ab*weight
    elif extype == 1:
        ngrid = weight.shape[-1]
        gv_ba = cp.empty((2,5,ngrid))
        # Note *2 and *0.5 like in function uks_sf_mgga_wv1
        gv_ba = cp.einsum('xp,yp,xyvzp->vzp', rho1, rho1, kxc_sf , optimize=True)

        gv_ba[:,1:] *=2.0
        return gv_ba*weight
    # rho1_ab,rho1_ba = rho1
    # ngrid = weight.shape[-1]
    # gv_ab, gv_ba = cp.empty((2,2,5,ngrid))
    # # Note *2 and *0.5 like in function uks_sf_mgga_wv1
    # gv_ab = cp.einsum('xp,yp,xyvzp->vzp', rho1_ab, rho1_ab-rho1_ba, kxc_sf , optimize=True)
    # gv_ba = cp.einsum('xp,yp,xyvzp->vzp', rho1_ba, rho1_ba-rho1_ab, kxc_sf , optimize=True)

    # gv_ab[:,1:] *=2.0
    # gv_ba[:,1:] *=2.0
    # return gv_ab*weight, gv_ba*weight

def uks_sf_mgga_wv1(rho1, fxc_sf,weight):
    rho1_ab,rho1_ba = rho1
    # fxc_sf with a shape (5,5,ngrid), 5 means I, \nabla_x,y,z s, u
    # s_s, s_Ns, Ns_s, Ns_Ns, s_u, u_s, u_Ns, Ns_u, u_u
    ngrid = weight.shape[-1]
    wv_ab, wv_ba = cp.empty((2,5,ngrid))
    wv_ab = cp.einsum('yp,xyp->xp',  rho1_ab,fxc_sf)
    wv_ba = cp.einsum('yp,xyp->xp',  rho1_ba,fxc_sf)
    # wv_ab[0] = wv_ab[0] *2 *.5 # *2 bacause of kernel, *0.5 for the (x + x.T)*0.5
    # wv_ba[0] = wv_ba[0] *2 *.5

    # Don't forget (sigma_x sigma_x + sigma_y sigma_y) needs *2 for kernel term.
    wv_ab[1:4] *=2.0
    wv_ba[1:4] *=2.0
    # *0.5 below is for tau->ao
    wv_ab[4] *= 0.5
    wv_ba[4] *= 0.5
    return wv_ab*weight, wv_ba*weight

def uks_sf_mgga_wv2_p(rho1, kxc_sf,weight):
    rho1_ab,rho1_ba = rho1
    # kxc_sf with a shape (5,5,2,5,ngrid), 5 means s \nabla_x,y,z s, u
    # s_s    ->  0: n, \nabla_x,y,z n, tau ;  1: s, \nabla_x,y,z s, u
    # s_Ns   ->
    # Ns_s   ->
    # Ns_Ns  ->
    # s_u    ->
    # u_s    ->
    # u_Ns   ->
    # Ns_u   ->
    # u_u    ->
    ngrid = weight.shape[-1]
    gv_ab, gv_ba = cp.empty((2,2,5,ngrid))
    # Note *2 and *0.5 like in function uks_sf_mgga_wv1
    gv_ab = cp.einsum('xp,yp,xyvzp->vzp', rho1_ab, rho1_ab+rho1_ba, kxc_sf, optimize=True)
    gv_ba = cp.einsum('xp,yp,xyvzp->vzp', rho1_ba, rho1_ba+rho1_ab, kxc_sf, optimize=True)

    gv_ab[:,1:4] *=2.0
    gv_ba[:,1:4] *=2.0
    gv_ab[:,4] *= 0.5
    gv_ba[:,4] *= 0.5
    return gv_ab*weight, gv_ba*weight

def uks_sf_mgga_wv2_m(rho1, kxc_sf,weight):
    rho1_ab,rho1_ba = rho1
    ngrid = weight.shape[-1]
    gv_ab, gv_ba = cp.empty((2,2,5,ngrid))
    # Note *2 and *0.5 like in function uks_sf_mgga_wv1
    gv_ab = cp.einsum('xp,yp,xyvzp->vzp', rho1_ab, rho1_ab-rho1_ba, kxc_sf , optimize=True)
    gv_ba = cp.einsum('xp,yp,xyvzp->vzp', rho1_ba, rho1_ba-rho1_ab, kxc_sf , optimize=True)

    gv_ab[:,1:4] *=2.0
    gv_ba[:,1:4] *=2.0
    gv_ab[:,4] *= 0.5
    gv_ba[:,4] *= 0.5
    return gv_ab*weight, gv_ba*weight


def _contract_xc_kernel_z(td_grad, xc_code, dmvo, max_memory=2000):
    mol = td_grad.base._scf.mol
    mf = td_grad.base._scf
    grids = mf.grids

    ni = mf._numint
    xctype = ni._xc_type(xc_code)

    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    nao = mo_coeff[0].shape[0]
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    opt = getattr(ni, "gdftopt", None)
    if opt is None:
        ni.build(mol, grids.coords)
        opt = ni.gdftopt
    _sorted_mol = opt._sorted_mol
    mo_coeff = opt.sort_orbitals(mo_coeff, axis=[1])

    dmvo = [(dmvo[0]+dmvo[0].T)*.5,
            (dmvo[1]+dmvo[1].T)*.5]

    dmvo0 = opt.sort_orbitals(dmvo[0], axis=[0, 1])
    dmvo1 = opt.sort_orbitals(dmvo[1], axis=[0, 1])

    f1vo = cp.zeros((2,4,nao,nao))
    deriv = 2

    if xctype == "LDA":
        fmat_, ao_deriv = tdrks._lda_eval_mat_, 1
    elif xctype == "GGA":
        fmat_, ao_deriv = _gga_eval_mat_, 2
    elif xctype == "MGGA":
        fmat_, ao_deriv = tdrks._mgga_eval_mat_, 2

    if xctype == 'LDA':

        for ao, mask, weight, coords \
                in ni.block_loop(_sorted_mol, grids, nao, ao_deriv, max_memory):
            mo_coeff_mask_a = mo_coeff[0, mask]
            mo_coeff_mask_b = mo_coeff[1, mask]
            dmvo0_mask = dmvo0[mask[:, None], mask]
            dmvo1_mask = dmvo1[mask[:, None], mask]
            rho = (ni.eval_rho2(_sorted_mol, ao[0], mo_coeff_mask_a, mo_occ[0], mask, xctype),
                   ni.eval_rho2(_sorted_mol, ao[0], mo_coeff_mask_b, mo_occ[1], mask, xctype))
            vxc, fxc = ni.eval_xc_eff(xc_code, rho, deriv=deriv, spin=1)[1:3]
            u_u, u_d, d_d = fxc[0].T * weight
            rho1a = ni.eval_rho(_sorted_mol, ao[0], dmvo0_mask, mask, xctype, hermi=1)
            rho1b = ni.eval_rho(_sorted_mol, ao[0], dmvo1_mask, mask, xctype, hermi=1)

            fmat_(_sorted_mol, f1vo[0], ao, u_u*rho1a+u_d*rho1b, mask, shls_slice, ao_loc)
            fmat_(_sorted_mol, f1vo[1], ao, u_d*rho1a+d_d*rho1b, mask, shls_slice, ao_loc)

    elif xctype == 'GGA':
        for ao, mask, weight, coords \
                in ni.block_loop(_sorted_mol, grids, nao, ao_deriv, max_memory):
            mo_coeff_mask_a = mo_coeff[0, mask]
            mo_coeff_mask_b = mo_coeff[1, mask]
            dmvo0_mask = dmvo0[mask[:, None], mask]
            dmvo1_mask = dmvo1[mask[:, None], mask]
            with_lapl = False
            rho = cp.stack((ni.eval_rho2(_sorted_mol, ao, mo_coeff_mask_a, mo_occ[0], mask, xctype, with_lapl),
                   ni.eval_rho2(_sorted_mol, ao, mo_coeff_mask_b, mo_occ[1], mask, xctype, with_lapl)))
            vxc, fxc = ni.eval_xc_eff(xc_code, rho, deriv=deriv, spin=1)[1:3]

            rho1 = cp.stack((
                    ni.eval_rho(_sorted_mol, ao, dmvo0_mask, mask, xctype, hermi=1, with_lapl=with_lapl),
                    ni.eval_rho(_sorted_mol, ao, dmvo1_mask, mask, xctype, hermi=1, with_lapl=with_lapl)))
            tmp = contract("axg,axbyg->byg", rho1, fxc)
            wv = contract("byg,g->byg", tmp, weight)
            wv[:, 0] *= 0.5
            fmat_(_sorted_mol, f1vo[0], ao, wv[0], mask, shls_slice, ao_loc)
            fmat_(_sorted_mol, f1vo[1], ao, wv[1], mask, shls_slice, ao_loc)

    elif xctype == 'MGGA':
        raise NotImplementedError(f'td-uks for functional {xc_code}')

    elif xctype == 'HF':
        pass
    else:
        raise NotImplementedError(f'td-uks for functional {xc_code}')

    f1vo[:,1:] *= -1
    f1vo = opt.unsort_orbitals(f1vo, axis=[2, 3])
    return f1vo

class Gradients(tdrhf_grad.Gradients):
    @lib.with_doc(grad_elec.__doc__)
    def grad_elec(self, xy, singlet=None, atmlst=None, verbose=None):
        return grad_elec(self, xy, atmlst, self.max_memory, self.verbose)

Grad = Gradients

from pyscf import sftda
sftda.uks_sf.TDA_SF.Gradients = sftda.uks_sf.TDDFT_SF.Gradients = lib.class_as_method(Gradients)