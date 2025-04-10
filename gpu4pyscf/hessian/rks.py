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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#  modified by Xiaojie Wu <wxj6000@gmail.com>

'''
Non-relativistic RKS analytical Hessian
'''

from concurrent.futures import ThreadPoolExecutor
import numpy
import cupy
from pyscf import lib
from gpu4pyscf.hessian import rhf as rhf_hess
from gpu4pyscf.grad import rhf as rhf_grad
from gpu4pyscf.grad import rks as rks_grad
from gpu4pyscf.dft import numint
from gpu4pyscf.lib.cupy_helper import (contract, add_sparse, get_avail_mem,
                                       reduce_to_device, transpose_sum)
from gpu4pyscf.lib import logger
from gpu4pyscf.__config__ import _streams, num_devices
from gpu4pyscf.hessian import jk

def partial_hess_elec(hessobj, mo_energy=None, mo_coeff=None, mo_occ=None,
                      atmlst=None, max_memory=4000, verbose=None):
    log = logger.new_logger(hessobj, verbose)
    time0 = t1 = (logger.process_clock(), logger.perf_counter())

    mol = hessobj.mol
    mf = hessobj.base
    ni = mf._numint
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None:    mo_occ = mf.mo_occ
    if mo_coeff is None:  mo_coeff = mf.mo_coeff

    mocc = mo_coeff[:,mo_occ>0]
    dm0 = cupy.dot(mocc, mocc.T) * 2

    if mf.do_nlc():
        raise NotImplementedError("2nd derivative of NLC is not implemented.")

    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
    with_k = ni.libxc.is_hybrid_xc(mf.xc)
    j_factor = 1.
    k_factor = 0.
    if with_k:
        if omega == 0:
            k_factor = hyb
        elif alpha == 0: # LR=0, only SR exchange
            pass
        elif hyb == 0: # SR=0, only LR exchange
            k_factor = alpha
        else: # SR and LR exchange with different ratios
            k_factor = alpha
    de2, ejk = rhf_hess._partial_hess_ejk(hessobj, mo_energy, mo_coeff, mo_occ,
                                          atmlst, max_memory, verbose,
                                          j_factor, k_factor)
    de2 += ejk  # (A,B,dR_A,dR_B)
    if with_k and omega != 0:
        j_factor = 0.
        omega = -omega # Prefer computing the SR part
        if alpha == 0: # LR=0, only SR exchange
            k_factor = hyb
        elif hyb == 0: # SR=0, only LR exchange
            # full range exchange was computed in the previous step
            k_factor = -alpha
        else: # SR and LR exchange with different ratios
            k_factor = hyb - alpha # =beta
        vhfopt = mf._opt_gpu.get(omega, None)
        with mol.with_range_coulomb(omega):
            de2 += rhf_hess._partial_ejk_ip2(
                mol, dm0, vhfopt, j_factor, k_factor, verbose=verbose)

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, mf.max_memory*.9-mem_now)
    veff_diag = _get_vxc_diag(hessobj, mo_coeff, mo_occ, max_memory)
    t1 = log.timer_debug1('hessian of 2e part', *t1)

    aoslices = mol.aoslice_by_atom()
    vxc_dm = _get_vxc_deriv2(hessobj, mo_coeff, mo_occ, max_memory)
    if atmlst is None:
        atmlst = range(mol.natm)
    for i0, ia in enumerate(atmlst):
        p0, p1 = aoslices[ia,2:]
        veff = vxc_dm[ia]
        de2[i0,i0] += contract('xypq,pq->xy', veff_diag[:,:,p0:p1], dm0[p0:p1])*2
        for j0, ja in enumerate(atmlst[:i0+1]):
            q0, q1 = aoslices[ja][2:]
            de2[i0,j0] += 2.0 * veff[:,:,q0:q1].sum(axis=2)

        for j0 in range(i0):
            de2[j0,i0] = de2[i0,j0].T

    log.timer('RKS partial hessian', *time0)
    return de2

def make_h1(hessobj, mo_coeff, mo_occ, chkfile=None, atmlst=None, verbose=None):
    mol = hessobj.mol
    natm = mol.natm
    assert atmlst is None or atmlst == range(natm)
    mocc = mo_coeff[:,mo_occ>0]
    dm0 = numpy.dot(mocc, mocc.T) * 2
    avail_mem = get_avail_mem()
    max_memory = avail_mem * .8e-6
    h1mo = _get_vxc_deriv1(hessobj, mo_coeff, mo_occ, max_memory)
    h1mo += rhf_grad.get_grad_hcore(hessobj.base.Gradients())

    mf = hessobj.base
    ni = mf._numint
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
    with_k = ni.libxc.is_hybrid_xc(mf.xc)

    # Estimate the size of intermediate variables
    # dm, vj, and vk in [natm,3,nao_cart,nao_cart]
    nao_cart = mol.nao_cart()
    avail_mem -= 8 * h1mo.size
    slice_size = int(avail_mem*0.5) // (8*3*nao_cart*nao_cart*3)
    for atoms_slice in lib.prange(0, natm, slice_size):
        vj, vk = rhf_hess._get_jk_ip1(mol, dm0, with_k=with_k,
                                      atoms_slice=atoms_slice, verbose=verbose)
        veff = vj
        if with_k:
            vk *= .5 * hyb
            veff -= vk
        vj = vk = None
        if abs(omega) > 1e-10 and abs(alpha-hyb) > 1e-10:
            with mol.with_range_coulomb(omega):
                vk_lr = rhf_hess._get_jk_ip1(
                    mol, dm0, with_j=False, atoms_slice=atoms_slice, verbose=verbose)[1]
                vk_lr *= (alpha-hyb) * .5
                veff -= vk_lr
        atom0, atom1 = atoms_slice
        for i, ia in enumerate(range(atom0, atom1)):
            for ix in range(3):
                h1mo[ia,ix] += mo_coeff.T.dot(veff[i,ix].dot(mocc))
        vk_lr = veff = None
    return h1mo

XX, XY, XZ = 4, 5, 6
YX, YY, YZ = 5, 7, 8
ZX, ZY, ZZ = 6, 8, 9
XXX, XXY, XXZ, XYY, XYZ, XZZ = 10, 11, 12, 13, 14, 15
YYY, YYZ, YZZ, ZZZ = 16, 17, 18, 19

def _get_vxc_diag(hessobj, mo_coeff, mo_occ, max_memory):
    mol = hessobj.mol
    mf = hessobj.base
    if hessobj.grids is not None:
        grids = hessobj.grids
    else:
        grids = mf.grids
    if grids.coords is None:
        grids.build(with_non0tab=False)

    # move data to GPU
    mo_occ = cupy.asarray(mo_occ)
    mo_coeff = cupy.asarray(mo_coeff)

    ni = mf._numint
    xctype = ni._xc_type(mf.xc)
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    opt = getattr(ni, 'gdftopt', None)
    if opt is None:
        ni.build(mol, grids.coords)
        opt = ni.gdftopt
    _sorted_mol = opt._sorted_mol
    mo_coeff = opt.sort_orbitals(mo_coeff, axis=[0])
    nao = mo_coeff.shape[0]

    vmat = cupy.zeros((6,nao,nao))
    if xctype == 'LDA':
        ao_deriv = 2
        for ao, mask, weight, coords \
                in ni.block_loop(_sorted_mol, grids, nao, ao_deriv, max_memory):
            mo_coeff_mask = mo_coeff[mask,:]
            rho = numint.eval_rho2(_sorted_mol, ao[0], mo_coeff_mask, mo_occ, mask, xctype)
            vxc = ni.eval_xc_eff(mf.xc, rho, 1, xctype=xctype)[1]
            wv = weight * vxc[0]
            aow = numint._scale_ao(ao[0], wv)
            for i in range(6):
                vmat_tmp = numint._dot_ao_ao(mol, ao[i+4], aow, mask, shls_slice, ao_loc)
                add_sparse(vmat[i], vmat_tmp, mask)
            aow = None

    elif xctype == 'GGA':
        def contract_(ao, aoidx, wv, mask):
            aow = numint._scale_ao(ao[aoidx[0]], wv[1])
            aow+= numint._scale_ao(ao[aoidx[1]], wv[2])
            aow+= numint._scale_ao(ao[aoidx[2]], wv[3])
            return numint._dot_ao_ao(mol, aow, ao[0], mask, shls_slice, ao_loc)

        ao_deriv = 3
        for ao, mask, weight, coords \
                in ni.block_loop(_sorted_mol, grids, nao, ao_deriv, max_memory):
            mo_coeff_mask = mo_coeff[mask,:]
            rho = numint.eval_rho2(_sorted_mol, ao[:4], mo_coeff_mask, mo_occ, mask, xctype)
            vxc = ni.eval_xc_eff(mf.xc, rho, 1, xctype=xctype)[1]
            wv = weight * vxc
            #:aow = numpy.einsum('npi,np->pi', ao[:4], wv[:4])
            aow = numint._scale_ao(ao[:4], wv[:4])

            vmat_tmp = [0]*6
            for i in range(6):
                vmat_tmp[i] = numint._dot_ao_ao(mol, ao[i+4], aow, mask, shls_slice, ao_loc)

            vmat_tmp[0] += contract_(ao, [XXX,XXY,XXZ], wv, mask)
            vmat_tmp[1] += contract_(ao, [XXY,XYY,XYZ], wv, mask)
            vmat_tmp[2] += contract_(ao, [XXZ,XYZ,XZZ], wv, mask)
            vmat_tmp[3] += contract_(ao, [XYY,YYY,YYZ], wv, mask)
            vmat_tmp[4] += contract_(ao, [XYZ,YYZ,YZZ], wv, mask)
            vmat_tmp[5] += contract_(ao, [XZZ,YZZ,ZZZ], wv, mask)
            for i in range(6):
                add_sparse(vmat[i], vmat_tmp[i], mask)
            rho = vxc = wv = aow = None
    elif xctype == 'MGGA':
        def contract_(ao, aoidx, wv, mask):
            aow = numint._scale_ao(ao[aoidx[0]], wv[1])
            aow+= numint._scale_ao(ao[aoidx[1]], wv[2])
            aow+= numint._scale_ao(ao[aoidx[2]], wv[3])
            return numint._dot_ao_ao(mol, aow, ao[0], mask, shls_slice, ao_loc)

        ao_deriv = 3
        for ao, mask, weight, coords \
                in ni.block_loop(_sorted_mol, grids, nao, ao_deriv, max_memory):
            mo_coeff_mask = mo_coeff[mask,:]
            rho = numint.eval_rho2(_sorted_mol, ao[:10], mo_coeff_mask, mo_occ, mask, xctype)
            vxc = ni.eval_xc_eff(mf.xc, rho, 1, xctype=xctype)[1]
            wv = weight * vxc
            wv[4] *= .5  # for the factor 1/2 in tau
            #:aow = numpy.einsum('npi,np->pi', ao[:4], wv[:4])
            vmat_tmp = [0]*6
            aow = numint._scale_ao(ao[:4], wv[:4])
            for i in range(6):
                vmat_tmp[i] = numint._dot_ao_ao(mol, ao[i+4], aow, mask, shls_slice, ao_loc)

            vmat_tmp[0] += contract_(ao, [XXX,XXY,XXZ], wv, mask)
            vmat_tmp[1] += contract_(ao, [XXY,XYY,XYZ], wv, mask)
            vmat_tmp[2] += contract_(ao, [XXZ,XYZ,XZZ], wv, mask)
            vmat_tmp[3] += contract_(ao, [XYY,YYY,YYZ], wv, mask)
            vmat_tmp[4] += contract_(ao, [XYZ,YYZ,YZZ], wv, mask)
            vmat_tmp[5] += contract_(ao, [XZZ,YZZ,ZZZ], wv, mask)

            aow = [numint._scale_ao(ao[i], wv[4]) for i in range(1, 4)]
            for i, j in enumerate([XXX, XXY, XXZ, XYY, XYZ, XZZ]):
                vmat_tmp[i] += numint._dot_ao_ao(mol, ao[j], aow[0], mask, shls_slice, ao_loc)

            for i, j in enumerate([XXY, XYY, XYZ, YYY, YYZ, YZZ]):
                vmat_tmp[i] += numint._dot_ao_ao(mol, ao[j], aow[1], mask, shls_slice, ao_loc)

            for i, j in enumerate([XXZ, XYZ, XZZ, YYZ, YZZ, ZZZ]):
                vmat_tmp[i] += numint._dot_ao_ao(mol, ao[j], aow[2], mask, shls_slice, ao_loc)

            for i in range(6):
                add_sparse(vmat[i], vmat_tmp[i], mask)

    vmat = vmat[[0,1,2,
                 1,3,4,
                 2,4,5]]

    vmat = opt.unsort_orbitals(vmat, axis=[1,2])
    return vmat.reshape(3,3,nao,nao)

def _make_dR_rho1(ao, ao_dm0, atm_id, aoslices, xctype):
    p0, p1 = aoslices[atm_id][2:]
    ngrids = ao[0].shape[1]
    if xctype == 'GGA':
        rho1 = cupy.zeros((3,4,ngrids))
    elif xctype == 'MGGA':
        rho1 = cupy.zeros((3,5,ngrids))
        ao_dm0_x = ao_dm0[1][p0:p1]
        ao_dm0_y = ao_dm0[2][p0:p1]
        ao_dm0_z = ao_dm0[3][p0:p1]
        # (d_X \nabla mu) dot \nalba nu DM_{mu,nu}
        rho1[0,4] += numint._contract_rho(ao[XX,p0:p1], ao_dm0_x)
        rho1[0,4] += numint._contract_rho(ao[XY,p0:p1], ao_dm0_y)
        rho1[0,4] += numint._contract_rho(ao[XZ,p0:p1], ao_dm0_z)
        rho1[1,4] += numint._contract_rho(ao[YX,p0:p1], ao_dm0_x)
        rho1[1,4] += numint._contract_rho(ao[YY,p0:p1], ao_dm0_y)
        rho1[1,4] += numint._contract_rho(ao[YZ,p0:p1], ao_dm0_z)
        rho1[2,4] += numint._contract_rho(ao[ZX,p0:p1], ao_dm0_x)
        rho1[2,4] += numint._contract_rho(ao[ZY,p0:p1], ao_dm0_y)
        rho1[2,4] += numint._contract_rho(ao[ZZ,p0:p1], ao_dm0_z)
        rho1[:,4] *= .5
    else:
        raise RuntimeError

    ao_dm0_0 = ao_dm0[0][p0:p1]
    # (d_X \nabla_x mu) nu DM_{mu,nu}
    rho1[:,0] = numint._contract_rho1(ao[1:4,p0:p1], ao_dm0_0)
    rho1[0,1]+= numint._contract_rho(ao[XX,p0:p1], ao_dm0_0)
    rho1[0,2]+= numint._contract_rho(ao[XY,p0:p1], ao_dm0_0)
    rho1[0,3]+= numint._contract_rho(ao[XZ,p0:p1], ao_dm0_0)
    rho1[1,1]+= numint._contract_rho(ao[YX,p0:p1], ao_dm0_0)
    rho1[1,2]+= numint._contract_rho(ao[YY,p0:p1], ao_dm0_0)
    rho1[1,3]+= numint._contract_rho(ao[YZ,p0:p1], ao_dm0_0)
    rho1[2,1]+= numint._contract_rho(ao[ZX,p0:p1], ao_dm0_0)
    rho1[2,2]+= numint._contract_rho(ao[ZY,p0:p1], ao_dm0_0)
    rho1[2,3]+= numint._contract_rho(ao[ZZ,p0:p1], ao_dm0_0)
    # (d_X mu) (\nabla_x nu) DM_{mu,nu}
    rho1[:,1] += numint._contract_rho1(ao[1:4,p0:p1], ao_dm0[1][p0:p1])
    rho1[:,2] += numint._contract_rho1(ao[1:4,p0:p1], ao_dm0[2][p0:p1])
    rho1[:,3] += numint._contract_rho1(ao[1:4,p0:p1], ao_dm0[3][p0:p1])

    # *2 for |mu> DM <d_X nu|
    return rho1 * 2

def _d1d2_dot_(vmat, mol, ao1, ao2, mask, ao_loc, dR1_on_bra=True):
    shls_slice = None
    if dR1_on_bra:  # (d/dR1 bra) * (d/dR2 ket)
        for d1 in range(3):
            for d2 in range(3):
                vmat[d1,d2] += numint._dot_ao_ao(mol, ao1[d1], ao2[d2], mask,
                                                 shls_slice, ao_loc)
        #vmat += contract('xig,yjg->xyij', ao1, ao2)
    else:  # (d/dR2 bra) * (d/dR1 ket)
        for d1 in range(3):
            for d2 in range(3):
                vmat[d1,d2] += numint._dot_ao_ao(mol, ao1[d2], ao2[d1], mask,
                                                 shls_slice, ao_loc)
        #vmat += contract('yig,xjg->xyij', ao1, ao2)

def _get_vxc_deriv2_task(hessobj, grids, mo_coeff, mo_occ, max_memory, device_id=0, verbose=0):
    mol = hessobj.mol
    mf = hessobj.base
    ni = mf._numint
    nao = mol.nao
    opt = ni.gdftopt

    _sorted_mol = opt._sorted_mol
    xctype = ni._xc_type(mf.xc)
    aoslices = mol.aoslice_by_atom()
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    ngrids_glob = grids.coords.shape[0]
    grid_start, grid_end = numint.gen_grid_range(ngrids_glob, device_id)

    with cupy.cuda.Device(device_id), _streams[device_id]:
        log = logger.new_logger(mol, verbose)
        t1 = t0 = log.init_timer()
        mo_occ = cupy.asarray(mo_occ)
        mo_coeff = cupy.asarray(mo_coeff)
        dm0 = mf.make_rdm1(mo_coeff, mo_occ)
        dm0_sorted = opt.sort_orbitals(dm0, axis=[0,1])
        coeff = cupy.asarray(opt.coeff)
        vmat_dm = cupy.zeros((_sorted_mol.natm,3,3,nao))
        ipip = cupy.zeros((3,3,nao,nao))
        if xctype == 'LDA':
            ao_deriv = 1
            t1 = log.init_timer()
            for ao_mask, mask, weight, _ in ni.block_loop(_sorted_mol, grids, nao, ao_deriv, None,
                                                          grid_range=(grid_start, grid_end)):
                nao_non0 = len(mask)
                ao = contract('nip,ij->njp', ao_mask, coeff[mask])
                rho = numint.eval_rho2(_sorted_mol, ao[0], mo_coeff, mo_occ, mask, xctype)
                t1 = log.timer_debug2('eval rho', *t1)
                vxc, fxc = ni.eval_xc_eff(mf.xc, rho, 2, xctype=xctype)[1:3]
                t1 = log.timer_debug2('eval vxc', *t1)
                wv = weight * vxc[0]
                aow = [numint._scale_ao(ao[i], wv) for i in range(1, 4)]
                _d1d2_dot_(ipip, mol, aow, ao[1:4], mask, ao_loc, False)
                dm0_mask = dm0_sorted[mask[:,None], mask]

                ao_dm_mask = contract('nig,ij->njg', ao_mask[:4], dm0_mask)
                ao_dm0 = numint._dot_ao_dm(mol, ao[0], dm0, mask, shls_slice, ao_loc)
                wf = weight * fxc[0,0]
                for ia in range(_sorted_mol.natm):
                    p0, p1 = aoslices[ia][2:]
                    # *2 for \nabla|ket> in rho1
                    rho1 = contract('xig,ig->xg', ao[1:,p0:p1,:], ao_dm0[p0:p1,:]) * 2
                    # aow ~ rho1 ~ d/dR1
                    wv = wf * rho1
                    aow = cupy.empty_like(ao_dm_mask[1:4])
                    for i in range(3):
                        aow[i] = numint._scale_ao(ao_dm_mask[0], wv[i])
                    vmat_dm[ia][:,:,mask] += contract('yjg,xjg->xyj', ao_mask[1:4], aow)
                ao_dm0 = aow = None
                t1 = log.timer_debug2('integration', *t1)

            vmat_dm = opt.unsort_orbitals(vmat_dm, axis=[3])
            for ia in range(_sorted_mol.natm):
                p0, p1 = aoslices[ia][2:]
                vmat_dm[ia] += contract('xypq,pq->xyp', ipip[:,:,:,p0:p1], dm0[:,p0:p1])
        elif xctype == 'GGA':
            ao_deriv = 2
            t1 = log.init_timer()
            for ao_mask, mask, weight, _ in ni.block_loop(_sorted_mol, grids, nao, ao_deriv, None,
                                                          grid_range=(grid_start, grid_end)):
                nao_non0 = len(mask)
                ao = contract('nip,ij->njp', ao_mask, coeff[mask])
                rho = numint.eval_rho2(_sorted_mol, ao[:4], mo_coeff, mo_occ, mask, xctype)
                t1 = log.timer_debug2('eval rho', *t1)
                vxc, fxc = ni.eval_xc_eff(mf.xc, rho, 2, xctype=xctype)[1:3]
                t1 = log.timer_debug2('eval vxc', *t1)
                wv = weight * vxc
                wv[0] *= .5
                aow = rks_grad._make_dR_dao_w(ao, wv)
                _d1d2_dot_(ipip, mol, aow, ao[1:4], mask, ao_loc, False)
                ao_dm0 = [numint._dot_ao_dm(mol, ao[i], dm0, mask, shls_slice, ao_loc) for i in range(4)]
                wf = weight * fxc
                dm0_mask = dm0_sorted[mask[:,None], mask]
                ao_dm_mask = contract('nig,ij->njg', ao_mask[:4], dm0_mask)
                vmat_dm_tmp = cupy.empty([3,3,nao_non0])
                for ia in range(_sorted_mol.natm):
                    dR_rho1 = _make_dR_rho1(ao, ao_dm0, ia, aoslices, xctype)
                    wv = contract('xyg,sxg->syg', wf, dR_rho1)
                    wv[:,0] *= .5
                    for i in range(3):
                        aow = rks_grad._make_dR_dao_w(ao_mask, wv[i])
                        vmat_dm_tmp[i] = contract('xjg,jg->xj', aow, ao_dm_mask[0])
                    for i in range(3):
                        aow[i] = numint._scale_ao(ao_dm_mask[:4], wv[i,:4])
                    vmat_dm_tmp += contract('yjg,xjg->xyj', ao_mask[1:4], aow)
                    vmat_dm[ia][:,:,mask] += vmat_dm_tmp
                ao_dm0 = aow = None
                t1 = log.timer_debug2('integration', *t1)
            vmat_dm = opt.unsort_orbitals(vmat_dm, axis=[3])
            for ia in range(_sorted_mol.natm):
                p0, p1 = aoslices[ia][2:]
                vmat_dm[ia] += contract('xypq,pq->xyp', ipip[:,:,:,p0:p1], dm0[:,p0:p1])
                vmat_dm[ia] += contract('yxqp,pq->xyp', ipip[:,:,p0:p1], dm0[:,p0:p1])

        elif xctype == 'MGGA':
            XX, XY, XZ = 4, 5, 6
            YX, YY, YZ = 5, 7, 8
            ZX, ZY, ZZ = 6, 8, 9
            ao_deriv = 2
            t1 = log.init_timer()
            for ao_mask, mask, weight, _ in ni.block_loop(_sorted_mol, grids, nao, ao_deriv, None,
                                                          grid_range=(grid_start, grid_end)):
                nao_non0 = len(mask)
                ao = contract('nip,ij->njp', ao_mask, coeff[mask])
                rho = numint.eval_rho2(_sorted_mol, ao[:10], mo_coeff, mo_occ, mask, xctype)
                t1 = log.timer_debug2('eval rho', *t1)
                vxc, fxc = ni.eval_xc_eff(mf.xc, rho, 2, xctype=xctype)[1:3]
                t1 = log.timer_debug2('eval vxc', *t1)
                wv = weight * vxc
                wv[0] *= .5
                wv[4] *= .25
                aow = rks_grad._make_dR_dao_w(ao, wv)
                _d1d2_dot_(ipip, mol, aow, ao[1:4], mask, ao_loc, False)

                aow = [numint._scale_ao(ao[i], wv[4]) for i in range(4, 10)]
                _d1d2_dot_(ipip, mol, [aow[0], aow[1], aow[2]], [ao[XX], ao[XY], ao[XZ]], mask, ao_loc, False)
                _d1d2_dot_(ipip, mol, [aow[1], aow[3], aow[4]], [ao[YX], ao[YY], ao[YZ]], mask, ao_loc, False)
                _d1d2_dot_(ipip, mol, [aow[2], aow[4], aow[5]], [ao[ZX], ao[ZY], ao[ZZ]], mask, ao_loc, False)
                dm0_mask = dm0_sorted[mask[:,None], mask]
                ao_dm0 = [numint._dot_ao_dm(mol, ao[i], dm0, mask, shls_slice, ao_loc) for i in range(4)]
                ao_dm_mask = contract('nig,ij->njg', ao_mask[:4], dm0_mask)
                wf = weight * fxc
                for ia in range(_sorted_mol.natm):
                    dR_rho1 = _make_dR_rho1(ao, ao_dm0, ia, aoslices, xctype)
                    wv = contract('xyg,sxg->syg', wf, dR_rho1)
                    wv[:,0] *= .5
                    wv[:,4] *= .5  # for the factor 1/2 in tau
                    vmat_dm_tmp = cupy.empty([3,3,nao_non0])
                    for i in range(3):
                        aow = rks_grad._make_dR_dao_w(ao_mask, wv[i])
                        vmat_dm_tmp[i] = contract('xjg,jg->xj', aow, ao_dm_mask[0])

                    for i in range(3):
                        aow[i] = numint._scale_ao(ao_dm_mask[:4], wv[i,:4])
                    vmat_dm_tmp += contract('yjg,xjg->xyj', ao_mask[1:4], aow)

                    for i in range(3):
                        aow[i] = numint._scale_ao(ao_dm_mask[1], wv[i,4])
                    vmat_dm_tmp[:,0] += contract('jg,xjg->xj', ao_mask[XX], aow)
                    vmat_dm_tmp[:,1] += contract('jg,xjg->xj', ao_mask[XY], aow)
                    vmat_dm_tmp[:,2] += contract('jg,xjg->xj', ao_mask[XZ], aow)

                    for i in range(3):
                        aow[i] = numint._scale_ao(ao_dm_mask[2], wv[i,4])
                    vmat_dm_tmp[:,0] += contract('jg,xjg->xj', ao_mask[YX], aow)
                    vmat_dm_tmp[:,1] += contract('jg,xjg->xj', ao_mask[YY], aow)
                    vmat_dm_tmp[:,2] += contract('jg,xjg->xj', ao_mask[YZ], aow)

                    for i in range(3):
                        aow[i] = numint._scale_ao(ao_dm_mask[3], wv[i,4])
                    vmat_dm_tmp[:,0] += contract('jg,xjg->xj', ao_mask[ZX], aow)
                    vmat_dm_tmp[:,1] += contract('jg,xjg->xj', ao_mask[ZY], aow)
                    vmat_dm_tmp[:,2] += contract('jg,xjg->xj', ao_mask[ZZ], aow)

                    vmat_dm[ia][:,:,mask] += vmat_dm_tmp
                t1 = log.timer_debug2('integration', *t1)
            vmat_dm = opt.unsort_orbitals(vmat_dm, axis=[3])
            for ia in range(_sorted_mol.natm):
                p0, p1 = aoslices[ia][2:]
                vmat_dm[ia] += contract('xypq,pq->xyp', ipip[:,:,:,p0:p1], dm0[:,p0:p1])
                vmat_dm[ia] += contract('yxqp,pq->xyp', ipip[:,:,p0:p1], dm0[:,p0:p1])
        t0 = log.timer_debug1(f'vxc_deriv2 on Device {device_id}', *t0)
    return vmat_dm

def _get_vxc_deriv2(hessobj, mo_coeff, mo_occ, max_memory):
    '''Partially contracted vxc*dm'''
    mol = hessobj.mol
    mf = hessobj.base
    if hessobj.grids is not None:
        grids = hessobj.grids
    else:
        grids = mf.grids

    if grids.coords is None:
        grids.build(with_non0tab=True)

    ni = mf._numint
    opt = getattr(ni, 'gdftopt', None)
    if opt is None:
        ni.build(mol, grids.coords)
        opt = ni.gdftopt

    futures = []
    cupy.cuda.get_current_stream().synchronize()
    with ThreadPoolExecutor(max_workers=num_devices) as executor:
        for device_id in range(num_devices):
            future = executor.submit(
                _get_vxc_deriv2_task,
                hessobj, grids, mo_coeff, mo_occ, max_memory,
                device_id=device_id, verbose=mol.verbose)
            futures.append(future)
    vmat_dm_dist = [future.result() for future in futures]
    vmat_dm = reduce_to_device(vmat_dm_dist, inplace=True)
    return vmat_dm

def _get_vxc_deriv1_task(hessobj, grids, mo_coeff, mo_occ, max_memory, device_id=0):
    mol = hessobj.mol
    mf = hessobj.base
    ni = mf._numint
    nao, nmo = mo_coeff.shape
    natm = mol.natm
    opt = ni.gdftopt

    _sorted_mol = opt._sorted_mol
    xctype = ni._xc_type(mf.xc)
    aoslices = mol.aoslice_by_atom()
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    ngrids_glob = grids.coords.shape[0]
    grid_start, grid_end = numint.gen_grid_range(ngrids_glob, device_id)
    with cupy.cuda.Device(device_id), _streams[device_id]:
        mo_occ = cupy.asarray(mo_occ)
        mo_coeff = cupy.asarray(mo_coeff)
        coeff = cupy.asarray(opt.coeff)
        mocc = mo_coeff[:,mo_occ>0]
        dm0 = mf.make_rdm1(mo_coeff, mo_occ)
        nocc = mocc.shape[1]

        log = logger.new_logger(mol, mol.verbose)
        v_ip = cupy.zeros((3,nao,nao))
        vmat = cupy.zeros((natm,3,nao,nocc))
        max_memory = max(2000, max_memory-vmat.size*8/1e6)
        t1 = t0 = log.init_timer()
        if xctype == 'LDA':
            ao_deriv = 1
            for ao, mask, weight, _ in ni.block_loop(_sorted_mol, grids, nao, ao_deriv, None,
                                                     grid_range=(grid_start, grid_end)):
                ao = contract('nip,ij->njp', ao, coeff[mask])
                rho = numint.eval_rho2(_sorted_mol, ao[0], mo_coeff, mo_occ, mask, xctype)
                t1 = log.timer_debug2('eval rho', *t1)
                vxc, fxc = ni.eval_xc_eff(mf.xc, rho, 2, xctype=xctype)[1:3]
                t1 = log.timer_debug2('eval vxc', *t1)
                wv = weight * vxc[0]
                aow = numint._scale_ao(ao[0], wv)
                v_ip += rks_grad._d1_dot_(ao[1:4], aow.T)
                mo = contract('xig,ip->xpg', ao, mocc)
                ao_dm0 = numint._dot_ao_dm(mol, ao[0], dm0, mask, shls_slice, ao_loc)
                wf = weight * fxc[0,0]
                for ia in range(natm):
                    p0, p1 = aoslices[ia][2:]
    # First order density = rho1 * 2.  *2 is not applied because + c.c. in the end
                    rho1 = contract('xig,ig->xg', ao[1:,p0:p1,:], ao_dm0[p0:p1,:])
                    wv = wf * rho1
                    aow = [numint._scale_ao(ao[0], wv[i]) for i in range(3)]
                    mow = [numint._scale_ao(mo[0], wv[i]) for i in range(3)]
                    vmat[ia] += rks_grad._d1_dot_(aow, mo[0].T)
                    vmat[ia] += rks_grad._d1_dot_(mow, ao[0].T).transpose([0,2,1])
                ao_dm0 = aow = None
                t1 = log.timer_debug2('integration', *t1)
        elif xctype == 'GGA':
            ao_deriv = 2
            for ao, mask, weight, _ in ni.block_loop(_sorted_mol, grids, nao, ao_deriv, None,
                                                     grid_range=(grid_start, grid_end)):
                ao = contract('nip,ij->njp', ao, coeff[mask])
                rho = numint.eval_rho2(_sorted_mol, ao[:4], mo_coeff, mo_occ, mask, xctype)
                t1 = log.timer_debug2('eval rho', *t1)
                vxc, fxc = ni.eval_xc_eff(mf.xc, rho, 2, xctype=xctype)[1:3]
                t1 = log.timer_debug2('eval vxc', *t1)
                wv = weight * vxc
                wv[0] *= .5
                v_ip += rks_grad._gga_grad_sum_(ao, wv)
                mo = contract('xig,ip->xpg', ao, mocc)
                ao_dm0 = [numint._dot_ao_dm(mol, ao[i], dm0, mask, shls_slice, ao_loc)
                        for i in range(4)]
                wf = weight * fxc
                for ia in range(natm):
                    dR_rho1 = _make_dR_rho1(ao, ao_dm0, ia, aoslices, xctype)
                    wv = contract('xyg,sxg->syg', wf, dR_rho1)
                    wv[:,0] *= .5
                    aow = [numint._scale_ao(ao[:4], wv[i,:4]) for i in range(3)]
                    mow = [numint._scale_ao(mo[:4], wv[i,:4]) for i in range(3)]
                    vmat[ia] += rks_grad._d1_dot_(aow, mo[0].T)
                    vmat[ia] += rks_grad._d1_dot_(mow, ao[0].T).transpose([0,2,1])
                t1 = log.timer_debug2('integration', *t1)
                ao_dm0 = aow = None
        elif xctype == 'MGGA':
            if grids.level < 5:
                log.warn('MGGA Hessian is sensitive to dft grids.')
            ao_deriv = 2
            for ao, mask, weight, _ in ni.block_loop(_sorted_mol, grids, nao, ao_deriv, None,
                                                     grid_range=(grid_start, grid_end)):
                ao = contract('nip,ij->njp', ao, coeff[mask])
                rho = numint.eval_rho2(_sorted_mol, ao[:10], mo_coeff, mo_occ, mask, xctype)
                t1 = log.timer_debug2('eval rho', *t1)
                vxc, fxc = ni.eval_xc_eff(mf.xc, rho, 2, xctype=xctype)[1:3]
                t1 = log.timer_debug2('eval vxc', *t0)
                wv = weight * vxc
                wv[0] *= .5
                wv[4] *= .5  # for the factor 1/2 in tau
                v_ip += rks_grad._gga_grad_sum_(ao, wv)
                v_ip += rks_grad._tau_grad_dot_(ao, wv[4])
                mo = contract('xig,ip->xpg', ao, mocc)
                ao_dm0 = [numint._dot_ao_dm(mol, ao[i], dm0, mask, shls_slice, ao_loc) for i in range(4)]
                wf = weight * fxc
                for ia in range(natm):
                    dR_rho1 = _make_dR_rho1(ao, ao_dm0, ia, aoslices, xctype)
                    wv = contract('xyg,sxg->syg', wf, dR_rho1)
                    wv[:,0] *= .5
                    wv[:,4] *= .25
                    aow = [numint._scale_ao(ao[:4], wv[i,:4]) for i in range(3)]
                    mow = [numint._scale_ao(mo[:4], wv[i,:4]) for i in range(3)]
                    vmat[ia] += rks_grad._d1_dot_(aow, mo[0].T)
                    vmat[ia] += rks_grad._d1_dot_(mow, ao[0].T).transpose([0,2,1])

                    for j in range(1, 4):
                        aow = [numint._scale_ao(ao[j], wv[i,4]) for i in range(3)]
                        mow = [numint._scale_ao(mo[j], wv[i,4]) for i in range(3)]
                        vmat[ia] += rks_grad._d1_dot_(aow, mo[j].T)
                        vmat[ia] += rks_grad._d1_dot_(mow, ao[j].T).transpose([0,2,1])
                ao_dm0 = aow = None
                t1 = log.timer_debug2('integration', *t1)
        t0 = log.timer_debug1(f'vxc_deriv1 on Device {device_id}', *t0)

        # Inplace transform the AO to MO.
        v_mo = cupy.ndarray((natm,3,nmo,nocc), dtype=vmat.dtype, memptr=vmat.data)
        vmat_tmp = cupy.empty([3,nao,nao])
        for ia in range(natm):
            p0, p1 = aoslices[ia][2:]
            vmat_tmp[:] = 0.
            vmat_tmp[:,p0:p1] += v_ip[:,p0:p1]
            vmat_tmp[:,:,p0:p1] += v_ip[:,p0:p1].transpose(0,2,1)
            tmp = contract('xij,jq->xiq', vmat_tmp, mocc)
            tmp += vmat[ia]
            contract('xiq,ip->xpq', tmp, mo_coeff, alpha=-1., out=v_mo[ia])
    return v_mo

def _get_vxc_deriv1(hessobj, mo_coeff, mo_occ, max_memory):
    '''
    Derivatives of Vxc matrix in MO bases
    '''
    mol = hessobj.mol
    mf = hessobj.base
    if hessobj.grids is not None:
        grids = hessobj.grids
    else:
        grids = mf.grids

    if grids.coords is None:
        grids.build(with_non0tab=True)

    ni = mf._numint
    opt = getattr(ni, 'gdftopt', None)
    if opt is None:
        ni.build(mol, grids.coords)
        opt = ni.gdftopt

    futures = []
    cupy.cuda.get_current_stream().synchronize()
    with ThreadPoolExecutor(max_workers=num_devices) as executor:
        for device_id in range(num_devices):
            future = executor.submit(
                _get_vxc_deriv1_task,
                hessobj, grids, mo_coeff, mo_occ, max_memory,
                device_id=device_id)
            futures.append(future)
    vmat_dist = [future.result() for future in futures]
    vmat = reduce_to_device(vmat_dist, inplace=True)
    return vmat

def _nr_rks_fxc_mo_task(ni, mol, grids, xc_code, fxc, mo_coeff, mo1, mocc,
                        verbose=None, hermi=1, device_id=0):
    with cupy.cuda.Device(device_id), _streams[device_id]:
        if mo_coeff is not None: mo_coeff = cupy.asarray(mo_coeff)
        if mo1 is not None: mo1 = cupy.asarray(mo1)
        if mocc is not None: mocc = cupy.asarray(mocc)
        if fxc is not None: fxc = cupy.asarray(fxc)

        assert isinstance(verbose, int)
        log = logger.new_logger(mol, verbose)
        xctype = ni._xc_type(xc_code)
        opt = getattr(ni, 'gdftopt', None)

        _sorted_mol = opt.mol
        nao = mol.nao
        nset = mo1.shape[0]
        vmat = cupy.zeros((nset, nao, nao))

        if xctype == 'LDA':
            ao_deriv = 0
        else:
            ao_deriv = 1

        ngrids_glob = grids.coords.shape[0]
        grid_start, grid_end = numint.gen_grid_range(ngrids_glob, device_id)
        ngrids_local = grid_end - grid_start
        log.debug(f"{ngrids_local} grids on Device {device_id}")

        p0 = p1 = grid_start
        t1 = t0 = log.init_timer()
        for ao, mask, weights, coords in ni.block_loop(_sorted_mol, grids, nao, ao_deriv,
                                                       max_memory=None, blksize=None,
                                                       grid_range=(grid_start, grid_end)):
            p0, p1 = p1, p1+len(weights)
            occ_coeff_mask = mocc[mask]
            rho1 = numint.eval_rho4(_sorted_mol, ao, 2.0*occ_coeff_mask, mo1[:,mask],
                                    xctype=xctype, hermi=hermi)
            t1 = log.timer_debug2('eval rho', *t1)

            # precompute fxc_w
            if xctype == 'LDA':
                fxc_w = fxc[0,0,p0:p1] * weights
                wv = rho1 * fxc_w
            else:
                fxc_w = fxc[:,:,p0:p1] * weights
                wv = contract('axg,xyg->ayg', rho1, fxc_w)

            for i in range(nset):
                if xctype == 'LDA':
                    vmat_tmp = ao.dot(numint._scale_ao(ao, wv[i]).T)
                elif xctype == 'GGA':
                    wv[i,0] *= .5
                    aow = numint._scale_ao(ao, wv[i])
                    vmat_tmp = aow.dot(ao[0].T)
                elif xctype == 'NLC':
                    raise NotImplementedError('NLC')
                else:
                    wv[i,0] *= .5
                    wv[i,4] *= .5
                    vmat_tmp = ao[0].dot(numint._scale_ao(ao[:4], wv[i,:4]).T)
                    vmat_tmp+= numint._tau_dot(ao, ao, wv[i,4])
                add_sparse(vmat[i], vmat_tmp, mask)

            t1 = log.timer_debug2('integration', *t1)
            ao = rho1 = None
        t0 = log.timer_debug1(f'vxc on Device {device_id} ', *t0)
        if xctype != 'LDA':
            transpose_sum(vmat)
        vmat = jk._ao2mo(vmat, mocc, mo_coeff)
    return vmat

def nr_rks_fxc_mo(ni, mol, grids, xc_code, dm0=None, dms=None, mo_coeff=None, relativity=0, hermi=0,
               rho0=None, vxc=None, fxc=None, max_memory=2000, verbose=None):
    log = logger.new_logger(mol, verbose)
    t0 = log.init_timer()
    if fxc is None:
        raise RuntimeError('fxc was not initialized')
    #xctype = ni._xc_type(xc_code)
    opt = getattr(ni, 'gdftopt', None)
    if opt is None or mol not in [opt.mol, opt._sorted_mol]:
        ni.build(mol, grids.coords)
        opt = ni.gdftopt

    nao = mol.nao
    dms = cupy.asarray(dms)
    dm_shape = dms.shape
    # AO basis -> gdftopt AO basis
    with_mocc = hasattr(dms, 'mo1')
    mo1 = mocc = None
    if with_mocc:
        mo1 = opt.sort_orbitals(dms.mo1, axis=[1])
        mocc = opt.sort_orbitals(dms.occ_coeff, axis=[0])
    mo_coeff = opt.sort_orbitals(mo_coeff, axis=[0])
    dms = opt.sort_orbitals(dms.reshape(-1,nao,nao), axis=[1,2])
    
    futures = []
    cupy.cuda.get_current_stream().synchronize()
    with ThreadPoolExecutor(max_workers=num_devices) as executor:
        for device_id in range(num_devices):
            future = executor.submit(
                _nr_rks_fxc_mo_task,
                ni, mol, grids, xc_code, fxc, mo_coeff, mo1, mocc,
                verbose=log.verbose, hermi=hermi, device_id=device_id)
            futures.append(future)
    dms = None
    vmat_dist = []
    for future in futures:
        vmat_dist.append(future.result())
    vmat = reduce_to_device(vmat_dist, inplace=True)

    if len(dm_shape) == 2:
        vmat = vmat[0]
    t0 = log.timer_debug1('nr_rks_fxc', *t0)
    return cupy.asarray(vmat)

def get_veff_resp_mo(hessobj, mol, dms, mo_coeff, mo_occ, hermi=1, omega=None):
    mol = hessobj.mol
    mf = hessobj.base
    grids = getattr(mf, 'cphf_grids', None)
    if grids is not None:
        logger.info(mf, 'Secondary grids defined for CPHF in Hessian')
    else:
        # If cphf_grids is not defined, e.g object defined from CPU
        grids = getattr(mf, 'grids', None)
        logger.info(mf, 'Primary grids is used for CPHF in Hessian')

    if grids and grids.coords is None:
        grids.build(mol=mol, with_non0tab=False, sort_grids=True)

    ni = mf._numint
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
    hybrid = ni.libxc.is_hybrid_xc(mf.xc)
    assert not mf.do_nlc()
    hermi = 1

    mocc = mo_coeff[:,mo_occ>0]
    nocc = mocc.shape[1]
    nao, nmo = mo_coeff.shape
    # TODO: evaluate v1 in MO
    rho0, vxc, fxc = ni.cache_xc_kernel(mol, grids, mf.xc,
                                        mo_coeff, mo_occ, 0)
    v1 = nr_rks_fxc_mo(ni, mol, grids, mf.xc, None, dms, mo_coeff, 0, hermi,
                                    rho0, vxc, fxc, max_memory=None)
    v1 = v1.reshape(-1,nmo*nocc)
    
    if hybrid:
        vj, vk = hessobj.get_jk_mo(mol, dms, mo_coeff, mo_occ, hermi=1)
        vk *= hyb
        if omega > 1e-10:  # For range separated Coulomb
            _, vk_lr = hessobj.get_jk_mo(mol, dms, mo_coeff, mo_occ, hermi,
                                        with_j=False, omega=omega)
            vk_lr *= (alpha-hyb)
            vk += vk_lr
        v1 += vj - .5 * vk
    else:
        v1 += hessobj.get_jk_mo(mol, dms, mo_coeff, mo_occ, hermi=1,
                                with_k=False)[0]

    return v1


class Hessian(rhf_hess.HessianBase):
    '''Non-relativistic RKS hessian'''

    from gpu4pyscf.lib.utils import to_gpu, device

    _keys = {'grids', 'grid_response'}

    def __init__(self, mf):
        rhf_hess.Hessian.__init__(self, mf)
        self.grids = None
        self.grid_response = False

    partial_hess_elec = partial_hess_elec
    hess_elec = rhf_hess.hess_elec
    make_h1 = make_h1
    gen_vind = rhf_hess.gen_vind
    get_jk_mo = rhf_hess._get_jk_mo
    get_veff_resp_mo = get_veff_resp_mo

from gpu4pyscf import dft
dft.rks.RKS.Hessian = lib.class_as_method(Hessian)
