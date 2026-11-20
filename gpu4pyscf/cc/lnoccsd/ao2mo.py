"""GPU density-fitting transformations for global ovL and fragment MO integrals.

The global transformation produces (occupied, virtual, auxiliary) factors.
The fragment transformation targets the *real RHF molecular* PySCF-forge modified CCSD and
DFCCSD ERI structures.  One-electron terms and frozen-core treatment are
performed by the original _ChemistsERIs._common_init_ and are not replaced.

Algorithm
---------
1. Stream PySCF with_df.loop() AO DF Cholesky factors (AO s2 packed).
2. On GPU, expand AO s2 and apply C^T L[P,AO,AO] C using batched GEMMs.
3. Store the transformed Loo, Lov and packed Lvv factors in bounded host
   scratch (NumPy or HDF5), keeping only AO blocks on device.
4. Contract the LAS integral sectors using chunked GPU GEMMs, writing
   output to PySCF-compatible HDF5 datasets.  Full creates packed vvvv;
   DF writes only vvL and NEVER assembles full vvvv.

"""
from __future__ import annotations

import time
import numpy as np
import cupy as cp


def transform_ao_df_block(ao_factors, mo_coeff, *, packed=True):
    """Return (P,nmo,nmo) real DF 3-center integrals in an AO->MO basis.

    PySCF's AO `s2` order is lower-triangular row-major order.
    The same function is used by synthetic tests on NumPy and by CUDA.
    """
    c = cp.asarray(mo_coeff, dtype=cp.float64)
    z = cp.asarray(ao_factors, dtype=cp.float64)
    if c.ndim != 2 or z.ndim != 2 and z.ndim != 3:
        raise ValueError('Expected MO matrix (nao,nmo) and AO factors')
    nao, _ = c.shape
    npair = nao*(nao+1)//2
    if packed:
        if z.ndim != 2 or z.shape[1] != npair:
            raise ValueError(f'Expected AO s2 shape (naux_block,{npair}), got {z.shape}')
        ind = cp.tril_indices(nao)
        full = cp.zeros((z.shape[0],nao,nao),dtype=cp.float64)
        full[:,ind[0],ind[1]] = z
        full[:,ind[1],ind[0]] = z
    else:
        if z.shape[1:] != (nao,nao):
            raise ValueError('Expected AO s1 factors (naux_block,nao,nao)')
        full = z
    # CuPy batched DGEMM; neither the AO nor the MO pair dimensions are
    # collapsed to create a four-index intermediate.
    intermediate = cp.matmul(full,c)
    result = cp.matmul(c.T[None,:,:],intermediate)
    return result


def _packed_virtual(transformed, no):
    nv = transformed.shape[1]-no
    ind = cp.tril_indices(nv)
    return transformed[:,no+ind[0],no+ind[1]]


def _allocate_factor_scratch(naux,no,nv,host_scratch_mib):
    from pyscf import lib
    dims={'oo':no*no,'ov':no*nv,'vo':no*nv,'vv':nv*(nv+1)//2}
    nbytes=8*naux*sum(dims.values())
    scratch=None
    if nbytes <= int(host_scratch_mib)*2**20:
        out={key:np.empty((naux,n),dtype=np.float64) for key,n in dims.items()}
    else:
        scratch=lib.H5TmpFile()
        # Factor scratch is one-time; HDF5 protects against unbounded host RAM.
        out={key:scratch.create_dataset(key,(naux,n),'f8') for key,n in dims.items()}
    return out,scratch,nbytes


def _contract_rows(A,B,destination,*,workspace_bytes, aux_block_cap=None):
    """Stream (A.T @ B) in output-row / auxiliary blocks to a host dataset.

    A,B shapes: (naux,nleft), (naux,nright). Both may be HDF5 datasets.
    Output may be large, but GPU buffers are bounded by workspace_bytes.
    """
    naux,nleft=A.shape
    if B.shape[0]!=naux or destination.shape!=(nleft,B.shape[1]):
        raise ValueError('Bad GEMM tensor shapes')
    nright=B.shape[1]
    budget=max(8*1024*1024,int(workspace_bytes))
    # An output row can be large; avoid allocating full output on device.
    row_block=max(1,min(nleft,int((budget//2)//max(8,8*nright))))
    for i0 in range(0,nleft,row_block):
        i1=min(nleft,i0+row_block)
        out=cp.zeros((i1-i0,nright),dtype=cp.float64)
        # Remaining budget shared by A(aux,row_block) and B(aux,nright).
        pcap=max(1,int((budget-out.nbytes)//max(8,8*((i1-i0)+nright))))
        if aux_block_cap is not None:
            pcap=min(pcap,int(aux_block_cap))
        for p0 in range(0,naux,pcap):
            p1=min(naux,p0+pcap)
            # Slice is read on CPU, where HDF5 may be the scratch backend.
            aa=cp.asarray(np.asarray(A[p0:p1,i0:i1]),dtype=cp.float64)
            bb=cp.asarray(np.asarray(B[p0:p1,:]),dtype=cp.float64)
            out += cp.matmul(aa.T,bb)
        destination[i0:i1,:]=cp.asnumpy(out)
    cp.cuda.get_current_stream().synchronize()


def _write_mo_eris(eris,fac,*,mode,workspace_bytes,max_full_gib):
    from pyscf import lib
    no=eris.nocc
    nv=eris.fock.shape[0]-no
    p=nv*(nv+1)//2
    file_obj=lib.H5TmpFile()
    eris.feri = file_obj
    eris.oooo=file_obj.create_dataset('oooo',(no,no,no,no),'f8')
    eris.ovoo=file_obj.create_dataset('ovoo',(no,nv,no,no),'f8')
    eris.ovov=file_obj.create_dataset('ovov',(no,nv,no,nv),'f8')
    eris.ovvo=file_obj.create_dataset('ovvo',(no,nv,nv,no),'f8')
    eris.oovv=file_obj.create_dataset('oovv',(no,no,nv,nv),'f8')
    eris.ovvv=file_obj.create_dataset('ovvv',(no,nv,p),'f8')
    if mode == 'full':
        full_bytes=8*p*p
        if full_bytes>float(max_full_gib)*2**30:
            raise MemoryError(f'Packed vvvv would require {full_bytes/2**30:.2f} GiB of scratch; '
                              'select eri_mode=df or increase max_full_gib')
        eris.vvvv=file_obj.create_dataset('vvvv',(p,p),'f8')
    else:
        naux=fac['vv'].shape[0]
        eris.vvL=file_obj.create_dataset('vvL',(p,naux),'f8')
        # Read/write row blocks so factor scratch can be disk backed.
        for p0 in range(0,p,512):
            eris.vvL[p0:p0+512,:]=np.asarray(fac['vv'][:,p0:p0+512]).T
    # Explicit chemists' ordering; matches PySCF-forge's definitions:
    # (ov|oo), (ov|ov), (ov|vo), (oo|vv), (ov|vv) and packed (vv|vv).
    sectors=[('oooo','oo','oo',(no,no,no,no)),
             ('ovoo','ov','oo',(no,nv,no,no)),
             ('ovov','ov','ov',(no,nv,no,nv)),
             ('ovvo','ov','vo',(no,nv,nv,no)),
             ('ovvv','ov','vv',(no,nv,p))]
    for key,xa,xb,shape in sectors:
        dest=getattr(eris,key)
        left,right=fac[xa].shape[1],fac[xb].shape[1]
        flatrow=max(1,min(left,int((workspace_bytes//2)//max(8,8*right))))
        A,B=fac[xa],fac[xb]
        # Write each row slab directly to HDF5 using the original tensor
        # axes; never materialize the entire ovvv or ovoo sector on host.
        stride=shape[1]
        inner=shape[2:]
        for r0 in range(0,left,flatrow):
            r1=min(left,r0+flatrow)
            out=cp.zeros((r1-r0,right),dtype=cp.float64)
            pcap=max(1,int((workspace_bytes-out.nbytes)//max(8,8*((r1-r0)+right))))
            for k0 in range(0,A.shape[0],pcap):
                k1=min(A.shape[0],k0+pcap)
                ag=cp.asarray(np.asarray(A[k0:k1,r0:r1]))
                bg=cp.asarray(np.asarray(B[k0:k1,:]))
                out += ag.T @ bg
            slab=cp.asnumpy(out)
            pos=r0
            while pos<r1:
                first,second=divmod(pos,stride)
                count=min(r1-pos,stride-second)
                dest[first,second:second+count]=slab[pos-r0:pos-r0+count].reshape((count,)+inner)
                pos+=count
    # oovv is (oo|vv); build a packed-virtual intermediate in RAM or
    # HDF5 depending on the budget, then unpack one occupied pair at a
    # time. This avoids a full host-side (no**2,nv,nv) scratch allocation.
    nocc_pair_rows=no*no
    packed_bytes=8*nocc_pair_rows*p
    if packed_bytes<=workspace_bytes:
        temp=np.empty((nocc_pair_rows,p),dtype=np.float64)
        tmp_on_disk=False
    else:
        temp=file_obj.create_dataset('_oovv_tril_scratch',(nocc_pair_rows,p),'f8')
        tmp_on_disk=True
    _contract_rows(fac['oo'],fac['vv'],temp,workspace_bytes=workspace_bytes)
    tri=np.tril_indices(nv)
    for i in range(no):
        for j in range(no):
            packed=np.asarray(temp[i*no+j,:])
            sq=np.empty((nv,nv),dtype=np.float64)
            sq[tri[0],tri[1]]=packed
            sq[tri[1],tri[0]]=packed
            eris.oovv[i,j]=sq
    del temp
    if tmp_on_disk:
        del file_obj['_oovv_tril_scratch']
    if mode=='full':
        _contract_rows(fac['vv'],fac['vv'],eris.vvvv,workspace_bytes=workspace_bytes)
    cp.cuda.get_current_stream().synchronize()
    return eris


def fragment_ao2mo(mcc,mo_coeff=None,*,eri_mode='full',
                    workspace_mib=256,ao_block_mib=96,
                    host_scratch_mib=1024,max_full_gib=16):
    """Construct PySCF-forge-compatible LAS ERIs using GPU GEMMs.

    Preserves original PySCF-forge `_common_init_` for effective Fock,
    including all frozen-occupied/SCF environmental terms.  Requires an
    RHF molecular SCF with DF AO factors. Does not support complex/PBC.
    """
    if eri_mode not in ('full','df'):
        raise ValueError('eri_mode must be full or df (never auto)')
    if min(workspace_mib,ao_block_mib,host_scratch_mib,max_full_gib)<=0:
        raise ValueError('All integral workspace budgets must be positive')
    from pyscf.lno import lnoccsd as impl
    if hasattr(mcc._scf,'kpt') or hasattr(mcc._scf,'kpts'):
        raise NotImplementedError('02 supports molecular RHF only, not PBC')
    df=getattr(mcc._scf,'with_df',None)
    if df is None:
        raise ValueError('02 requires DF-RHF AO factors (mf.with_df)')
    if eri_mode=='df':
        eris=impl._DFChemistsERIs()
    else:
        eris=impl._ChemistsERIs()
    # This calls the exact original Fock/HF initialization and removes
    # frozen MOs via _mo_without_core. Do NOT form Fock from LAS occ only.
    eris._common_init_(mcc,mo_coeff)
    coeff=np.asarray(eris.mo_coeff)
    if coeff.dtype!=np.float64 or not np.isrealobj(coeff):
        raise NotImplementedError('02 supports real float64 molecular RHF only')
    nao,nmo=coeff.shape
    no=eris.nocc
    nv=nmo-no
    if no<=0 or nv<=0:
        raise ValueError('Empty LAS occupied or virtual subspace')
    naux=int(df.get_naoaux())
    if eri_mode=='full':
        npair=nv*(nv+1)//2
        if (8*npair*npair)>float(max_full_gib)*2**30:
            raise MemoryError(
                f'Packed vvvv exceeds {max_full_gib:g} GiB storage budget; '
                'use eri_mode="df" or increase max_full_gib')
    factors,scratch,host_bytes=_allocate_factor_scratch(naux,no,nv,host_scratch_mib)
    t0=time.perf_counter()
    # Determine number of AO factors by workspace. PySCF may choose a larger
    # default with_df.loop() block; split the yielded block without copy.
    bytes_per_factor=8*(nao*nao + nao*nmo + nmo*nmo + nao*(nao+1)//2)
    block_cap=max(1,int(ao_block_mib*2**20)//bytes_per_factor)
    tri=cp.tril_indices(nv)
    aux_offset=0
    for ao_block in df.loop():
        raw=np.asarray(ao_block)
        if raw.ndim!=2 or raw.shape[1]!=nao*(nao+1)//2 or raw.dtype!=np.float64:
            raise ValueError(f'PySCF DF AO blocks must be float64 s2 packed; got {raw.shape}, {raw.dtype}')
        for start in range(0,raw.shape[0],block_cap):
            stop=min(raw.shape[0],start+block_cap)
            npart=stop-start
            mo=transform_ao_df_block(raw[start:stop],coeff,packed=True)
            factors['oo'][aux_offset:aux_offset+npart,:]=cp.asnumpy(mo[:,:no,:no].reshape(npart,-1))
            factors['ov'][aux_offset:aux_offset+npart,:]=cp.asnumpy(mo[:,:no,no:].reshape(npart,-1))
            factors['vo'][aux_offset:aux_offset+npart,:]=cp.asnumpy(mo[:,no:,:no].reshape(npart,-1))
            factors['vv'][aux_offset:aux_offset+npart,:]=cp.asnumpy(mo[:,no+tri[0],no+tri[1]])
            aux_offset+=npart
            del mo
    if aux_offset!=naux:
        raise RuntimeError(f'DF loop produced {aux_offset} aux vectors; with_df reports {naux}')
    ao2mo_s=time.perf_counter()-t0
    result=_write_mo_eris(eris,factors,mode=eri_mode,
                         workspace_bytes=int(workspace_mib)*2**20,
                         max_full_gib=max_full_gib)
    cp.cuda.get_current_stream().synchronize()
    result._gpu_ao2mo_info={
        'backend':'gpu','eri_mode':eri_mode,'nao':nao,'nmo':nmo,
        'naux':naux,'ao_df_mo_transform_s':ao2mo_s,
        'total_gpu_ao2mo_s':time.perf_counter()-t0,
        'factor_scratch_bytes':host_bytes,
        'scratch_storage':'hdf5' if scratch is not None else 'ram',
        'gpu_workspace_mib':int(workspace_mib),
        'ao_block_mib':int(ao_block_mib),
    }
    # HDF5 scratch must remain accessible until all block products finished.
    if scratch is not None:
        scratch.close()
    return result


def compare_eri_objects(gpu,reference,*,eri_mode):
    """Return per-tensor max absolute errors for the ORIGINAL CPU AO2MO.

    This compares every sector, not merely CCSD energies or LAS overlap.
    The frozen-core/environmental Fock and MO denominators are included.
    """
    keys=['fock','mo_energy','oooo','ovoo','ovov','ovvo','oovv','ovvv']
    keys.append('vvvv' if eri_mode=='full' else 'vvL')
    errs={}
    for k in keys:
        ga=np.asarray(getattr(gpu,k))
        ref=np.asarray(getattr(reference,k))
        if ga.shape!=ref.shape:
            raise AssertionError(f'AO2MO mismatch in {k}: {ga.shape} != {ref.shape}')
        errs[k]=float(np.max(np.abs(ga-ref))) if ga.size else 0.0
    return errs


def transform_ovl(
    with_df, coeff_occ, coeff_vir, out, *,
    workspace_mib=256, host_scratch_mib=256,
):
    """Stream DF factors and write the (ov|P) matrix into NumPy or HDF5.

    `with_df.loop(blksize)` returns auxiliary-major AO factors in PySCF's
    packed lower-triangle `s2` layout (real RHF); full AO-square layout is
    supported as well.  `out` is NumPy array or h5py dataset.  Avoid any
    persistent O(nAO^4) or O(nAux*nAO^2) intermediates.
    """
    co = np.asarray(coeff_occ)
    cv = np.asarray(coeff_vir)
    if co.dtype != np.float64 or cv.dtype != np.float64:
        raise NotImplementedError('04 supports only real float64 molecular RHF')
    if co.ndim != 2 or cv.ndim != 2 or co.shape[0] != cv.shape[0]:
        raise ValueError('MO coefficient matrices must have matching AO dimensions')
    nao, no = co.shape
    nv = cv.shape[1]
    naux = int(with_df.get_naoaux())
    if out.shape != (no, nv, naux):
        raise ValueError(f'Unexpected ovL storage shape {out.shape}, expected {(no,nv,naux)}')
    if np.dtype(out.dtype) != np.dtype(np.float64):
        raise ValueError('ovL output must be float64')
    if workspace_mib < 1 or host_scratch_mib < 1:
        raise ValueError('Workspace and host-scratch budgets must be positive')
    # GPU transient: unpacked AO L, intermediate L@Cvir, transformed block,
    # plus packed input and a conservative 2x margin for BLAS workspaces.
    bytes_per_aux = 8*(nao*nao + nao*nv + no*nv + nao*(nao+1)//2)
    gpu_budget = workspace_mib * 2**20
    aux_block = min(naux, max(1, int(gpu_budget // (2*bytes_per_aux))))
    if 2*bytes_per_aux > gpu_budget:
        raise MemoryError('04 GPU workspace cannot hold a single unpacked AO factor; '
                          'increase global_ovl_workspace_mib')
    # With HDF5, write complete occupied slabs, not millions of thin HDF5
    # hyperslabs.  Full NumPy storage can be written auxiliary-slab-wise.
    is_hdf5 = not isinstance(out, np.ndarray)
    if is_hdf5:
        occ_block = min(no, max(1, (host_scratch_mib*2**20)//max(1, 8*nv*naux)))
        if 8*nv*naux > host_scratch_mib*2**20:
            raise MemoryError('Host scratch too small for one occupied HDF5 slab')
    else:
        occ_block = no
    ti, tj = np.tril_indices(nao)
    ii = cp.asarray(ti)
    jj = cp.asarray(tj)
    cvir = cp.asarray(cv, order='C')
    # Keep occupied coefficients on the device and index only the current
    # occupied slab; this limits transferred coefficient data and GPU memory.
    cocc = cp.asarray(co, order='C')
    max_block_seen = 0
    n_df_passes = 0
    n_df_chunks = 0
    start = time.perf_counter()
    for i0 in range(0,no,occ_block):
        i1 = min(no,i0+occ_block)
        if is_hdf5:
            host_slab = np.empty((i1-i0,nv,naux),dtype=np.float64)
        p = 0
        n_df_passes += 1
        # with_df.loop is the official PySCF DF factor iterator; do not use
        # an AO shell integral assumption or assume a built _cderi ndarray.
        for raw in with_df.loop(blksize=aux_block):
            raw = np.asarray(raw)
            if raw.ndim != 2:
                raise ValueError('DF blocks must be two-dimensional auxiliary-major arrays')
            if raw.dtype != np.float64:
                raise NotImplementedError('Complex/signed periodic DF blocks are unsupported')
            for k0 in range(0,raw.shape[0],aux_block):
                slab = raw[k0:k0+aux_block]
                b = slab.shape[0]
                if p+b > naux:
                    raise RuntimeError('DF loop returned more auxiliary functions than get_naoaux')
                src = cp.asarray(slab)
                if src.shape[1] == nao*(nao+1)//2:
                    # Expand the real symmetric packed AO factor on device.
                    aofac = cp.empty((b,nao,nao),dtype=cp.float64)
                    aofac[:,ii,jj] = src
                    aofac[:,jj,ii] = src
                elif src.shape[1] == nao*nao:
                    aofac = src.reshape(b,nao,nao)
                else:
                    raise ValueError(f'Unexpected DF AO factor width {src.shape[1]}')
                # Two GEMMs: L^P_{mu,nu} C_{nu,a}, then C_{mu,i} L^P_{mu,a}
                # -> (naux_block, nocc_block, nvir).  Real RHF, no conjugation.
                partial = cp.matmul(aofac,cvir)
                transformed = cp.matmul(cocc[:,i0:i1].T[None,:,:],partial)
                block_host = cp.asnumpy(transformed.transpose(1,2,0))
                if is_hdf5:
                    host_slab[:,:,p:p+b] = block_host
                else:
                    out[i0:i1,:,p:p+b] = block_host
                p += b
                n_df_chunks += 1
                max_block_seen = max(max_block_seen,b)
                del src,aofac,partial,transformed,block_host
        if p != naux:
            raise RuntimeError(f'DF loop returned {p} of expected {naux} auxiliary functions')
        if is_hdf5:
            out[i0:i1,:,:] = host_slab
            del host_slab
    cp.cuda.get_current_stream().synchronize()
    return {
        'backend':'gpu',
        'nao':int(nao),'nocc':int(no),'nvir':int(nv),'naux':int(naux),
        'output_storage':'hdf5' if is_hdf5 else 'ram',
        'aux_block_requested':int(aux_block),
        'aux_block_max_seen':int(max_block_seen),
        'occupied_block':int(occ_block),
        'df_stream_passes':int(n_df_passes),
        'df_chunks_processed':int(n_df_chunks),
        'workspace_mib':int(workspace_mib),
        'host_scratch_mib':int(host_scratch_mib),
        'elapsed_transform_s':float(time.perf_counter()-start),
        'ovL_bytes':int(no*nv*naux*8),
    }


def build_global_eris(mylno, *, workspace_mib=256,
                      host_scratch_mib=256):
    """Build PySCF-forge's original global LNO ERI interface with GPU ovL.

    The original class implements get_occ_blk, xform_occ, etc.; constructing
    it and only replacing its `build` operation guarantees compatibility
    with subsequent LNO MP2 density construction.
    """
    from pyscf.lno import lno as lno_impl
    from pyscf import lib
    import h5py
    if mylno.with_df is None:
        raise RuntimeError('04 requires an RHF with density fitting enabled')
    occ,vir = mylno.split_mo_coeff()[1:3]
    occ=np.asarray(occ)
    vir=np.asarray(vir)
    if occ.dtype != np.float64 or vir.dtype != np.float64:
        raise NotImplementedError('04 requires real molecular RHF')
    naux=int(mylno.with_df.get_naoaux())
    nocc,nvir=occ.shape[1],vir.shape[1]
    ovl_bytes=nocc*nvir*naux*8
    mem_now=mylno.max_memory-lib.current_memory()[0]
    existing=getattr(mylno,'_ovL',None)
    if existing is not None:
        # Reuse/restart is an I/O operation; the original implementation
        # checks the shape and opens the stored file without transforming.
        if not isinstance(existing,str):
            raise NotImplementedError('04 only supports string paths for precomputed ovL')
        obj=lno_impl._LNODFOUTCOREERIS(
            mylno.with_df,occ,vir,mylno.max_memory,ovL=existing,
            verbose=mylno.verbose,stdout=mylno.stdout)
        obj.build()
        return obj,{
            'backend':'reused-existing','nocc':int(nocc),'nvir':int(nvir),
            'naux':naux,'ovL_bytes':int(ovl_bytes),
            'output_storage':'hdf5','elapsed_transform_s':0.0,
        }
    outcore=bool(getattr(mylno,'force_outcore_ao2mo',False)
                 or getattr(mylno,'_ovL_to_save',None) is not None
                 or ovl_bytes/(2**20) > mem_now*.5)
    if outcore:
        obj=lno_impl._LNODFOUTCOREERIS(
            mylno.with_df,occ,vir,mylno.max_memory,
            ovL=None,ovL_to_save=mylno._ovL_to_save,
            verbose=mylno.verbose,stdout=mylno.stdout)
        target=getattr(mylno,'_ovL_to_save',None)
        obj.feri=h5py.File(target,'w') if isinstance(target,str) else lib.H5TmpFile()
        obj.ovL=obj.feri.create_dataset('ovL',(nocc,nvir,naux),
                                        dtype=np.float64,chunks=(1,nvir,naux))
    else:
        obj=lno_impl._LNODFINCOREERIS(
            mylno.with_df,occ,vir,mylno.max_memory,
            verbose=mylno.verbose,stdout=mylno.stdout)
        obj.ovL=np.empty((nocc,nvir,naux),dtype=np.float64)
    try:
        info=transform_ovl(mylno.with_df,occ,vir,obj.ovL,
                           workspace_mib=workspace_mib,
                           host_scratch_mib=host_scratch_mib)
    except BaseException:
        if outcore:
            obj.feri.close()
        raise
    obj._gpu_global_ovl_info=info
    return obj,info


def compare_global_eris(gpu, reference, *, occ_block=2):
    """Maximum absolute elementwise ovL deviation with bounded host scratch."""
    if (gpu.nocc,gpu.nvir,gpu.naux)!=(reference.nocc,reference.nvir,reference.naux):
        raise AssertionError('CPU/GPU global ovL dimensions differ')
    max_abs=0.0
    for i0 in range(0,gpu.nocc,occ_block):
        i1=min(gpu.nocc,i0+occ_block)
        delta=np.asarray(gpu.get_occ_blk(i0,i1))-np.asarray(reference.get_occ_blk(i0,i1))
        if delta.size:
            max_abs=max(max_abs,float(np.max(np.abs(delta))))
    return max_abs
