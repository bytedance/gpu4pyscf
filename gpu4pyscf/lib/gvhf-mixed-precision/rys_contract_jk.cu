/*
 * Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "vhf.cuh"
#include "rys_roots.cu"
#include "create_tasks.cu"

// TODO: benchmark performance for 34, 36, 41, 43, 45, 47, 51, 57
#define GOUT_WIDTH      42

template<typename FloatType>
__device__
static void rys_jk_general(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                           ShellQuartet *shl_quartet_idx, int ntasks)
{
    // sq is short for shl_quartet
    int sq_id = threadIdx.x;
    int nsq_per_block = blockDim.x;
    int gout_id = threadIdx.y;
    int gout_stride = blockDim.y;
    int li = bounds.li;
    int lj = bounds.lj;
    int lk = bounds.lk;
    int ll = bounds.ll;
    int nfi = bounds.nfi;
    int nfk = bounds.nfk;
    int nfij = bounds.nfij;
    int nfkl = bounds.nfkl;
    int iprim = bounds.iprim;
    int jprim = bounds.jprim;
    int kprim = bounds.kprim;
    int lprim = bounds.lprim;
    int lij = li + lj;
    int lkl = lk + ll;
    int nroots = bounds.nroots;
    int stride_j = bounds.stride_j;
    int stride_k = bounds.stride_k;
    int stride_l = bounds.stride_l;
    int g_size = stride_l * (ll + 1);
    int *idx_ij = c_g_pair_idx + c_g_pair_offsets[li*LMAX1+lj];
    int *idy_ij = idx_ij + nfij;
    int *idz_ij = idy_ij + nfij;
    int *idx_kl = c_g_pair_idx + c_g_pair_offsets[lk*LMAX1+ll];
    int *idy_kl = idx_kl + nfkl;
    int *idz_kl = idy_kl + nfkl;
    int *bas = envs.bas;
    int *ao_loc = envs.ao_loc;
    int nbas = envs.nbas;
    int nao = ao_loc[nbas];
    //double *env = c_env;
    double *env = envs.env;
    FloatType omega = env[PTR_RANGE_OMEGA];

    extern __shared__ char shared_memory[];
    FloatType *rw_cache = (FloatType *)shared_memory;
    FloatType *rw = rw_cache + sq_id;
    FloatType *g = rw + nsq_per_block * nroots*2;
    FloatType *gx = g;
    FloatType *gy = g + nsq_per_block * g_size;
    FloatType *gz = gy + nsq_per_block * g_size;
    FloatType *rjri = gz + nsq_per_block * g_size;
    FloatType *rlrk = rjri + nsq_per_block * 3;
    FloatType *Rpq = rlrk + nsq_per_block * 3;
    FloatType *cicj_cache = Rpq + nsq_per_block * 3;
    FloatType gout[GOUT_WIDTH];

    for (int task0 = 0; task0 < ntasks; task0 += nsq_per_block) {
        __syncthreads();
        int task_id = task0 + sq_id;
        FloatType fac_sym = PI_FAC;
        ShellQuartet sq;
        if (task_id >= ntasks) {
            // To avoid __syncthreads blocking blocking idle warps, all remaining
            // threads compute a valid shell quartet with zero normalization factor
            sq = shl_quartet_idx[0];
            fac_sym = 0.;
        } else {
            sq = shl_quartet_idx[task_id];
        }
        int ish = sq.i;
        int jsh = sq.j;
        int ksh = sq.k;
        int lsh = sq.l;
        //int sh_ij = (ish % TILE) * TILE + (jsh % TILE);
        if (ish == jsh) fac_sym *= static_cast<FloatType>(0.5);
        if (ksh == lsh) fac_sym *= static_cast<FloatType>(0.5);
        if (ish*nbas+jsh == ksh*nbas+lsh) fac_sym *= static_cast<FloatType>(0.5);
        int i0 = ao_loc[ish];
        int j0 = ao_loc[jsh];
        int k0 = ao_loc[ksh];
        int l0 = ao_loc[lsh];
        double *expi = env + bas[ish*BAS_SLOTS+PTR_EXP];
        double *expj = env + bas[jsh*BAS_SLOTS+PTR_EXP];
        double *expk = env + bas[ksh*BAS_SLOTS+PTR_EXP];
        double *expl = env + bas[lsh*BAS_SLOTS+PTR_EXP];
        double *ci = env + bas[ish*BAS_SLOTS+PTR_COEFF];
        double *cj = env + bas[jsh*BAS_SLOTS+PTR_COEFF];
        double *ck = env + bas[ksh*BAS_SLOTS+PTR_COEFF];
        double *cl = env + bas[lsh*BAS_SLOTS+PTR_COEFF];
        double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double *rl = env + bas[lsh*BAS_SLOTS+PTR_BAS_COORD];
        const FloatType xjxi = static_cast<FloatType>(rj[0]) - static_cast<FloatType>(ri[0]);
        const FloatType yjyi = static_cast<FloatType>(rj[1]) - static_cast<FloatType>(ri[1]);
        const FloatType zjzi = static_cast<FloatType>(rj[2]) - static_cast<FloatType>(ri[2]);
        const FloatType rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
        if (gout_id == 0) {
            const FloatType xlxk = static_cast<FloatType>(rl[0]) - static_cast<FloatType>(rk[0]);
            const FloatType ylyk = static_cast<FloatType>(rl[1]) - static_cast<FloatType>(rk[1]);
            const FloatType zlzk = static_cast<FloatType>(rl[2]) - static_cast<FloatType>(rk[2]);
            rjri[0*nsq_per_block] = xjxi;
            rjri[1*nsq_per_block] = yjyi;
            rjri[2*nsq_per_block] = zjzi;
            rlrk[0*nsq_per_block] = xlxk;
            rlrk[1*nsq_per_block] = ylyk;
            rlrk[2*nsq_per_block] = zlzk;
        }
        for (int ij = gout_id; ij < iprim*jprim; ij += gout_stride) {
            int ip = ij / jprim;
            int jp = ij % jprim;
            const FloatType ai = static_cast<FloatType>(expi[ip]);
            const FloatType aj = static_cast<FloatType>(expj[jp]);
            const FloatType aij = ai + aj;
            const FloatType theta_ij = ai * aj / aij;
            const FloatType Kab = MixedPrecisionOperator<FloatType>::fp_exp(-theta_ij * rr_ij);
            cicj_cache[ij*nsq_per_block] = fac_sym * static_cast<FloatType>(ci[ip]) * static_cast<FloatType>(cj[jp]) * Kab;
        }
        for (int gout_start = 0; gout_start < nfij*nfkl; gout_start+=gout_stride*GOUT_WIDTH) {
#pragma unroll
        for (int n = 0; n < GOUT_WIDTH; ++n) { gout[n] = 0; }

        for (int klp = 0; klp < kprim*lprim; ++klp) {
            int kp = klp / lprim;
            int lp = klp % lprim;
            const FloatType ak = static_cast<FloatType>(expk[kp]);
            const FloatType al = static_cast<FloatType>(expl[lp]);
            const FloatType akl = ak + al;
            const FloatType al_akl = al / akl;
            __syncthreads();
            if (gout_id == 0) {
                const FloatType xlxk = rlrk[0*nsq_per_block];
                const FloatType ylyk = rlrk[1*nsq_per_block];
                const FloatType zlzk = rlrk[2*nsq_per_block];
                const FloatType rr_kl = xlxk*xlxk + ylyk*ylyk + zlzk*zlzk;
                const FloatType theta_kl = ak * al / akl;
                const FloatType Kcd = MixedPrecisionOperator<FloatType>::fp_exp(-theta_kl * rr_kl);
                const FloatType ckcl = static_cast<FloatType>(ck[kp]) * static_cast<FloatType>(cl[lp]) * Kcd;
                gx[0] = ckcl;
            }
            for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                int ip = ijp / jprim;
                int jp = ijp % jprim;
                const FloatType ai = static_cast<FloatType>(expi[ip]);
                const FloatType aj = static_cast<FloatType>(expj[jp]);
                const FloatType aij = ai + aj;
                const FloatType aj_aij = aj / aij;
                const FloatType xij = static_cast<FloatType>(ri[0]) + rjri[0*nsq_per_block] * aj_aij;
                const FloatType yij = static_cast<FloatType>(ri[1]) + rjri[1*nsq_per_block] * aj_aij;
                const FloatType zij = static_cast<FloatType>(ri[2]) + rjri[2*nsq_per_block] * aj_aij;
                const FloatType xkl = static_cast<FloatType>(rk[0]) + rlrk[0*nsq_per_block] * al_akl;
                const FloatType ykl = static_cast<FloatType>(rk[1]) + rlrk[1*nsq_per_block] * al_akl;
                const FloatType zkl = static_cast<FloatType>(rk[2]) + rlrk[2*nsq_per_block] * al_akl;
                const FloatType xpq = xij - xkl;
                const FloatType ypq = yij - ykl;
                const FloatType zpq = zij - zkl;
                __syncthreads();
                if (gout_id == 0) {
                    Rpq[0*nsq_per_block] = xpq;
                    Rpq[1*nsq_per_block] = ypq;
                    Rpq[2*nsq_per_block] = zpq;
                    const FloatType cicj = cicj_cache[ijp*nsq_per_block];
                    gy[0] = cicj / (aij*akl* MixedPrecisionOperator<FloatType>::fp_sqrt(aij+akl));
                }
                const FloatType rr = xpq*xpq + ypq*ypq + zpq*zpq;
                const FloatType theta = aij * akl / (aij + akl);
                rys_roots_rs_mixed_precision<FloatType>(nroots, theta, rr, omega, rw, nsq_per_block, gout_id, gout_stride);
                FloatType s0x, s1x, s2x;
                for (int irys = 0; irys < nroots; ++irys) {
                    __syncthreads();
                    if (gout_id == 0) {
                        gz[0] = rw[(irys*2+1)*nsq_per_block];
                    }
                    const FloatType rt = rw[irys*2*nsq_per_block];
                    const FloatType rt_aa = rt / (aij + akl);

                    // TRR
                    //for i in range(lij):
                    //    trr(i+1,0) = c0 * trr(i,0) + i*b10 * trr(i-1,0)
                    //for k in range(lkl):
                    //    for i in range(lij+1):
                    //        trr(i,k+1) = c0p * trr(i,k) + k*b01 * trr(i,k-1) + i*b00 * trr(i-1,k)
                    if (lij > 0) {
                        const FloatType rt_aij = rt_aa * akl;
                        const FloatType b10 = static_cast<FloatType>(0.5)/aij * (static_cast<FloatType>(1.0) - rt_aij);
                        __syncthreads();
                        // gx(0,n+1) = c0*gx(0,n) + n*b10*gx(0,n-1)
                        for (int n = gout_id; n < 3; n += gout_stride) {
                            FloatType *_gx = g + n * g_size * nsq_per_block;
                            const FloatType Rpa = rjri[n*nsq_per_block] * aj_aij;
                            const FloatType c0x = Rpa - rt_aij * Rpq[n*nsq_per_block];
                            s0x = _gx[0];
                            s1x = c0x * s0x;
                            _gx[nsq_per_block] = s1x;
                            for (int i = 1; i < lij; ++i) {
                                s2x = c0x * s1x + i * b10 * s0x;
                                _gx[(i+1)*nsq_per_block] = s2x;
                                s0x = s1x;
                                s1x = s2x;
                            }
                        }
                    }

                    if (lkl > 0) {
                        const FloatType rt_akl = rt_aa * aij;
                        const FloatType b00 = static_cast<FloatType>(0.5) * rt_aa;
                        const FloatType b01 = static_cast<FloatType>(0.5)/akl * (static_cast<FloatType>(1.0) - rt_akl);
                        int lij3 = (lij+1)*3;
                        for (int n = gout_id; n < lij3+gout_id; n += gout_stride) {
                            __syncthreads();
                            int i = n / 3; //for i in range(lij+1):
                            int _ix = n % 3; // TODO: remove _ix for nroots > 2
                            FloatType *_gx = g + (i + _ix * g_size) * nsq_per_block;
                            const FloatType Rqc = rlrk[_ix*nsq_per_block] * al_akl;
                            const FloatType cpx = Rqc + rt_akl * Rpq[_ix*nsq_per_block];
                            //for i in range(lij+1):
                            //    trr(i,1) = c0p * trr(i,0) + i*b00 * trr(i-1,0)
                            if (n < lij3) {
                                s0x = _gx[0];
                                s1x = cpx * s0x;
                                if (i > 0) {
                                    s1x += i * b00 * _gx[-nsq_per_block];
                                }
                                _gx[stride_k*nsq_per_block] = s1x;
                            }

                            //for k in range(1, lkl):
                            //    for i in range(lij+1):
                            //        trr(i,k+1) = cp * trr(i,k) + k*b01 * trr(i,k-1) + i*b00 * trr(i-1,k)
                            for (int k = 1; k < lkl; ++k) {
                                __syncthreads();
                                if (n < lij3) {
                                    s2x = cpx*s1x + k*b01*s0x;
                                    if (i > 0) {
                                        s2x += i * b00 * _gx[(k*stride_k-1)*nsq_per_block];
                                    }
                                    _gx[(k*stride_k+stride_k)*nsq_per_block] = s2x;
                                    s0x = s1x;
                                    s1x = s2x;
                                }
                            }
                        }
                    }

                    // hrr
                    // g(i,j+1) = rirj * g(i,j) +  g(i+1,j)
                    // g(...,k,l+1) = rkrl * g(...,k,l) + g(...,k+1,l)
                    if (lj > 0) {
                        __syncthreads();
                        if (task_id < ntasks) {
                            int lkl3 = (lkl+1)*3;
                            for (int m = gout_id; m < lkl3; m += gout_stride) {
                                int k = m / 3;
                                int _ix = m % 3;
                                const FloatType xjxi = rjri[_ix*nsq_per_block];
                                FloatType *_gx = g + (_ix*g_size + k*stride_k) * nsq_per_block;
                                for (int j = 0; j < lj; ++j) {
                                    int ij = (lij-j) + j*stride_j;
                                    s1x = _gx[ij*nsq_per_block];
                                    for (--ij; ij >= j*stride_j; --ij) {
                                        s0x = _gx[ij*nsq_per_block];
                                        _gx[(ij+stride_j)*nsq_per_block] = s1x - xjxi * s0x;
                                        s1x = s0x;
                                    }
                                }
                            }
                        }
                    }
                    if (ll > 0) {
                        __syncthreads();
                        if (task_id < ntasks) {
                            for (int n = gout_id; n < stride_k*3; n += gout_stride) {
                                int i = n / 3;
                                int _ix = n % 3;
                                const FloatType xlxk = rlrk[_ix*nsq_per_block];
                                FloatType *_gx = g + (_ix*g_size + i) * nsq_per_block;
                                for (int l = 0; l < ll; ++l) {
                                    int kl = (lkl-l)*stride_k + l*stride_l;
                                    s1x = _gx[kl*nsq_per_block];
                                    for (kl-=stride_k; kl >= l*stride_l; kl-=stride_k) {
                                        s0x = _gx[kl*nsq_per_block];
                                        _gx[(kl+stride_l)*nsq_per_block] = s1x - xlxk * s0x;
                                        s1x = s0x;
                                    }
                                }
                            }
                        }
                    }

                    __syncthreads();
#pragma unroll
                    for (int n = 0; n < GOUT_WIDTH; ++n) {
                        int ijkl = gout_start + n*gout_stride+gout_id;
                        int kl = ijkl / nfij;
                        int ij = ijkl % nfij;
                        if (kl >= nfkl) break;
                        int addrx = (idx_ij[ij] + idx_kl[kl] * stride_k) * nsq_per_block;
                        int addry = (idy_ij[ij] + idy_kl[kl] * stride_k) * nsq_per_block;
                        int addrz = (idz_ij[ij] + idz_kl[kl] * stride_k) * nsq_per_block;
                        gout[n] += gx[addrx] * gy[addry] * gz[addrz];
                    }
                }
            }
        }
        if (task_id >= ntasks) {
            continue;
        }
        double *dm = jk.dm;
        double *vj = jk.vj;
        double *vk = jk.vk;
        int do_j = vj != NULL;
        int do_k = vk != NULL;
        for (int i_dm = 0; i_dm < jk.n_dm; ++i_dm) {
#pragma unroll
            for (int n = 0; n < GOUT_WIDTH; ++n) {
                int ijkl = (gout_start + n*gout_stride+gout_id);
                int kl = ijkl / nfij;
                int ij = ijkl % nfij;
                if (kl >= nfkl) break;
                double s = static_cast<double>(gout[n]);
                int i = ij % nfi;
                int j = ij / nfi;
                int k = kl % nfk;
                int l = kl / nfk;
                int _i = i + i0;
                int _j = j + j0;
                int _k = k + k0;
                int _l = l + l0;
                if (do_j) {
                    int _ji = _j*nao+_i;
                    int _lk = _l*nao+_k;
                    atomicAdd(vj+_lk, s * dm[_ji]);
                    atomicAdd(vj+_ji, s * dm[_lk]);
                }
                if (do_k) {
                    int _jl = _j*nao+_l;
                    int _jk = _j*nao+_k;
                    int _il = _i*nao+_l;
                    int _ik = _i*nao+_k;
                    atomicAdd(vk+_ik, s * dm[_jl]);
                    atomicAdd(vk+_il, s * dm[_jk]);
                    atomicAdd(vk+_jk, s * dm[_il]);
                    atomicAdd(vk+_jl, s * dm[_ik]);
                }
            }
            vj += nao * nao;
            vk += nao * nao;
            dm += nao * nao;
        }
    } }
}

template <typename FloatType>
__global__
void rys_jk_kernel(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds,
                   ShellQuartet *pool, uint32_t *batch_head)
{
    int b_id = blockIdx.x;
    int t_id = threadIdx.y * blockDim.x + threadIdx.x;
    ShellQuartet *shl_quartet_idx = pool + b_id * QUEUE_DEPTH;
    extern __shared__ int batch_id[];
    if (t_id == 0) {
        batch_id[0] = atomicAdd(batch_head, 1);
    }
    __syncthreads();
    int nbatches_kl = (bounds.ntile_kl_pairs + TILES_IN_BATCH - 1) / TILES_IN_BATCH;
    int nbatches = bounds.ntile_ij_pairs * nbatches_kl;
    while (batch_id[0] < nbatches) {
        int batch_ij = batch_id[0] / nbatches_kl;
        int batch_kl = batch_id[0] % nbatches_kl;
        double omega = envs.env[PTR_RANGE_OMEGA];
        int ntasks;
        if (omega >= 0) {
            ntasks = _fill_jk_tasks_mixed_precision<FloatType>(shl_quartet_idx, envs, jk, bounds,
                                                               batch_ij, batch_kl);
        } else {
            ntasks = _fill_sr_jk_tasks(shl_quartet_idx, envs, jk, bounds,
                                       batch_ij, batch_kl);
        }
        if (ntasks > 0) {
            rys_jk_general<FloatType>(envs, jk, bounds, shl_quartet_idx, ntasks);
            __syncthreads();
        }
        if (t_id == 0) {
            batch_id[0] = atomicAdd(batch_head, 1);
            atomicAdd(batch_head+1, ntasks);
        }
        __syncthreads();
    }
}

template __global__ void rys_jk_kernel<double>(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds, ShellQuartet *pool, uint32_t *batch_head);
template __global__ void rys_jk_kernel< float>(RysIntEnvVars envs, JKMatrix jk, BoundsInfo bounds, ShellQuartet *pool, uint32_t *batch_head);
