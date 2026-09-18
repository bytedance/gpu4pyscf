/*
 * Copyright 2025 The PySCF Developers. All Rights Reserved.
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
#include <cuda.h>
#include <cuda_runtime.h>
#include "gvhf-rys/rys_roots.cu"
#include "gvhf-rys/rys_contract_k.cuh"
#include "gvhf-rys/build_rys_gxyz.cuh"

#define THREADS 256
#define GOUT_WIDTH 54

// Logarithmic distance bound for (ij|k), with no density-matrix condition.
// shell_bounds stores min exponent, max exponent and log(sum |coeff|).
// Replacing each contracted shell by its diffuse Gaussian envelope bounds
// all primitives, including contractions with alternating signs.
__device__ static
bool sr_int3c2e_task_keep(int ish, int jsh, int ksh, RysIntEnvVars envs,
                        float *shell_bounds, double omega2, float log_cutoff)
{
    if (log_cutoff == -INFINITY) return true;
    int *bas = envs.bas;
    double *env = envs.env;
    double ai = shell_bounds[ish*3];
    double aj = shell_bounds[jsh*3];
    double ak = shell_bounds[ksh*3];
    double aij = ai + aj;
    double aj_aij = aj / aij;
    double theta_ij = ai * aj_aij;
    // The auxiliary shell replaces the second Gaussian product in RSJK.
    double theta = 1. / (1./aij + 1./ak + 1./omega2);
    double *ri = env + bas[ish*BAS_SLOTS+PTR_BAS_COORD];
    double *rj = env + bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
    double *rk = env + bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
    double rr_ij = 0, rr = 0, rr_ik = 0, rr_jk = 0;
    for (int x = 0; x < 3; ++x) {
        double ji = rj[x] - ri[x];
        double pk = ri[x] + aj_aij*ji - rk[x];
        rr_ij += ji*ji;
        rr += pk*pk;
        rr_ik += (ri[x]-rk[x])*(ri[x]-rk[x]);
        rr_jk += (rj[x]-rk[x])*(rj[x]-rk[x]);
    }
    int li = bas[ish*BAS_SLOTS+ANG_OF];
    int lj = bas[jsh*BAS_SLOTS+ANG_OF];
    int lk = bas[ksh*BAS_SLOTS+ANG_OF];
    // Gaussian moment bounds with one extra power for the derivative.
    // The shifts cover the entire SR Rys interval, not just its lower end.
    double order = li + lj + lk + 1;
    double dij = sqrt(rr_ij), r = sqrt(rr);
    double shift_ij = ak/(aij+ak)*r;
    double shift_k = aij/(aij+ak)*r;
    double di = fmax(1., aj_aij*dij + shift_ij + sqrt(order/(2*aij)));
    double dj = fmax(1., ai/aij*dij + shift_ij + sqrt(order/(2*aij)));
    double dk = fmax(1., shift_k + sqrt(order/(2*ak)));
    // F0_SR <= exp(-theta*R_PK^2). Omitting the additional inverse-distance
    // decay used by RSJK keeps this bound conservative near coincident centers.
    double bound = shell_bounds[ish*3+2] + shell_bounds[jsh*3+2]
                 + shell_bounds[ksh*3+2] + log(PI_FAC)
                 - log(aij*ak*sqrt(aij+ak))
                 + li*log(di) + lj*log(dj) + lk*log(dk)
                 - theta_ij*rr_ij - theta*rr;
    // Bound both orbital derivatives and their sum (the auxiliary response).
    bound += log(2*shell_bounds[ish*3+1]*di + li
               + 2*shell_bounds[jsh*3+1]*dj + lj);
    // Strain also contracts the derivatives with inter-center distances.
    bound += log(1. + sqrt(rr_ik) + sqrt(rr_jk));
    return bound + 1. > log_cutoff; // margin for single-precision shell data
}

// Compact one bounded tile, as in the RSJK task generator. All threads must
// participate, including those beyond the final candidate. The scan scratch
// is reused by the integral recurrence after this function returns.
__device__ static
int fill_sr_int3c2e_tasks(int *tasks, int *scan, int first, int last,
                        int shl_pair0, int nksh, int ksh0,
                        uint32_t *bas_ij_idx, RysIntEnvVars envs,
                        int cell_nbas, float *shell_bounds,
                        double omega2, float log_cutoff)
{
    int tid = threadIdx.x;
    int task = first + tid;
    int keep = 0;
    if (task < last) {
        int pair_ij = shl_pair0 + task/nksh;
        int ksh = ksh0 + task%nksh;
        int ish = bas_ij_idx[pair_ij] / envs.nbas;
        int jsh = bas_ij_idx[pair_ij] % envs.nbas;
        keep = ish >= jsh % cell_nbas &&
               sr_int3c2e_task_keep(ish, jsh, ksh, envs,
                                   shell_bounds, omega2, log_cutoff);
    }
    scan[tid] = keep;
    __syncthreads();
    for (int offset = 1; offset < THREADS; offset <<= 1) {
        int val = tid >= offset ? scan[tid-offset] : 0;
        __syncthreads();
        scan[tid] += val;
        __syncthreads();
    }
    if (keep) tasks[scan[tid]-1] = task;
    int ntasks = scan[THREADS-1];
    __syncthreads();
    return ntasks;
}

// Molecular shell-triplet evaluation on explicit translated shells.  ao_loc
// addresses unit-cell densities; PTR_BAS_COORD addresses super-molecule centers.
__global__ static
void ejk_int3c2e_supermol_kernel(double *ejk, double *sigma, double *dm,
        double *dm_auxvec, double omega, RysIntEnvVars envs,
        uint32_t *bas_ij_idx, int *shl_pair_offsets, int *ksh_offsets,
        int *gout_stride_lookup, int *ao_pair_loc, int aux_offset, int naux,
        int cell_nbas, int cell_natm, int nao,
        float *shell_bounds, float log_cutoff)
{
    int thread_id = threadIdx.x;
    int *bas = envs.bas;
    double *env = envs.env;
    int nbas = envs.nbas;
    int shl_pair0 = shl_pair_offsets[blockIdx.x];
    int shl_pair1 = shl_pair_offsets[blockIdx.x+1];
    int ksh0 = ksh_offsets[2*blockIdx.y];
    int ksh1 = ksh_offsets[2*blockIdx.y+1];
    int ish0 = bas_ij_idx[shl_pair0] / nbas;
    int jsh0 = bas_ij_idx[shl_pair0] % nbas;
    int li = bas[ish0*BAS_SLOTS+ANG_OF];
    int lj = bas[jsh0*BAS_SLOTS+ANG_OF];
    int lk = bas[ksh0*BAS_SLOTS+ANG_OF];
    int iprim = bas[ish0*BAS_SLOTS+NPRIM_OF];
    int jprim = bas[jsh0*BAS_SLOTS+NPRIM_OF];
    int kprim = bas[ksh0*BAS_SLOTS+NPRIM_OF];
    int nroots = ((li+lj+1+lk)/2+1)*2;
    int nf = c_nf[li]*c_nf[lj]*c_nf[lk];
    int g_size = (li+2)*(lj+1)*(lk+1);
    int gout_stride = gout_stride_lookup[lk*LMAX1*LMAX1+li*LMAX1+lj];
    int nst_per_block = THREADS / gout_stride;
    int gout_id = thread_id / nst_per_block;
    int st_id = thread_id % nst_per_block;
    extern __shared__ double workspace[];
    int *tasks = (int *)workspace;
    double *shared_memory = workspace + THREADS*sizeof(int)/sizeof(double);
    double *rjri = shared_memory + st_id;
    double *Rpq = shared_memory + nst_per_block*3 + st_id;
    double *gx = shared_memory + nst_per_block*6 + st_id;
    double *rw = shared_memory + nst_per_block*(g_size*3+6) + st_id;
    int idx_i = lex_xyz_offset(li);
    int idx_j = lex_xyz_offset(lj);
    int idx_k = lex_xyz_offset(lk);
    double sigma_xx = 0, sigma_xy = 0, sigma_xz = 0;
    double sigma_yx = 0, sigma_yy = 0, sigma_yz = 0;
    double sigma_zx = 0, sigma_zy = 0, sigma_zz = 0;
    int nksh = ksh1-ksh0;
    int ncandidates = (shl_pair1-shl_pair0)*nksh;
    for (int first = 0; first < ncandidates; first += THREADS) {
        int ntasks = fill_sr_int3c2e_tasks(
            tasks, (int *)shared_memory, first, ncandidates, shl_pair0, nksh,
            ksh0, bas_ij_idx, envs, cell_nbas, shell_bounds, omega*omega, log_cutoff);
        for (int task_id = st_id; task_id < ntasks+st_id; task_id += nst_per_block) {
            int task = tasks[task_id < ntasks ? task_id : 0];
            int pair_ij = shl_pair0 + task/nksh;
            int ksh = ksh0 + task%nksh;
            int ish = bas_ij_idx[pair_ij] / nbas;
            int jsh = bas_ij_idx[pair_ij] % nbas;
            int jsh_cell0 = jsh % cell_nbas;
            double fac = ish == jsh_cell0 ? PI_FAC*.5 : PI_FAC;
            if (ish < jsh_cell0) fac = 0;
            int k0 = envs.ao_loc[ksh];
            int nfi = c_nf[li], nfj = c_nf[lj];
            float div_nfi = c_div_nf[li], div_nfj = c_div_nf[lj];
            double dm_tensor[GOUT_WIDTH];
            for (int n = 0; n < GOUT_WIDTH; ++n) {
                uint32_t ijk = n*gout_stride+gout_id;
                if (ijk >= nf) break;
                uint32_t jk = ijk*div_nfi;
                uint32_t i = ijk-jk*nfi;
                uint32_t k = jk*div_nfj;
                uint32_t j = jk-k*nfj;
                if (dm == NULL) {
                    size_t pair_offset = ao_pair_loc[pair_ij];
                    dm_tensor[n] = dm_auxvec[(pair_offset+j*nfi+i)*naux+k0+k-aux_offset]*fac;
                } else {
                    int i0 = envs.ao_loc[ish];
                    int j0 = envs.ao_loc[jsh];
                    dm_tensor[n] = dm[(j0+j)*nao+i0+i]*dm_auxvec[k0+k]*fac;
                }
            }
            int expk = bas[ksh*BAS_SLOTS+PTR_EXP];
            int ck = bas[ksh*BAS_SLOTS+PTR_COEFF];
            int ri = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
            int rj = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
            int rk = bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
            double xj = env[rj], yj = env[rj+1], zj = env[rj+2];
            double xk = env[rk], yk = env[rk+1], zk = env[rk+2];
            double grad_ix = 0, grad_iy = 0, grad_iz = 0;
            double grad_jx = 0, grad_jy = 0, grad_jz = 0;
            double v_ix = 0, v_iy = 0, v_iz = 0;
            double v_jx = 0, v_jy = 0, v_jz = 0;
            for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                __syncthreads();
                int expi = bas[ish*BAS_SLOTS+PTR_EXP];
                int expj = bas[jsh*BAS_SLOTS+PTR_EXP];
                int ip = ijp / jprim;
                int jp = ijp - jprim * ip;
                double ai = env[expi+ip];
                double aj = env[expj+jp];
                double aij = ai + aj;
                double aj_aij = aj / aij;
                if (gout_id == 0) {
                    int ci = bas[ish*BAS_SLOTS+PTR_COEFF];
                    int cj = bas[jsh*BAS_SLOTS+PTR_COEFF];
                    double xi = env[ri+0];
                    double yi = env[ri+1];
                    double zi = env[ri+2];
                    double xjLxi = xj - xi;
                    double yjLyi = yj - yi;
                    double zjLzi = zj - zi;
                    double fac_ij = 0;
                    if (task_id < ntasks) {
                        double rr_ij = xjLxi * xjLxi + yjLyi * yjLyi + zjLzi * zjLzi;
                        double theta_ij = ai * aj_aij;
                        double Kab = theta_ij * rr_ij;
                        double cicj = env[ci+ip] * env[cj+jp];
                        fac_ij = exp(-Kab) * cicj;
                    }
                    double xij = xjLxi * aj_aij + xi;
                    double yij = yjLyi * aj_aij + yi;
                    double zij = zjLzi * aj_aij + zi;
                    double xpq = xij - xk;
                    double ypq = yij - yk;
                    double zpq = zij - zk;
                    rjri[0*nst_per_block] = xjLxi;
                    rjri[1*nst_per_block] = yjLyi;
                    rjri[2*nst_per_block] = zjLzi;
                    Rpq[0*nst_per_block] = xpq;
                    Rpq[1*nst_per_block] = ypq;
                    Rpq[2*nst_per_block] = zpq;
                    int gx_len = g_size * nst_per_block;
                    gx[gx_len] = fac_ij;
                }
                for (int kp = 0; kp < kprim; ++kp) {
                    double ak = env[expk+kp];
                    double theta = aij * ak / (aij + ak);
                    __syncthreads();
                    if (gout_id == 0) {
                        gx[0] = env[ck+kp] / (aij*ak*sqrt(aij+ak));
                    }
                    double xpq = Rpq[0*nst_per_block];
                    double ypq = Rpq[1*nst_per_block];
                    double zpq = Rpq[2*nst_per_block];
                    double rr = xpq*xpq + ypq*ypq + zpq*zpq;
                    rys_roots_rs(nroots, theta, rr, omega, rw, nst_per_block, gout_id, gout_stride);
                    for (int irys = 0; irys < nroots; ++irys) {
                        int lij = li + lj + 1;
                        int stride_j = li + 2;
                        int stride_k = stride_j * (lj + 1);
                        BUILD_3C_GXYZ(lj, lk, nst_per_block, task_id < ntasks);
                        if (task_id < ntasks) {
                            int nst = nst_per_block;
                            int nfi = c_nf[li];
                            int nfj = c_nf[lj];
                            float div_nfi = c_div_nf[li];
                            float div_nfj = c_div_nf[lj];
                            int i_1 =          nst;
                            int j_1 = stride_j*nst;
                            double ai2 = ai * 2;
                            double aj2 = aj * 2;
#pragma unroll
                            for (int n = 0; n < GOUT_WIDTH; ++n) {
                                uint32_t ijk = n*gout_stride+gout_id;
                                if (ijk >= nf) break;
                                uint32_t jk = ijk * div_nfi;
                                uint32_t i = ijk - jk * nfi;
                                uint32_t k = jk * div_nfj;
                                uint32_t j = jk - k * nfj;
                                int ix = _c_cartesian_lexical_xyz[idx_i + i*3+0];
                                int iy = _c_cartesian_lexical_xyz[idx_i + i*3+1];
                                int iz = _c_cartesian_lexical_xyz[idx_i + i*3+2];
                                int jx = _c_cartesian_lexical_xyz[idx_j + j*3+0];
                                int jy = _c_cartesian_lexical_xyz[idx_j + j*3+1];
                                int jz = _c_cartesian_lexical_xyz[idx_j + j*3+2];
                                int kx = _c_cartesian_lexical_xyz[idx_k + k*3+0];
                                int ky = _c_cartesian_lexical_xyz[idx_k + k*3+1];
                                int kz = _c_cartesian_lexical_xyz[idx_k + k*3+2];
                                int addrx = (ix + jx*stride_j + kx*stride_k) * nst;
                                int addry = (iy + jy*stride_j + ky*stride_k + g_size) * nst;
                                int addrz = (iz + jz*stride_j + kz*stride_k + g_size*2) * nst;
                                double Ix = gx[addrx];
                                double Iy = gx[addry];
                                double Iz = gx[addrz];
                                double dm_val = dm_tensor[n];
                                double prod_xy = Ix * Iy * dm_val;
                                double prod_xz = Ix * Iz * dm_val;
                                double prod_yz = Iy * Iz * dm_val;
                                double gix = gx[addrx+i_1];
                                double giy = gx[addry+i_1];
                                double giz = gx[addrz+i_1];
                                double fix = ai2 * gix; if (ix > 0) { fix -= ix * gx[addrx-i_1]; } v_ix += fix * prod_yz;
                                double fiy = ai2 * giy; if (iy > 0) { fiy -= iy * gx[addry-i_1]; } v_iy += fiy * prod_xz;
                                double fiz = ai2 * giz; if (iz > 0) { fiz -= iz * gx[addrz-i_1]; } v_iz += fiz * prod_xy;
                                double fjx = aj2 * (gix - rjri[0*nst] * Ix); if (jx > 0) { fjx -= jx * gx[addrx-j_1]; } v_jx += fjx * prod_yz;
                                double fjy = aj2 * (giy - rjri[1*nst] * Iy); if (jy > 0) { fjy -= jy * gx[addry-j_1]; } v_jy += fjy * prod_xz;
                                double fjz = aj2 * (giz - rjri[2*nst] * Iz); if (jz > 0) { fjz -= jz * gx[addrz-j_1]; } v_jz += fjz * prod_xy;
                            }
                        }
                    }
                }
            }
            if (task_id < ntasks) {
                double xixk = env[ri+0] - xk;
                double yiyk = env[ri+1] - yk;
                double zizk = env[ri+2] - zk;
                double xjxk = xj - xk;
                double yjyk = yj - yk;
                double zjzk = zj - zk;
                sigma_xx += v_ix * xixk + v_jx * xjxk;
                sigma_xy += v_ix * yiyk + v_jx * yjyk;
                sigma_xz += v_ix * zizk + v_jx * zjzk;
                sigma_yx += v_iy * xixk + v_jy * xjxk;
                sigma_yy += v_iy * yiyk + v_jy * yjyk;
                sigma_yz += v_iy * zizk + v_jy * zjzk;
                sigma_zx += v_iz * xixk + v_jz * xjxk;
                sigma_zy += v_iz * yiyk + v_jz * yjyk;
                sigma_zz += v_iz * zizk + v_jz * zjzk;
                grad_ix += v_ix;
                grad_iy += v_iy;
                grad_iz += v_iz;
                grad_jx += v_jx;
                grad_jy += v_jy;
                grad_jz += v_jz;
            }
            __syncthreads();
            int ia = bas[ish*BAS_SLOTS+ATOM_OF] % cell_natm;
            int ja = bas[jsh*BAS_SLOTS+ATOM_OF] % cell_natm;
            int ka = bas[ksh*BAS_SLOTS+ATOM_OF] % cell_natm;
            double *reduce = shared_memory + thread_id;
            double grad_kx = -grad_ix - grad_jx;
            double grad_ky = -grad_iy - grad_jy;
            double grad_kz = -grad_iz - grad_jz;
            reduce[0*THREADS] = grad_kx;
            reduce[1*THREADS] = grad_ky;
            reduce[2*THREADS] = grad_kz;
            reduce[3*THREADS] = grad_ix;
            reduce[4*THREADS] = grad_iy;
            reduce[5*THREADS] = grad_iz;
            reduce[6*THREADS] = grad_jx;
            reduce[7*THREADS] = grad_jy;
            reduce[8*THREADS] = grad_jz;
            for (int i = gout_stride/2; i > 0; i >>= 1) {
                __syncthreads();
                if (gout_id < i) {
                    for (int n = 0; n < 9; ++n) {
                        reduce[n*THREADS] += reduce[n*THREADS+i*nst_per_block];
                    }
                }
            }
            if (gout_id == 0) {
                atomicAdd(ejk+ka*3+0, reduce[0*THREADS]);
                atomicAdd(ejk+ka*3+1, reduce[1*THREADS]);
                atomicAdd(ejk+ka*3+2, reduce[2*THREADS]);
                atomicAdd(ejk+ia*3+0, reduce[3*THREADS]);
                atomicAdd(ejk+ia*3+1, reduce[4*THREADS]);
                atomicAdd(ejk+ia*3+2, reduce[5*THREADS]);
                atomicAdd(ejk+ja*3+0, reduce[6*THREADS]);
                atomicAdd(ejk+ja*3+1, reduce[7*THREADS]);
                atomicAdd(ejk+ja*3+2, reduce[8*THREADS]);
            }
            __syncthreads();
        }
        __syncthreads();
    } // next candidate tile
    atomicAdd(sigma+0, sigma_xx);
    atomicAdd(sigma+1, sigma_xy);
    atomicAdd(sigma+2, sigma_xz);
    atomicAdd(sigma+3, sigma_yx);
    atomicAdd(sigma+4, sigma_yy);
    atomicAdd(sigma+5, sigma_yz);
    atomicAdd(sigma+6, sigma_zx);
    atomicAdd(sigma+7, sigma_zy);
    atomicAdd(sigma+8, sigma_zz);
}

extern "C"
int PBCsr_ejk_int3c2e_supermol(double *ejk, double *sigma, double *dm,
        double *dm_auxvec, double omega, RysIntEnvVars *envs,
        int shm_size, int nbatches_shl_pair, int nbatches_ksh,
        uint32_t *bas_ij_idx, int *shl_pair_offsets, int *ksh_offsets,
        int *gout_stride_lookup, int *ao_pair_loc, int aux_offset, int naux,
        int cell_nbas, int cell_natm, int nao,
        float *shell_bounds, float log_cutoff)
{
    cudaFuncSetAttribute(ejk_int3c2e_supermol_kernel,
                        cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    dim3 blocks(nbatches_shl_pair, nbatches_ksh);
    ejk_int3c2e_supermol_kernel<<<blocks, THREADS, shm_size>>>(
        ejk, sigma, dm, dm_auxvec, omega, *envs, bas_ij_idx,
        shl_pair_offsets, ksh_offsets, gout_stride_lookup, ao_pair_loc,
        aux_offset, naux, cell_nbas, cell_natm, nao, shell_bounds, log_cutoff);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in ejk_int3c2e_supermol: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
