
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define KERNEL_ARGS \
    double *ejk, double *sigma, double *dm, double *dm_auxvec, \
    double omega, PBCIntEnvVars& envs, \
    uint32_t *img_pool, uint32_t *sub_task_idx, int num_sub_tasks, \
    int img_tile_size, ShellTripletTaskInfo *ijk_tasks_info, \
    int iprim, int jprim, int kprim, \
    uint32_t *bas_ij_idx, int *ao_pair_loc, int aux_offset, \
    int nauxbas, int naux, int nao, int thread_id, double *shared_memory \

#define LAUNCH_KERNEL(KERNEL) \
    KERNEL(ejk, sigma, dm, dm_auxvec, omega, envs, \
    img_pool, sub_task_idx, num_sub_tasks, img_tile_size, ijk_tasks_info, \
    iprim, jprim, kprim, bas_ij_idx, ao_pair_loc, \
    aux_offset, nauxbas, naux, nao, thread_id, shared_memory)


__device__ inline
void int3c2e_ip1_000(KERNEL_ARGS)
{
    int st_id = thread_id;
    constexpr int nst_per_block = THREADS;
    int ncells = envs.bvk_ncells;
    int *bas = envs.bas;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    int nimgs = envs.nimgs;
    double *rw = shared_memory + st_id;
    for (int task_id = st_id; task_id < num_sub_tasks; task_id += nst_per_block) {
        int ijk_id = sub_task_idx[task_id];
        ShellTripletTaskInfo *ijk_task = ijk_tasks_info + ijk_id;
        int img_start = ijk_task->img_count;
        int ksh = ijk_task->ksh;
        int pair_ij = ijk_task->pair_ij;
        uint32_t bas_ij = bas_ij_idx[pair_ij];
        int bvk_nbas = envs.nbas * ncells;
        int ish = bas_ij / bvk_nbas;
        int jsh = bas_ij - bvk_nbas * ish;
        int ish_cell0 = ish;
        int jsh_cell0 = jsh % envs.nbas;
        double fac = PI_FAC;
        if (ish_cell0 == jsh_cell0) {
            fac *= .5;
        } else if (ish_cell0 < jsh_cell0) {
            fac = 0;
        }
        int k_cell_id = (ksh - bvk_nbas) / nauxbas;
        int ksh_cell0 = ksh - k_cell_id * nauxbas;
        int k0 = envs.ao_loc[ksh_cell0] - envs.ao_loc[bvk_nbas];
        double dm_tensor[1];
        if (task_id < num_sub_tasks) {
            if (dm == NULL) {
                size_t pair_offset = ao_pair_loc[pair_ij];
                int bvk_naux = naux * ncells;
                double *dm_local = dm_auxvec + (pair_offset * ncells + k_cell_id) * naux + k0 - aux_offset;
#pragma unroll
                for (int k = 0; k < 1; ++k) {
#pragma unroll
                for (int ij = 0; ij < 1; ++ij) {
                    dm_tensor[k*1+ij] = dm_local[ij*bvk_naux + k] * fac;
                } }
            } else {
                int i0 = envs.ao_loc[ish];
                int j0 = envs.ao_loc[jsh];
                double *dm_local = dm + j0 * nao + i0;
#pragma unroll
                for (int k = 0; k < 1; ++k) {
#pragma unroll
                for (int j = 0; j < 1; ++j) {
#pragma unroll
                for (int i = 0; i < 1; ++i) {
                    dm_tensor[(k*1+j)*1+i] = dm_local[j*nao+i] * dm_auxvec[k0+k] * fac;
                } } }
            }
        }
        int expi = bas[ish*BAS_SLOTS+PTR_EXP];
        int expj = bas[jsh*BAS_SLOTS+PTR_EXP];
        int expk = bas[ksh*BAS_SLOTS+PTR_EXP];
        int ci = bas[ish*BAS_SLOTS+PTR_COEFF];
        int cj = bas[jsh*BAS_SLOTS+PTR_COEFF];
        int ck = bas[ksh*BAS_SLOTS+PTR_COEFF];
        int ri = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        int rj = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        int rk = bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double grad_ix = 0;
        double grad_iy = 0;
        double grad_iz = 0;
        double grad_jx = 0;
        double grad_jy = 0;
        double grad_jz = 0;
        for (int img = 0; img < img_tile_size; img++) {
            int img_jk = 0;
            if (task_id < num_sub_tasks) {
                img_jk = img_pool[ijk_id+POOL_SIZE*(img_start+img)];
            }
            int jL = img_jk / nimgs;
            int kL = img_jk - nimgs * jL;
            double xi = env[ri+0];
            double yi = env[ri+1];
            double zi = env[ri+2];
            double xj = env[rj+0] + img_coords[jL*3+0];
            double yj = env[rj+1] + img_coords[jL*3+1];
            double zj = env[rj+2] + img_coords[jL*3+2];
            double xk = env[rk+0] + img_coords[kL*3+0];
            double yk = env[rk+1] + img_coords[kL*3+1];
            double zk = env[rk+2] + img_coords[kL*3+2];
            double xjxi = xj - xi;
            double yjyi = yj - yi;
            double zjzi = zj - zi;
            double rr_ij = xjxi * xjxi + yjyi * yjyi + zjzi * zjzi;
            double v_ix = 0;
            double v_iy = 0;
            double v_iz = 0;
            double v_jx = 0;
            double v_jy = 0;
            double v_jz = 0;
            double prod_xy;
            double prod_xz;
            double prod_yz;
            double Ix, Iy, Iz;
            double fxi, fyi, fzi;
            double fxj, fyj, fzj;
            int ijkprim = iprim * jprim * kprim;
            for (int ijkp = 0; ijkp < ijkprim; ++ijkp) {
                int ijp = ijkp / kprim;
                int kp = ijkp - kprim * ijp;
                int ip = ijp / jprim;
                int jp = ijp - jprim * ip;
                double ai = env[expi+ip];
                double aj = env[expj+jp];
                double ak = env[expk+kp];
                double ai2 = ai * 2;
                double aj2 = aj * 2;
                double aij = ai + aj;
                double aj_aij = aj / aij;
                double theta_ij = ai * aj_aij;
                double Kab = theta_ij * rr_ij;
                double cijk = env[ci+ip] * env[cj+jp] * env[ck+kp];
                double fac1 = cijk * exp(-Kab) / (aij*ak*sqrt(aij+ak));
                double xij = xjxi * aj_aij + xi;
                double yij = yjyi * aj_aij + yi;
                double zij = zjzi * aj_aij + zi;
                double xpq = xij - xk;
                double ypq = yij - yk;
                double zpq = zij - zk;
                double rr = xpq*xpq + ypq*ypq + zpq*zpq;
                double theta = aij * ak / (aij + ak);
                double theta_rr = theta * rr;
                rys_roots(1, theta_rr, rw, nst_per_block, 0, 1);
                double theta_fac = omega * omega / (omega * omega + theta);
                double *rw1 = rw + 2*nst_per_block;
                rys_roots(1, theta_fac*theta_rr, rw1, nst_per_block, 0, 1);
                double sqrt_theta_fac = -sqrt(theta_fac);
                for (int irys = 0; irys < 1; irys++) {
                    rw1[ irys*2   *nst_per_block] *= theta_fac;
                    rw1[(irys*2+1)*nst_per_block] *= sqrt_theta_fac;
                }
                for (int irys = 0; irys < 2; ++irys) {
                    double wt = rw[(2*irys+1)*nst_per_block];
                    double rt = rw[ 2*irys   *nst_per_block];
                    double rt_aa = rt / (aij + ak);
                    Ix = 1;
                    Iy = fac1;
                    Iz = wt;
                    prod_xy = Ix * Iy * dm_tensor[0];
                    prod_xz = Ix * Iz * dm_tensor[0];
                    prod_yz = Iy * Iz * dm_tensor[0];
                    double rt_aij = rt_aa * ak;
                    double c0x = xjxi * aj_aij - xpq*rt_aij;
                    double trr_10x = c0x * 1;
                    fxi = ai2 * trr_10x;
                    v_ix += fxi * prod_yz;
                    double c0y = yjyi * aj_aij - ypq*rt_aij;
                    double trr_10y = c0y * fac1;
                    fyi = ai2 * trr_10y;
                    v_iy += fyi * prod_xz;
                    double c0z = zjzi * aj_aij - zpq*rt_aij;
                    double trr_10z = c0z * wt;
                    fzi = ai2 * trr_10z;
                    v_iz += fzi * prod_xy;
                    double hrr_010x = trr_10x - xjxi * 1;
                    fxj = aj2 * hrr_010x;
                    v_jx += fxj * prod_yz;
                    double hrr_010y = trr_10y - yjyi * fac1;
                    fyj = aj2 * hrr_010y;
                    v_jy += fyj * prod_xz;
                    double hrr_010z = trr_10z - zjzi * wt;
                    fzj = aj2 * hrr_010z;
                    v_jz += fzj * prod_xy;
                }
            }
            double xixk = xi - xk;
            double yiyk = yi - yk;
            double zizk = zi - zk;
            double xjxk = xj - xk;
            double yjyk = yj - yk;
            double zjzk = zj - zk;
            sigma[0] += v_ix * xixk + v_jx * xjxk;
            sigma[1] += v_ix * yiyk + v_jx * yjyk;
            sigma[2] += v_ix * zizk + v_jx * zjzk;
            sigma[3] += v_iy * xixk + v_jy * xjxk;
            sigma[4] += v_iy * yiyk + v_jy * yjyk;
            sigma[5] += v_iy * zizk + v_jy * zjzk;
            sigma[6] += v_iz * xixk + v_jz * xjxk;
            sigma[7] += v_iz * yiyk + v_jz * yjyk;
            sigma[8] += v_iz * zizk + v_jz * zjzk;
            grad_ix += v_ix;
            grad_iy += v_iy;
            grad_iz += v_iz;
            grad_jx += v_jx;
            grad_jy += v_jy;
            grad_jz += v_jz;
        }
        int ia = bas[ish*BAS_SLOTS+ATOM_OF] % envs.cell0_natm;
        int ja = bas[jsh*BAS_SLOTS+ATOM_OF] % envs.cell0_natm;
        int ka = bas[ksh*BAS_SLOTS+ATOM_OF] % envs.cell0_natm;
        double grad_kx = -grad_ix - grad_jx;
        double grad_ky = -grad_iy - grad_jy;
        double grad_kz = -grad_iz - grad_jz;
        atomicAdd(ejk+ka*3+0, grad_kx);
        atomicAdd(ejk+ka*3+1, grad_ky);
        atomicAdd(ejk+ka*3+2, grad_kz);
        atomicAdd(ejk+ia*3+0, grad_ix);
        atomicAdd(ejk+ia*3+1, grad_iy);
        atomicAdd(ejk+ia*3+2, grad_iz);
        atomicAdd(ejk+ja*3+0, grad_jx);
        atomicAdd(ejk+ja*3+1, grad_jy);
        atomicAdd(ejk+ja*3+2, grad_jz);
    }
}

__device__ inline
void int3c2e_ip1_100(KERNEL_ARGS)
{
    int st_id = thread_id;
    constexpr int nst_per_block = THREADS;
    int ncells = envs.bvk_ncells;
    int *bas = envs.bas;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    int nimgs = envs.nimgs;
    double *rw = shared_memory + st_id;
    for (int task_id = st_id; task_id < num_sub_tasks; task_id += nst_per_block) {
        int ijk_id = sub_task_idx[task_id];
        ShellTripletTaskInfo *ijk_task = ijk_tasks_info + ijk_id;
        int img_start = ijk_task->img_count;
        int ksh = ijk_task->ksh;
        int pair_ij = ijk_task->pair_ij;
        uint32_t bas_ij = bas_ij_idx[pair_ij];
        int bvk_nbas = envs.nbas * ncells;
        int ish = bas_ij / bvk_nbas;
        int jsh = bas_ij - bvk_nbas * ish;
        int ish_cell0 = ish;
        int jsh_cell0 = jsh % envs.nbas;
        double fac = PI_FAC;
        if (ish_cell0 == jsh_cell0) {
            fac *= .5;
        } else if (ish_cell0 < jsh_cell0) {
            fac = 0;
        }
        int k_cell_id = (ksh - bvk_nbas) / nauxbas;
        int ksh_cell0 = ksh - k_cell_id * nauxbas;
        int k0 = envs.ao_loc[ksh_cell0] - envs.ao_loc[bvk_nbas];
        double dm_tensor[3];
        if (task_id < num_sub_tasks) {
            if (dm == NULL) {
                size_t pair_offset = ao_pair_loc[pair_ij];
                int bvk_naux = naux * ncells;
                double *dm_local = dm_auxvec + (pair_offset * ncells + k_cell_id) * naux + k0 - aux_offset;
#pragma unroll
                for (int k = 0; k < 1; ++k) {
#pragma unroll
                for (int ij = 0; ij < 3; ++ij) {
                    dm_tensor[k*3+ij] = dm_local[ij*bvk_naux + k] * fac;
                } }
            } else {
                int i0 = envs.ao_loc[ish];
                int j0 = envs.ao_loc[jsh];
                double *dm_local = dm + j0 * nao + i0;
#pragma unroll
                for (int k = 0; k < 1; ++k) {
#pragma unroll
                for (int j = 0; j < 1; ++j) {
#pragma unroll
                for (int i = 0; i < 3; ++i) {
                    dm_tensor[(k*1+j)*3+i] = dm_local[j*nao+i] * dm_auxvec[k0+k] * fac;
                } } }
            }
        }
        int expi = bas[ish*BAS_SLOTS+PTR_EXP];
        int expj = bas[jsh*BAS_SLOTS+PTR_EXP];
        int expk = bas[ksh*BAS_SLOTS+PTR_EXP];
        int ci = bas[ish*BAS_SLOTS+PTR_COEFF];
        int cj = bas[jsh*BAS_SLOTS+PTR_COEFF];
        int ck = bas[ksh*BAS_SLOTS+PTR_COEFF];
        int ri = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        int rj = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        int rk = bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double grad_ix = 0;
        double grad_iy = 0;
        double grad_iz = 0;
        double grad_jx = 0;
        double grad_jy = 0;
        double grad_jz = 0;
        for (int img = 0; img < img_tile_size; img++) {
            int img_jk = 0;
            if (task_id < num_sub_tasks) {
                img_jk = img_pool[ijk_id+POOL_SIZE*(img_start+img)];
            }
            int jL = img_jk / nimgs;
            int kL = img_jk - nimgs * jL;
            double xi = env[ri+0];
            double yi = env[ri+1];
            double zi = env[ri+2];
            double xj = env[rj+0] + img_coords[jL*3+0];
            double yj = env[rj+1] + img_coords[jL*3+1];
            double zj = env[rj+2] + img_coords[jL*3+2];
            double xk = env[rk+0] + img_coords[kL*3+0];
            double yk = env[rk+1] + img_coords[kL*3+1];
            double zk = env[rk+2] + img_coords[kL*3+2];
            double xjxi = xj - xi;
            double yjyi = yj - yi;
            double zjzi = zj - zi;
            double rr_ij = xjxi * xjxi + yjyi * yjyi + zjzi * zjzi;
            double v_ix = 0;
            double v_iy = 0;
            double v_iz = 0;
            double v_jx = 0;
            double v_jy = 0;
            double v_jz = 0;
            double prod_xy;
            double prod_xz;
            double prod_yz;
            double Ix, Iy, Iz;
            double fxi, fyi, fzi;
            double fxj, fyj, fzj;
            int ijkprim = iprim * jprim * kprim;
            for (int ijkp = 0; ijkp < ijkprim; ++ijkp) {
                int ijp = ijkp / kprim;
                int kp = ijkp - kprim * ijp;
                int ip = ijp / jprim;
                int jp = ijp - jprim * ip;
                double ai = env[expi+ip];
                double aj = env[expj+jp];
                double ak = env[expk+kp];
                double ai2 = ai * 2;
                double aj2 = aj * 2;
                double aij = ai + aj;
                double aj_aij = aj / aij;
                double theta_ij = ai * aj_aij;
                double Kab = theta_ij * rr_ij;
                double cijk = env[ci+ip] * env[cj+jp] * env[ck+kp];
                double fac1 = cijk * exp(-Kab) / (aij*ak*sqrt(aij+ak));
                double xij = xjxi * aj_aij + xi;
                double yij = yjyi * aj_aij + yi;
                double zij = zjzi * aj_aij + zi;
                double xpq = xij - xk;
                double ypq = yij - yk;
                double zpq = zij - zk;
                double rr = xpq*xpq + ypq*ypq + zpq*zpq;
                double theta = aij * ak / (aij + ak);
                double theta_rr = theta * rr;
                rys_roots(2, theta_rr, rw, nst_per_block, 0, 1);
                double theta_fac = omega * omega / (omega * omega + theta);
                double *rw1 = rw + 4*nst_per_block;
                rys_roots(2, theta_fac*theta_rr, rw1, nst_per_block, 0, 1);
                double sqrt_theta_fac = -sqrt(theta_fac);
                for (int irys = 0; irys < 2; irys++) {
                    rw1[ irys*2   *nst_per_block] *= theta_fac;
                    rw1[(irys*2+1)*nst_per_block] *= sqrt_theta_fac;
                }
                for (int irys = 0; irys < 4; ++irys) {
                    double wt = rw[(2*irys+1)*nst_per_block];
                    double rt = rw[ 2*irys   *nst_per_block];
                    double rt_aa = rt / (aij + ak);
                    double rt_aij = rt_aa * ak;
                    double b10 = .5/aij * (1 - rt_aij);
                    double c0x = xjxi * aj_aij - xpq*rt_aij;
                    double trr_10x = c0x * 1;
                    Ix = trr_10x;
                    Iy = fac1;
                    Iz = wt;
                    prod_xy = Ix * Iy * dm_tensor[0];
                    prod_xz = Ix * Iz * dm_tensor[0];
                    prod_yz = Iy * Iz * dm_tensor[0];
                    double trr_20x = c0x * trr_10x + 1*b10 * 1;
                    fxi = ai2 * trr_20x;
                    fxi -= 1 * 1;
                    v_ix += fxi * prod_yz;
                    double c0y = yjyi * aj_aij - ypq*rt_aij;
                    double trr_10y = c0y * fac1;
                    fyi = ai2 * trr_10y;
                    v_iy += fyi * prod_xz;
                    double c0z = zjzi * aj_aij - zpq*rt_aij;
                    double trr_10z = c0z * wt;
                    fzi = ai2 * trr_10z;
                    v_iz += fzi * prod_xy;
                    double hrr_110x = trr_20x - xjxi * trr_10x;
                    fxj = aj2 * hrr_110x;
                    v_jx += fxj * prod_yz;
                    double hrr_010y = trr_10y - yjyi * fac1;
                    fyj = aj2 * hrr_010y;
                    v_jy += fyj * prod_xz;
                    double hrr_010z = trr_10z - zjzi * wt;
                    fzj = aj2 * hrr_010z;
                    v_jz += fzj * prod_xy;
                    Ix = 1;
                    Iy = trr_10y;
                    Iz = wt;
                    prod_xy = Ix * Iy * dm_tensor[1];
                    prod_xz = Ix * Iz * dm_tensor[1];
                    prod_yz = Iy * Iz * dm_tensor[1];
                    fxi = ai2 * trr_10x;
                    v_ix += fxi * prod_yz;
                    double trr_20y = c0y * trr_10y + 1*b10 * fac1;
                    fyi = ai2 * trr_20y;
                    fyi -= 1 * fac1;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * trr_10z;
                    v_iz += fzi * prod_xy;
                    double hrr_010x = trr_10x - xjxi * 1;
                    fxj = aj2 * hrr_010x;
                    v_jx += fxj * prod_yz;
                    double hrr_110y = trr_20y - yjyi * trr_10y;
                    fyj = aj2 * hrr_110y;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_010z;
                    v_jz += fzj * prod_xy;
                    Ix = 1;
                    Iy = fac1;
                    Iz = trr_10z;
                    prod_xy = Ix * Iy * dm_tensor[2];
                    prod_xz = Ix * Iz * dm_tensor[2];
                    prod_yz = Iy * Iz * dm_tensor[2];
                    fxi = ai2 * trr_10x;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * trr_10y;
                    v_iy += fyi * prod_xz;
                    double trr_20z = c0z * trr_10z + 1*b10 * wt;
                    fzi = ai2 * trr_20z;
                    fzi -= 1 * wt;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_010x;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_010y;
                    v_jy += fyj * prod_xz;
                    double hrr_110z = trr_20z - zjzi * trr_10z;
                    fzj = aj2 * hrr_110z;
                    v_jz += fzj * prod_xy;
                }
            }
            double xixk = xi - xk;
            double yiyk = yi - yk;
            double zizk = zi - zk;
            double xjxk = xj - xk;
            double yjyk = yj - yk;
            double zjzk = zj - zk;
            sigma[0] += v_ix * xixk + v_jx * xjxk;
            sigma[1] += v_ix * yiyk + v_jx * yjyk;
            sigma[2] += v_ix * zizk + v_jx * zjzk;
            sigma[3] += v_iy * xixk + v_jy * xjxk;
            sigma[4] += v_iy * yiyk + v_jy * yjyk;
            sigma[5] += v_iy * zizk + v_jy * zjzk;
            sigma[6] += v_iz * xixk + v_jz * xjxk;
            sigma[7] += v_iz * yiyk + v_jz * yjyk;
            sigma[8] += v_iz * zizk + v_jz * zjzk;
            grad_ix += v_ix;
            grad_iy += v_iy;
            grad_iz += v_iz;
            grad_jx += v_jx;
            grad_jy += v_jy;
            grad_jz += v_jz;
        }
        int ia = bas[ish*BAS_SLOTS+ATOM_OF] % envs.cell0_natm;
        int ja = bas[jsh*BAS_SLOTS+ATOM_OF] % envs.cell0_natm;
        int ka = bas[ksh*BAS_SLOTS+ATOM_OF] % envs.cell0_natm;
        double grad_kx = -grad_ix - grad_jx;
        double grad_ky = -grad_iy - grad_jy;
        double grad_kz = -grad_iz - grad_jz;
        atomicAdd(ejk+ka*3+0, grad_kx);
        atomicAdd(ejk+ka*3+1, grad_ky);
        atomicAdd(ejk+ka*3+2, grad_kz);
        atomicAdd(ejk+ia*3+0, grad_ix);
        atomicAdd(ejk+ia*3+1, grad_iy);
        atomicAdd(ejk+ia*3+2, grad_iz);
        atomicAdd(ejk+ja*3+0, grad_jx);
        atomicAdd(ejk+ja*3+1, grad_jy);
        atomicAdd(ejk+ja*3+2, grad_jz);
    }
}

__device__ inline
void int3c2e_ip1_001(KERNEL_ARGS)
{
    int st_id = thread_id;
    constexpr int nst_per_block = THREADS;
    int ncells = envs.bvk_ncells;
    int *bas = envs.bas;
    double *env = envs.env;
    double *img_coords = envs.img_coords;
    int nimgs = envs.nimgs;
    double *rw = shared_memory + st_id;
    for (int task_id = st_id; task_id < num_sub_tasks; task_id += nst_per_block) {
        int ijk_id = sub_task_idx[task_id];
        ShellTripletTaskInfo *ijk_task = ijk_tasks_info + ijk_id;
        int img_start = ijk_task->img_count;
        int ksh = ijk_task->ksh;
        int pair_ij = ijk_task->pair_ij;
        uint32_t bas_ij = bas_ij_idx[pair_ij];
        int bvk_nbas = envs.nbas * ncells;
        int ish = bas_ij / bvk_nbas;
        int jsh = bas_ij - bvk_nbas * ish;
        int ish_cell0 = ish;
        int jsh_cell0 = jsh % envs.nbas;
        double fac = PI_FAC;
        if (ish_cell0 == jsh_cell0) {
            fac *= .5;
        } else if (ish_cell0 < jsh_cell0) {
            fac = 0;
        }
        int k_cell_id = (ksh - bvk_nbas) / nauxbas;
        int ksh_cell0 = ksh - k_cell_id * nauxbas;
        int k0 = envs.ao_loc[ksh_cell0] - envs.ao_loc[bvk_nbas];
        double dm_tensor[3];
        if (task_id < num_sub_tasks) {
            if (dm == NULL) {
                size_t pair_offset = ao_pair_loc[pair_ij];
                int bvk_naux = naux * ncells;
                double *dm_local = dm_auxvec + (pair_offset * ncells + k_cell_id) * naux + k0 - aux_offset;
#pragma unroll
                for (int k = 0; k < 3; ++k) {
#pragma unroll
                for (int ij = 0; ij < 1; ++ij) {
                    dm_tensor[k*1+ij] = dm_local[ij*bvk_naux + k] * fac;
                } }
            } else {
                int i0 = envs.ao_loc[ish];
                int j0 = envs.ao_loc[jsh];
                double *dm_local = dm + j0 * nao + i0;
#pragma unroll
                for (int k = 0; k < 3; ++k) {
#pragma unroll
                for (int j = 0; j < 1; ++j) {
#pragma unroll
                for (int i = 0; i < 1; ++i) {
                    dm_tensor[(k*1+j)*1+i] = dm_local[j*nao+i] * dm_auxvec[k0+k] * fac;
                } } }
            }
        }
        int expi = bas[ish*BAS_SLOTS+PTR_EXP];
        int expj = bas[jsh*BAS_SLOTS+PTR_EXP];
        int expk = bas[ksh*BAS_SLOTS+PTR_EXP];
        int ci = bas[ish*BAS_SLOTS+PTR_COEFF];
        int cj = bas[jsh*BAS_SLOTS+PTR_COEFF];
        int ck = bas[ksh*BAS_SLOTS+PTR_COEFF];
        int ri = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        int rj = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        int rk = bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
        double grad_ix = 0;
        double grad_iy = 0;
        double grad_iz = 0;
        double grad_jx = 0;
        double grad_jy = 0;
        double grad_jz = 0;
        for (int img = 0; img < img_tile_size; img++) {
            int img_jk = 0;
            if (task_id < num_sub_tasks) {
                img_jk = img_pool[ijk_id+POOL_SIZE*(img_start+img)];
            }
            int jL = img_jk / nimgs;
            int kL = img_jk - nimgs * jL;
            double xi = env[ri+0];
            double yi = env[ri+1];
            double zi = env[ri+2];
            double xj = env[rj+0] + img_coords[jL*3+0];
            double yj = env[rj+1] + img_coords[jL*3+1];
            double zj = env[rj+2] + img_coords[jL*3+2];
            double xk = env[rk+0] + img_coords[kL*3+0];
            double yk = env[rk+1] + img_coords[kL*3+1];
            double zk = env[rk+2] + img_coords[kL*3+2];
            double xjxi = xj - xi;
            double yjyi = yj - yi;
            double zjzi = zj - zi;
            double rr_ij = xjxi * xjxi + yjyi * yjyi + zjzi * zjzi;
            double v_ix = 0;
            double v_iy = 0;
            double v_iz = 0;
            double v_jx = 0;
            double v_jy = 0;
            double v_jz = 0;
            double prod_xy;
            double prod_xz;
            double prod_yz;
            double Ix, Iy, Iz;
            double fxi, fyi, fzi;
            double fxj, fyj, fzj;
            int ijkprim = iprim * jprim * kprim;
            for (int ijkp = 0; ijkp < ijkprim; ++ijkp) {
                int ijp = ijkp / kprim;
                int kp = ijkp - kprim * ijp;
                int ip = ijp / jprim;
                int jp = ijp - jprim * ip;
                double ai = env[expi+ip];
                double aj = env[expj+jp];
                double ak = env[expk+kp];
                double ai2 = ai * 2;
                double aj2 = aj * 2;
                double aij = ai + aj;
                double aj_aij = aj / aij;
                double theta_ij = ai * aj_aij;
                double Kab = theta_ij * rr_ij;
                double cijk = env[ci+ip] * env[cj+jp] * env[ck+kp];
                double fac1 = cijk * exp(-Kab) / (aij*ak*sqrt(aij+ak));
                double xij = xjxi * aj_aij + xi;
                double yij = yjyi * aj_aij + yi;
                double zij = zjzi * aj_aij + zi;
                double xpq = xij - xk;
                double ypq = yij - yk;
                double zpq = zij - zk;
                double rr = xpq*xpq + ypq*ypq + zpq*zpq;
                double theta = aij * ak / (aij + ak);
                double theta_rr = theta * rr;
                rys_roots(2, theta_rr, rw, nst_per_block, 0, 1);
                double theta_fac = omega * omega / (omega * omega + theta);
                double *rw1 = rw + 4*nst_per_block;
                rys_roots(2, theta_fac*theta_rr, rw1, nst_per_block, 0, 1);
                double sqrt_theta_fac = -sqrt(theta_fac);
                for (int irys = 0; irys < 2; irys++) {
                    rw1[ irys*2   *nst_per_block] *= theta_fac;
                    rw1[(irys*2+1)*nst_per_block] *= sqrt_theta_fac;
                }
                for (int irys = 0; irys < 4; ++irys) {
                    double wt = rw[(2*irys+1)*nst_per_block];
                    double rt = rw[ 2*irys   *nst_per_block];
                    double rt_aa = rt / (aij + ak);
                    double b00 = .5 * rt_aa;
                    double rt_ak = rt_aa * aij;
                    double cpx = xpq*rt_ak;
                    double trr_01x = cpx * 1;
                    Ix = trr_01x;
                    Iy = fac1;
                    Iz = wt;
                    prod_xy = Ix * Iy * dm_tensor[0];
                    prod_xz = Ix * Iz * dm_tensor[0];
                    prod_yz = Iy * Iz * dm_tensor[0];
                    double rt_aij = rt_aa * ak;
                    double c0x = xjxi * aj_aij - xpq*rt_aij;
                    double trr_10x = c0x * 1;
                    double trr_11x = cpx * trr_10x + 1*b00 * 1;
                    fxi = ai2 * trr_11x;
                    v_ix += fxi * prod_yz;
                    double c0y = yjyi * aj_aij - ypq*rt_aij;
                    double trr_10y = c0y * fac1;
                    fyi = ai2 * trr_10y;
                    v_iy += fyi * prod_xz;
                    double c0z = zjzi * aj_aij - zpq*rt_aij;
                    double trr_10z = c0z * wt;
                    fzi = ai2 * trr_10z;
                    v_iz += fzi * prod_xy;
                    double hrr_011x = trr_11x - xjxi * trr_01x;
                    fxj = aj2 * hrr_011x;
                    v_jx += fxj * prod_yz;
                    double hrr_010y = trr_10y - yjyi * fac1;
                    fyj = aj2 * hrr_010y;
                    v_jy += fyj * prod_xz;
                    double hrr_010z = trr_10z - zjzi * wt;
                    fzj = aj2 * hrr_010z;
                    v_jz += fzj * prod_xy;
                    double cpy = ypq*rt_ak;
                    double trr_01y = cpy * fac1;
                    Ix = 1;
                    Iy = trr_01y;
                    Iz = wt;
                    prod_xy = Ix * Iy * dm_tensor[1];
                    prod_xz = Ix * Iz * dm_tensor[1];
                    prod_yz = Iy * Iz * dm_tensor[1];
                    fxi = ai2 * trr_10x;
                    v_ix += fxi * prod_yz;
                    double trr_11y = cpy * trr_10y + 1*b00 * fac1;
                    fyi = ai2 * trr_11y;
                    v_iy += fyi * prod_xz;
                    fzi = ai2 * trr_10z;
                    v_iz += fzi * prod_xy;
                    double hrr_010x = trr_10x - xjxi * 1;
                    fxj = aj2 * hrr_010x;
                    v_jx += fxj * prod_yz;
                    double hrr_011y = trr_11y - yjyi * trr_01y;
                    fyj = aj2 * hrr_011y;
                    v_jy += fyj * prod_xz;
                    fzj = aj2 * hrr_010z;
                    v_jz += fzj * prod_xy;
                    double cpz = zpq*rt_ak;
                    double trr_01z = cpz * wt;
                    Ix = 1;
                    Iy = fac1;
                    Iz = trr_01z;
                    prod_xy = Ix * Iy * dm_tensor[2];
                    prod_xz = Ix * Iz * dm_tensor[2];
                    prod_yz = Iy * Iz * dm_tensor[2];
                    fxi = ai2 * trr_10x;
                    v_ix += fxi * prod_yz;
                    fyi = ai2 * trr_10y;
                    v_iy += fyi * prod_xz;
                    double trr_11z = cpz * trr_10z + 1*b00 * wt;
                    fzi = ai2 * trr_11z;
                    v_iz += fzi * prod_xy;
                    fxj = aj2 * hrr_010x;
                    v_jx += fxj * prod_yz;
                    fyj = aj2 * hrr_010y;
                    v_jy += fyj * prod_xz;
                    double hrr_011z = trr_11z - zjzi * trr_01z;
                    fzj = aj2 * hrr_011z;
                    v_jz += fzj * prod_xy;
                }
            }
            double xixk = xi - xk;
            double yiyk = yi - yk;
            double zizk = zi - zk;
            double xjxk = xj - xk;
            double yjyk = yj - yk;
            double zjzk = zj - zk;
            sigma[0] += v_ix * xixk + v_jx * xjxk;
            sigma[1] += v_ix * yiyk + v_jx * yjyk;
            sigma[2] += v_ix * zizk + v_jx * zjzk;
            sigma[3] += v_iy * xixk + v_jy * xjxk;
            sigma[4] += v_iy * yiyk + v_jy * yjyk;
            sigma[5] += v_iy * zizk + v_jy * zjzk;
            sigma[6] += v_iz * xixk + v_jz * xjxk;
            sigma[7] += v_iz * yiyk + v_jz * yjyk;
            sigma[8] += v_iz * zizk + v_jz * zjzk;
            grad_ix += v_ix;
            grad_iy += v_iy;
            grad_iz += v_iz;
            grad_jx += v_jx;
            grad_jy += v_jy;
            grad_jz += v_jz;
        }
        int ia = bas[ish*BAS_SLOTS+ATOM_OF] % envs.cell0_natm;
        int ja = bas[jsh*BAS_SLOTS+ATOM_OF] % envs.cell0_natm;
        int ka = bas[ksh*BAS_SLOTS+ATOM_OF] % envs.cell0_natm;
        double grad_kx = -grad_ix - grad_jx;
        double grad_ky = -grad_iy - grad_jy;
        double grad_kz = -grad_iz - grad_jz;
        atomicAdd(ejk+ka*3+0, grad_kx);
        atomicAdd(ejk+ka*3+1, grad_ky);
        atomicAdd(ejk+ka*3+2, grad_kz);
        atomicAdd(ejk+ia*3+0, grad_ix);
        atomicAdd(ejk+ia*3+1, grad_iy);
        atomicAdd(ejk+ia*3+2, grad_iz);
        atomicAdd(ejk+ja*3+0, grad_jx);
        atomicAdd(ejk+ja*3+1, grad_jy);
        atomicAdd(ejk+ja*3+2, grad_jz);
    }
}

__device__ inline
int int3c2e_ip1_unrolled(double *ejk, double *sigma, double *dm, double *dm_auxvec,
                double omega, PBCIntEnvVars& envs,
                uint32_t *img_pool, uint32_t *sub_task_idx, int num_sub_tasks,
                int img_tile_size, ShellTripletTaskInfo *ijk_tasks_info,
                int iprim, int jprim, int kprim, int li, int lj, int lk,
                uint32_t *bas_ij_idx, int *ao_pair_loc, int aux_offset,
                int nauxbas, int naux, int nao, int thread_id, double *shared_memory)
{
    int kij_type = lk*25 + li*5 + lj;
    switch (kij_type) {
    case 0: // li=0 lj=0 lk=0
        LAUNCH_KERNEL(int3c2e_ip1_000); break;
    case 5: // li=1 lj=0 lk=0
        LAUNCH_KERNEL(int3c2e_ip1_100); break;
    case 25: // li=0 lj=0 lk=1
        LAUNCH_KERNEL(int3c2e_ip1_001); break;
    default: return 0;
    }
    return 1;
}
