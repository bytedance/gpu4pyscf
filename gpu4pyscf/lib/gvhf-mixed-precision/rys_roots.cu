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

#include "rys_roots.cuh"
#include "mixed_precision_helper.h"

#define SQRTPIE4        .8862269254527580136
#define PIE4            .7853981633974483096

__device__
static void rys_roots(int nroots, double x, double *rw,
                      int block_size, int rt_id, int stride)
{
    double *r = rw;
    double *w = rw + block_size;
    int block_size2 = block_size * 2;
    if (x < 3.e-7){
        int off = nroots * (nroots - 1) / 2;
        for (int i = rt_id; i < nroots; i += stride)  {
            r[i*block_size2] = ROOT_SMALLX_R0[off+i] + ROOT_SMALLX_R1[off+i] * x;
            w[i*block_size2] = ROOT_SMALLX_W0[off+i] + ROOT_SMALLX_W1[off+i] * x;
        }
        return;
    }

    if (x > 35+nroots*5) {
        int off = nroots * (nroots - 1) / 2;
        double t = sqrt(PIE4/x);
        for (int i = rt_id; i < nroots; i += stride)  {
            r[i*block_size2] = ROOT_LARGEX_R_DATA[off+i] / x;
            w[i*block_size2] = ROOT_LARGEX_W_DATA[off+i] * t;
        }
        return;
    }

    if (nroots == 1) {
        double tt = sqrt(x);
        double fmt0 = SQRTPIE4 / tt * erf(tt);
        w[0] = fmt0;
        double e = exp(-x);
        double b = .5 / x;
        double fmt1 = b * (fmt0 - e);
        r[0] = fmt1 / fmt0;
        return;
    }

    const double *datax = ROOT_RW_DATA + DEGREE1*INTERVALS * nroots*(nroots-1);
    int it = (int)(x * .4);
    double u = (x - it * 2.5) * 0.8 - 1.;
    double u2 = u * 2.;
    for (int i = rt_id; i < nroots*2; i += stride) {
        const double *c = datax + i * DEGREE1 * INTERVALS;
        //for i in range(2, degree + 1):
        //    c0, c1 = c[degree-i] - c1, c0 + c1*u2
        double c0 = c[it + DEGREE   *INTERVALS];
        double c1 = c[it +(DEGREE-1)*INTERVALS];
        double c2, c3;
#pragma unroll
        for (int n = DEGREE-2; n > 0; n-=2) {
            c2 = c[it + n   *INTERVALS] - c1;
            c3 = c0 + c1*u2;
            c1 = c2 + c3*u2;
            c0 = c[it +(n-1)*INTERVALS] - c3;
        }
        if (DEGREE % 2 == 0) {
            c2 = c[it] - c1;
            c3 = c0 + c1*u2;
            rw[i*block_size] = c2 + c3*u;
        } else {
            rw[i*block_size] = c0 + c1*u;
        }
    }
}

// rys_roots for range-separation Coulomb
__device__
static void rys_roots_rs(int nroots, double theta, double rr, double omega,
                         double *rw, int block_size, int rt_id, int stride)
{
    double theta_rr = theta * rr;
    if (omega == 0) {
        rys_roots(nroots, theta_rr, rw, block_size, rt_id, stride);
    } else if (omega > 0) {
        double theta_fac = omega * omega / (omega * omega + theta);
        rys_roots(nroots, theta_fac*theta_rr, rw, block_size, rt_id, stride);
        __syncthreads();
        double sqrt_theta_fac = sqrt(theta_fac);
        for (int irys = rt_id; irys < nroots; irys+=stride) {
            rw[ irys*2   *block_size] *= theta_fac;
            rw[(irys*2+1)*block_size] *= sqrt_theta_fac;
        }
    } else {
        int _nroots = nroots / 2;
        double *rw1 = rw + nroots*block_size;
        rys_roots(_nroots, theta_rr, rw1, block_size, rt_id, stride);
        double theta_fac = omega * omega / (omega * omega + theta);
        rys_roots(_nroots, theta_fac*theta_rr, rw, block_size, rt_id, stride);
        __syncthreads();
        double sqrt_theta_fac = -sqrt(theta_fac);
        for (int irys = rt_id; irys < _nroots; irys+=stride) {
            rw[ irys*2   *block_size] *= theta_fac;
            rw[(irys*2+1)*block_size] *= sqrt_theta_fac;
        }
    }
}

__device__
static void rys_roots_single_precision(const int nroots, const float x, float *rw,
                                       const int block_size, const int rt_id, const int stride)
{
    float *r = rw;
    float *w = rw + block_size;
    int block_size2 = block_size * 2;
    if (x < 3.e-7){
        int off = nroots * (nroots - 1) / 2;
        for (int i = rt_id; i < nroots; i += stride)  {
            r[i*block_size2] = SINGLE_ROOT_SMALLX_R0[off+i] + SINGLE_ROOT_SMALLX_R1[off+i] * x;
            w[i*block_size2] = SINGLE_ROOT_SMALLX_W0[off+i] + SINGLE_ROOT_SMALLX_W1[off+i] * x;
        }
        return;
    }

    if (x > 35+nroots*5) {
        int off = nroots * (nroots - 1) / 2;
        const float t = sqrtf(static_cast<float>(PIE4) / x);
        for (int i = rt_id; i < nroots; i += stride)  {
            r[i*block_size2] = SINGLE_ROOT_LARGEX_R_DATA[off+i] / x;
            w[i*block_size2] = SINGLE_ROOT_LARGEX_W_DATA[off+i] * t;
        }
        return;
    }

    if (nroots == 1) {
        const float tt = sqrtf(x);
        const float fmt0 = static_cast<float>(SQRTPIE4) / tt * erff(tt);
        w[0] = fmt0;
        const float e = expf(-x);
        const float b = 0.5f / x;
        const float fmt1 = b * (fmt0 - e);
        r[0] = fmt1 / fmt0;
        return;
    }

    const float *datax = SINGLE_ROOT_RW_DATA + SINGLE_DEGREE1*INTERVALS * nroots*(nroots-1);
    int it = (int)(x * 0.4f);
    const float u = (x - it * 2.5f) * 0.8f - 1.0f;
    const float u2 = u * 2.0f;
    for (int i = rt_id; i < nroots*2; i += stride) {
        const float *c = datax + i * SINGLE_DEGREE1 * INTERVALS;
        //for i in range(2, degree + 1):
        //    c0, c1 = c[degree-i] - c1, c0 + c1*u2
        float c0 = c[it + SINGLE_DEGREE   *INTERVALS];
        float c1 = c[it +(SINGLE_DEGREE-1)*INTERVALS];
        float c2, c3;
#pragma unroll
        for (int n = SINGLE_DEGREE-2; n > 0; n-=2) {
            c2 = c[it + n   *INTERVALS] - c1;
            c3 = c0 + c1*u2;
            c1 = c2 + c3*u2;
            c0 = c[it +(n-1)*INTERVALS] - c3;
        }
        if (SINGLE_DEGREE % 2 == 0) {
            c2 = c[it] - c1;
            c3 = c0 + c1*u2;
            rw[i*block_size] = c2 + c3*u;
        } else {
            rw[i*block_size] = c0 + c1*u;
        }
    }
}

template <typename FloatType>
__device__
static void rys_roots_mixed_precision(const int nroots, const FloatType x, FloatType *rw,
                                      const int block_size, const int rt_id, const int stride)
{
    if constexpr(IS_DOUBLE(FloatType))
        rys_roots(nroots, x, rw, block_size, rt_id, stride);
    else
        rys_roots_single_precision(nroots, x, rw, block_size, rt_id, stride);
}

// rys_roots for range-separation Coulomb
template <typename FloatType>
__device__
static void rys_roots_rs_mixed_precision(const int nroots, const FloatType theta, const FloatType rr, const FloatType omega,
                                         FloatType *rw, const int block_size, const int rt_id, const int stride)
{
    const FloatType theta_rr = theta * rr;
    if (omega == 0) {
        rys_roots_mixed_precision<FloatType>(nroots, theta_rr, rw, block_size, rt_id, stride);
    } else if (omega > 0) {
        const FloatType theta_fac = omega * omega / (omega * omega + theta);
        rys_roots_mixed_precision<FloatType>(nroots, theta_fac*theta_rr, rw, block_size, rt_id, stride);
        __syncthreads();
        const FloatType sqrt_theta_fac = sqrt(theta_fac);
        for (int irys = rt_id; irys < nroots; irys+=stride) {
            rw[ irys*2   *block_size] *= theta_fac;
            rw[(irys*2+1)*block_size] *= sqrt_theta_fac;
        }
    } else {
        int _nroots = nroots / 2;
        FloatType *rw1 = rw + nroots*block_size;
        rys_roots_mixed_precision<FloatType>(_nroots, theta_rr, rw1, block_size, rt_id, stride);
        const FloatType theta_fac = omega * omega / (omega * omega + theta);
        rys_roots_mixed_precision<FloatType>(_nroots, theta_fac*theta_rr, rw, block_size, rt_id, stride);
        __syncthreads();
        const FloatType sqrt_theta_fac = -sqrt(theta_fac);
        for (int irys = rt_id; irys < _nroots; irys+=stride) {
            rw[ irys*2   *block_size] *= theta_fac;
            rw[(irys*2+1)*block_size] *= sqrt_theta_fac;
        }
    }
}
