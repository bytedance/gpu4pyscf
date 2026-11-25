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
#include "gpu_compat.h"

__global__ static
void recontract_cderi_kernel(double *out, const double *inp,
                             const int *out_idx, const int *out_offsets,
                             const double *coef, int naux, int out_ncol,
                             int inp_ncol, int ncomp)
{
    setup_context();
    size_t width = (size_t)naux * ncomp;
    size_t size = (size_t)inp_ncol * width;
    for (size_t id = (size_t)blockIdx_x * blockDim_x + threadIdx_x;
         id < size; id += (size_t)gridDim_x * blockDim_x) {
        int pair = id / width;
        size_t a = id % width;
        size_t row = a / ncomp;
        int component = a % ncomp;
        double value = inp[id];
        for (int k = out_offsets[pair]; k < out_offsets[pair+1]; ++k) {
            size_t dst = (row * out_ncol + out_idx[k]) * ncomp + component;
            atomicAdd(out + dst, coef[k] * value);
        }
    }
}

extern "C" {
int PBCrecontract_cderi_gpu(double *out, const double *inp,
                            const int *out_idx, const int *out_offsets,
                            const double *coef, int naux, int out_ncol,
                            int inp_ncol, int ncomp, cudaStream_t stream)
{
    size_t size = (size_t)inp_ncol * naux * ncomp;
    if (size == 0) return 0;
    size_t nblocks = (size + 255) / 256;
    if (nblocks > 65535) nblocks = 65535;
    auto blocks = make_grid(nblocks);
    auto threads = make_block(256);
    LAUNCH_KERNEL(recontract_cderi_kernel, blocks, threads, 0, stream,
                  out, inp, out_idx, out_offsets, coef,
                  naux, out_ncol, inp_ncol, ncomp);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU Error in PBCrecontract_cderi_gpu: %s\n",
                cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}
