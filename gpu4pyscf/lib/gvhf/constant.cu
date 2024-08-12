/* Copyright 2023 The GPU4PySCF Authors. All Rights Reserved.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "constant.cuh"

__constant__ BasisProdCache c_bpcache;
__constant__ BasisProdCacheSinglePrecision c_bpcache_single;
__constant__ int16_t c_idx4c[NFffff*3];
__constant__ int c_idx[TOT_NF*3]; 
__constant__ int c_l_locs[GPU_LMAX+2];

__constant__ BasisProdOffsets c_offsets[MAX_STREAMS];
__constant__ GINTEnvVars c_envs[MAX_STREAMS];
__constant__ JKMatrix c_jk[MAX_STREAMS];