/*
 * gpu4pyscf is a plugin to use Nvidia GPU in PySCF package
 *
 * Copyright (C) 2022 Qiming Sun
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

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#include "gint.h"
#include "config.h"
#include "cuda_alloc.cuh"
#include "g2e.h"
/*
#include "cint2e.cuh"
#include "fill_ints.cu"
#include "g2e.cu"
#include "rys_roots.cu"
#include "g2e_root2.cu"
#include "g2e_root3.cu"
#include "g3c2e.cu"
#include "g3c2e_ip1.cu"
#include "g3c2e_ip2.cu"
*/
extern "C" {
__host__
void GINTdel_basis_prod(BasisProdCache **pbp)
{
    BasisProdCache *bpcache = *pbp;
    if (bpcache == NULL) {
        return;
    }

    if (bpcache->cptype != NULL) {
        free(bpcache->cptype);
        free(bpcache->primitive_pairs_locs);
    }

    if (bpcache->aexyz != NULL) {
        free(bpcache->aexyz);
    }

    if (bpcache->a12 != NULL) {
        FREE(bpcache->bas_coords);
        FREE(bpcache->bas_pair2bra);
        FREE(bpcache->ao_loc);
        FREE(bpcache->a12);
    }

    free(bpcache);
    *pbp = NULL;
}

__host__
void GINTdel_basis_prod_single_precision(BasisProdCache **pbp, BasisProdCacheSinglePrecision **pbp_single)
{
    BasisProdCache *bpcache = *pbp;
    if (bpcache == NULL) {
        return;
    }

    if (bpcache->cptype != NULL) {
        free(bpcache->cptype);
        free(bpcache->primitive_pairs_locs);
    }

    if (bpcache->aexyz != NULL) {
        free(bpcache->aexyz);
    }

    if (bpcache->a12 != NULL) {
        FREE(bpcache->bas_coords);
        FREE(bpcache->bas_pair2bra);
        FREE(bpcache->ao_loc);
        FREE(bpcache->a12);
    }

    free(bpcache);
    *pbp = NULL;

    BasisProdCacheSinglePrecision *bpcache_single = *pbp_single;
    if (bpcache_single->aexyz != NULL) {
        free(bpcache_single->aexyz);
    }
    if (bpcache_single->a12 != NULL) {
        FREE(bpcache_single->bas_coords);
        FREE(bpcache_single->a12);
    }

    free(bpcache_single);
    *pbp_single = NULL;
}

void GINTinit_basis_prod(BasisProdCache **pbp, double diag_fac, int *ao_loc,
                         int *bas_pair2shls, int *bas_pairs_locs, int ncptype,
                         int *atm, int natm, int *bas, int nbas, double *env)
{
    BasisProdCache *bpcache = (BasisProdCache *)malloc(sizeof(BasisProdCache));
    memset(bpcache, 0, sizeof(BasisProdCache));
    *pbp = bpcache;

    GINTinit_contraction_types(bpcache, bas_pair2shls, bas_pairs_locs, ncptype,
                               atm, natm, bas, nbas, env);
    int n_bas_pairs = bpcache->bas_pairs_locs[ncptype];
    int n_primitive_pairs = bpcache->primitive_pairs_locs[ncptype];
    double *aexyz = (double *)malloc(sizeof(double) * n_primitive_pairs * 7);
    GINTinit_aexyz(aexyz, bpcache, diag_fac, atm, natm, bas, nbas, env);
    bpcache->aexyz = aexyz;
    bpcache->bas_pair2shls = bas_pair2shls;

    // initialize ao_loc on GPU
    DEVICE_INIT(int, d_ao_loc, ao_loc, nbas+1);
    bpcache->ao_loc = d_ao_loc;

    // initialize basis coordinates on GPU memory
    bpcache->nbas = nbas;
    double *bas_coords = (double *)malloc(sizeof(double) * nbas * 3);
    GINTsort_bas_coordinates(bas_coords, atm, natm, bas, nbas, env);
    DEVICE_INIT(double, d_bas_coords, bas_coords, nbas * 3);
    bpcache->bas_coords = d_bas_coords;
    free(bas_coords);

    // initialize pair data on GPU memory
    DEVICE_INIT(double, d_aexyz, aexyz, n_primitive_pairs * 7);
    DEVICE_INIT(int, d_bas_pair2shls, bas_pair2shls, n_bas_pairs * 2);
    bpcache->a12 = d_aexyz;
    bpcache->e12 = d_aexyz + n_primitive_pairs * 1;
    bpcache->x12 = d_aexyz + n_primitive_pairs * 2;
    bpcache->y12 = d_aexyz + n_primitive_pairs * 3;
    bpcache->z12 = d_aexyz + n_primitive_pairs * 4;
    bpcache->a1  = d_aexyz + n_primitive_pairs * 5;
    bpcache->a2  = d_aexyz + n_primitive_pairs * 6;
    bpcache->bas_pair2bra = d_bas_pair2shls;
    bpcache->bas_pair2ket = d_bas_pair2shls + n_bas_pairs;
}

void GINTinit_basis_prod_mixed_precision(BasisProdCache **pbp_double, BasisProdCacheSinglePrecision **pbp_single, double diag_fac, int *ao_loc,
                                         int *bas_pair2shls, int *bas_pairs_locs, int ncptype,
                                         int *atm, int natm, int *bas, int nbas, double *env)
{
    BasisProdCache *bpcache = (BasisProdCache *)malloc(sizeof(BasisProdCache));
    memset(bpcache, 0, sizeof(BasisProdCache));
    *pbp_double = bpcache;

    GINTinit_contraction_types(bpcache, bas_pair2shls, bas_pairs_locs, ncptype,
                               atm, natm, bas, nbas, env);
    int n_bas_pairs = bpcache->bas_pairs_locs[ncptype];
    int n_primitive_pairs = bpcache->primitive_pairs_locs[ncptype];
    double *aexyz = (double *)malloc(sizeof(double) * n_primitive_pairs * 7);
    GINTinit_aexyz(aexyz, bpcache, diag_fac, atm, natm, bas, nbas, env);
    bpcache->aexyz = aexyz;
    bpcache->bas_pair2shls = bas_pair2shls;

    // initialize ao_loc on GPU
    DEVICE_INIT(int, d_ao_loc, ao_loc, nbas+1);
    bpcache->ao_loc = d_ao_loc;

    // initialize basis coordinates on GPU memory
    bpcache->nbas = nbas;
    double *bas_coords = (double *)malloc(sizeof(double) * nbas * 3);
    GINTsort_bas_coordinates(bas_coords, atm, natm, bas, nbas, env);
    DEVICE_INIT(double, d_bas_coords, bas_coords, nbas * 3);
    bpcache->bas_coords = d_bas_coords;

    // initialize pair data on GPU memory
    DEVICE_INIT(double, d_aexyz, aexyz, n_primitive_pairs * 7);
    DEVICE_INIT(int, d_bas_pair2shls, bas_pair2shls, n_bas_pairs * 2);
    bpcache->a12 = d_aexyz;
    bpcache->e12 = d_aexyz + n_primitive_pairs * 1;
    bpcache->x12 = d_aexyz + n_primitive_pairs * 2;
    bpcache->y12 = d_aexyz + n_primitive_pairs * 3;
    bpcache->z12 = d_aexyz + n_primitive_pairs * 4;
    bpcache->a1  = d_aexyz + n_primitive_pairs * 5;
    bpcache->a2  = d_aexyz + n_primitive_pairs * 6;
    bpcache->bas_pair2bra = d_bas_pair2shls;
    bpcache->bas_pair2ket = d_bas_pair2shls + n_bas_pairs;

    BasisProdCacheSinglePrecision *bpcache_single = (BasisProdCacheSinglePrecision *)malloc(sizeof(BasisProdCacheSinglePrecision));
    memset(bpcache_single, 0, sizeof(BasisProdCacheSinglePrecision));
    *pbp_single = bpcache_single;

    bpcache_single->nbas = bpcache->nbas;
    bpcache_single->ncptype = bpcache->ncptype;
    bpcache_single->cptype = bpcache->cptype; // This is pointer shallow copy!
    bpcache_single->bas_pairs_locs = bpcache->bas_pairs_locs; // This is pointer shallow copy!
    bpcache_single->primitive_pairs_locs = bpcache->primitive_pairs_locs; // This is pointer shallow copy!
    bpcache_single->bas_pair2shls = bpcache->bas_pair2shls; // This is pointer shallow copy!

    float *aexyz_single = (float *)malloc(sizeof(float) * n_primitive_pairs * 7);
    for (int i = 0; i < n_primitive_pairs * 7; i++)
        aexyz_single[i] = static_cast<float>(aexyz[i]);
    // for (int i = 0; i < n_primitive_pairs * 7; i++)
    //     if (isnan(aexyz_single[i])) {
    //         printf("i = %d is nan, aexyz[i] = %.10f\n", aexyz[i]);
    //         exit(1);
    //     }
    bpcache_single->aexyz = aexyz_single;

    bpcache_single->bas_pair2bra = bpcache->bas_pair2bra; // This is device pointer shallow copy!
    bpcache_single->bas_pair2ket = bpcache->bas_pair2ket; // This is device pointer shallow copy!
    bpcache_single->ao_loc = bpcache->ao_loc; // This is device pointer shallow copy!

    float *bas_coords_single = (float *)malloc(sizeof(float) * nbas * 3);
    for (int i = 0; i < nbas * 3; i++)
        bas_coords_single[i] = static_cast<float>(bas_coords[i]);
    // for (int i = 0; i < nbas * 3; i++)
    //     if (isnan(bas_coords_single[i])) {
    //         printf("i = %d is nan, bas_coords[i] = %.10f\n", bas_coords[i]);
    //         exit(1);
    //     }
    DEVICE_INIT(float, d_bas_coords_single, bas_coords_single, nbas * 3);
    bpcache_single->bas_coords = d_bas_coords_single;

    DEVICE_INIT(float, d_aexyz_single, aexyz_single, n_primitive_pairs * 7);
    bpcache_single->a12 = d_aexyz_single;
    bpcache_single->e12 = d_aexyz_single + n_primitive_pairs * 1;
    bpcache_single->x12 = d_aexyz_single + n_primitive_pairs * 2;
    bpcache_single->y12 = d_aexyz_single + n_primitive_pairs * 3;
    bpcache_single->z12 = d_aexyz_single + n_primitive_pairs * 4;
    bpcache_single->a1  = d_aexyz_single + n_primitive_pairs * 5;
    bpcache_single->a2  = d_aexyz_single + n_primitive_pairs * 6;

    free(bas_coords);
    free(bas_coords_single);
}
}

