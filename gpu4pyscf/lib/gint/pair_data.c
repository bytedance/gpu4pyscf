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
#include <math.h>
#include "g2e.h"

static double CINTcommon_fac_sp(int l)
{
        switch (l) {
                case 0: return 0.282094791773878143;
                case 1: return 0.488602511902919921;
                default: return 1;
        }
}

void GINTinit_contraction_types(BasisProdCache *bpcache,
                                int *bas_pair2shls, int *bas_pairs_locs, int ncptype,
                                int *atm, int natm, int *bas, int nbas, double *env)
{
        bpcache->ncptype = ncptype;
        bpcache->bas_pair2shls = bas_pair2shls;
        bpcache->bas_pairs_locs = bas_pairs_locs;

        ContractionProdType *cptype = (ContractionProdType *)malloc(sizeof(ContractionProdType) * ncptype);
        bpcache->cptype = cptype;
        int *primitive_pairs_locs = (int *)malloc(sizeof(int) * (ncptype + 1));
        bpcache->primitive_pairs_locs = primitive_pairs_locs;

        int n;
        int n_bas_pairs = bas_pairs_locs[ncptype];
        int *bas_pair2bra = bas_pair2shls;
        int *bas_pair2ket = bas_pair2shls + n_bas_pairs;
        int n_primitive_pairs = 0;
        primitive_pairs_locs[0] = 0;

        for (n = 0; n < ncptype; n++, cptype++) {
                int pair_id = bas_pairs_locs[n];
                int npairs = bas_pairs_locs[n+1] - bas_pairs_locs[n];
                int ish = bas_pair2bra[pair_id];
                int jsh = bas_pair2ket[pair_id];
                int li = bas[ANG_OF + ish * BAS_SLOTS];
                int lj = bas[ANG_OF + jsh * BAS_SLOTS];
                int npi = bas[NPRIM_OF + ish * BAS_SLOTS];
                int npj = bas[NPRIM_OF + jsh * BAS_SLOTS];
                cptype->l_bra = li;
                cptype->l_ket = lj;
                cptype->nprim_12 = npi * npj;
                cptype->npairs = npairs;
                n_primitive_pairs += npairs * npi * npj;
                primitive_pairs_locs[n+1] = n_primitive_pairs;
        }
}

void GINTsort_bas_coordinates(double *bas_coords, int *atm, int natm,
                              int *bas, int nbas, double *env)
{
        int ib, atm_id, ptr_coord;
        double *bas_x = bas_coords;
        double *bas_y = bas_x + nbas;
        double *bas_z = bas_y + nbas;
        for (ib = 0; ib < nbas; ib++) {
                atm_id = bas[ATOM_OF + ib * BAS_SLOTS];
                ptr_coord = atm[PTR_COORD + atm_id * ATM_SLOTS];
                bas_x[ib] = env[ptr_coord  ];
                bas_y[ib] = env[ptr_coord+1];
                bas_z[ib] = env[ptr_coord+2];
        }
}

void GINTinit_exponent(double *exp, int *bas, int nbas, double *env)
{
        int ib, ptr;
        for (ib = 0; ib < nbas; ib++) {
                ptr = bas[PTR_EXP + ib * BAS_SLOTS];
                exp[ib] = env[ptr];
        }
}

void GINTinit_aexyz(double *aexyz, const BasisProdCache *bpcache, const double diag_fac,
                    const int *atm, const int natm, const int *bas, const int nbas, const double *env)
{
        const int ncptype = bpcache->ncptype;
        const int n_bas_pairs = bpcache->bas_pairs_locs[ncptype];
        const int n_primitive_pairs = bpcache->primitive_pairs_locs[ncptype];
        const int *bas_pair2bra = bpcache->bas_pair2shls;
        const int *bas_pair2ket = bpcache->bas_pair2shls + n_bas_pairs;

        double *a12 = aexyz;
        double *e12 = a12 + n_primitive_pairs;
        double *x12 = e12 + n_primitive_pairs;
        double *y12 = x12 + n_primitive_pairs;
        double *z12 = y12 + n_primitive_pairs;
        double *a1  = z12 + n_primitive_pairs;
        double *a2  = a1 + n_primitive_pairs;

        int off = 0;
        for (int pair_id = 0; pair_id < n_bas_pairs; pair_id++) {
                const int ish = bas_pair2bra[pair_id];
                const int jsh = bas_pair2ket[pair_id];
                const int npi = bas[NPRIM_OF + ish * BAS_SLOTS];
                const int npj = bas[NPRIM_OF + jsh * BAS_SLOTS];
                const int ia = bas[ATOM_OF + ish * BAS_SLOTS];
                const int ja = bas[ATOM_OF + jsh * BAS_SLOTS];
                const int li = bas[ANG_OF + ish * BAS_SLOTS];
                const int lj = bas[ANG_OF + jsh * BAS_SLOTS];
                const double* ai = env + bas[PTR_EXP + ish * BAS_SLOTS];
                const double* aj = env + bas[PTR_EXP + jsh * BAS_SLOTS];
                const double* ci = env + bas[PTR_COEFF + ish * BAS_SLOTS];
                const double* cj = env + bas[PTR_COEFF + jsh * BAS_SLOTS];
                const double* ri = env + atm[PTR_COORD + ia * ATM_SLOTS];
                const double* rj = env + atm[PTR_COORD + ja * ATM_SLOTS];
                const double norm = CINTcommon_fac_sp(li) * CINTcommon_fac_sp(lj);

                const double rx = ri[0] - rj[0];
                const double ry = ri[1] - rj[1];
                const double rz = ri[2] - rj[2];
                const double dist_ij = rx * rx + ry * ry + rz * rz;

                for (int count = off, ip = 0; ip < npi; ip++) {
                for (int jp = 0; jp < npj; jp++, count++) {
                        const double aij = ai[ip] + aj[jp];
                        a12[count] = aij;
                        e12[count] = norm * ci[ip] * cj[jp] *
                                exp(-dist_ij * ai[ip] * aj[jp] / aij);
                        x12[count] = (ai[ip]*ri[0] + aj[jp]*rj[0]) / aij;
                        y12[count] = (ai[ip]*ri[1] + aj[jp]*rj[1]) / aij;
                        z12[count] = (ai[ip]*ri[2] + aj[jp]*rj[2]) / aij;
                        a1[count] = ai[ip];
                        a2[count] = aj[jp];
                } }

                if (ish == jsh) {
                        for (int count = 0; count < npi * npj; count++) {
                                e12[off + count] *= diag_fac;
                        }
                }
                off += npi * npj;
        }
}

void GINTinit_populate_pair_data(double *aexyz, int *i0j0, const BasisProdCache *bpcache, const double diag_fac,
                                 const int *atm, const int natm, const int *bas, const int nbas, const int *ao_loc,
                                 const double *env)
{
        const int ncptype = bpcache->ncptype;
        const int n_bas_pairs = bpcache->bas_pairs_locs[ncptype];
        const int n_primitive_pairs = bpcache->primitive_pairs_locs[ncptype];
        const int *bas_pair2bra = bpcache->bas_pair2shls;
        const int *bas_pair2ket = bpcache->bas_pair2shls + n_bas_pairs;

        double *a12 = aexyz;
        double *e12 = a12 + n_primitive_pairs;
        double *x12 = e12 + n_primitive_pairs;
        double *y12 = x12 + n_primitive_pairs;
        double *z12 = y12 + n_primitive_pairs;
        double *a1  = z12 + n_primitive_pairs;
        double *x1  = a1  + n_primitive_pairs;
        double *y1  = x1  + n_bas_pairs;
        double *z1  = y1  + n_bas_pairs;
        int *i0 = i0j0;
        int *j0 = i0 + n_bas_pairs;

        int off = 0;
        for (int pair_id = 0; pair_id < n_bas_pairs; pair_id++) {
                const int ish = bas_pair2bra[pair_id];
                const int jsh = bas_pair2ket[pair_id];
                const int npi = bas[NPRIM_OF + ish * BAS_SLOTS];
                const int npj = bas[NPRIM_OF + jsh * BAS_SLOTS];
                const int ia = bas[ATOM_OF + ish * BAS_SLOTS];
                const int ja = bas[ATOM_OF + jsh * BAS_SLOTS];
                const int li = bas[ANG_OF + ish * BAS_SLOTS];
                const int lj = bas[ANG_OF + jsh * BAS_SLOTS];
                const double* ai = env + bas[PTR_EXP + ish * BAS_SLOTS];
                const double* aj = env + bas[PTR_EXP + jsh * BAS_SLOTS];
                const double* ci = env + bas[PTR_COEFF + ish * BAS_SLOTS];
                const double* cj = env + bas[PTR_COEFF + jsh * BAS_SLOTS];
                const double* ri = env + atm[PTR_COORD + ia * ATM_SLOTS];
                const double* rj = env + atm[PTR_COORD + ja * ATM_SLOTS];
                const double norm = CINTcommon_fac_sp(li) * CINTcommon_fac_sp(lj);

                x1[pair_id] = ri[0];
                y1[pair_id] = ri[1];
                z1[pair_id] = ri[2];
                i0[pair_id] = ao_loc[ish];
                j0[pair_id] = ao_loc[jsh];

                const double rx = ri[0] - rj[0];
                const double ry = ri[1] - rj[1];
                const double rz = ri[2] - rj[2];
                const double dist_ij = rx * rx + ry * ry + rz * rz;

                for (int count = off, ip = 0; ip < npi; ip++) {
                for (int jp = 0; jp < npj; jp++, count++) {
                        const double aij = ai[ip] + aj[jp];
                        a12[count] = aij;
                        e12[count] = norm * ci[ip] * cj[jp] * exp(-dist_ij * ai[ip] * aj[jp] / aij);
                        x12[count] = (ai[ip]*ri[0] + aj[jp]*rj[0]) / aij;
                        y12[count] = (ai[ip]*ri[1] + aj[jp]*rj[1]) / aij;
                        z12[count] = (ai[ip]*ri[2] + aj[jp]*rj[2]) / aij;
                        a1[count] = ai[ip];
                } }

                if (ish == jsh) {
                        for (int count = 0; count < npi * npj; count++) {
                                e12[off + count] *= diag_fac;
                        }
                }
                off += npi * npj;
        }
}
