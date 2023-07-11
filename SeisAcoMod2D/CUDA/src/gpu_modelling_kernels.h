/* Modifications Copyright (C) 2023 Intel Corporation
 *
 * This Program is subject to the terms of the GNU General Public License v3.0 only.​
 * If a copy of the license was not distributed with this file, ​
 * you can obtain one at https://spdx.org/licenses/GPL-3.0-only.html
 *​
 *
 * SPDX-License-Identifier: GPL-3.0-only
 */

#ifndef CPU_MODELLING_KERNELS_H
#define CPU_MODELLING_KERNELS_H

void cpu_set_sg(int *sxz, int *sx_pos, int *sz_pos, int ns, int npml, int nnz);
void cpu_ricker(float fm, float dt, double* wlt, int nts);
void cpu_stagger(float *vpe, float *rhoe, float *bue, float *bu2, float *bu1, double *kappa, int nnz, int nnx);
void cpu_pml_coefficient_a(float fac, float *damp, float *damp1, float *spg, float *spg1, int npml, int n);
void cpu_pml_coefficient_b(float fac, float *damp, float *damp1, float *spg, float *spg1, int npml, int n);

// void cpu_record(double *h_p, float *seis_kt, int *h_Gxz, int ng);
// void cpu_fdoperator420(double *vx, double *der2, int nnz, int nnx);
// void cpu_fdoperator410(double *vz, double *der1, int nnz, int nnx);
// void cpu_fdoperator421(double *p,  double *der2, int nnz, int nnx);
// void cpu_fdoperator411(double *p,  double *der1, int nnz, int nnx);
// void cpu_compute_p(double *p, double *px, double *pz, int nnz, int nnx);
// void cpu_pml_p(double *px, double *pz, double *kappaw2, double *kappaw1, double *der2, double *der1, float *damp2a, float *damp1a, int nnz, int nnx);
// void cpu_pml_v(double *vx, double *vz, double *buw2,    double *buw1,    double *der2, double *der1, float *damp2b, float *damp1b, int nnz, int nnx);
// void cpu_transpose(float *inp, float *out, int n1, int n2);

void FDOperator_4(int it, int isource);

__global__ void gpu_record(double *d_p, float *seis_kt, int *d_Gxz, int ng);
__global__ void gpu_fdoperator420(double *vx, double *der2, int nnz, int nnx);
__global__ void gpu_fdoperator410(double *vz, double *der1, int nnz, int nnx);
__global__ void gpu_fdoperator421(double *p,  double *der2, int nnz, int nnx);
__global__ void gpu_fdoperator411(double *p,  double *der1, int nnz, int nnx);
__global__ void gpu_compute_p(double *p, double *px, double *pz, int nnz, int nnx);
__global__ void gpu_pml_p(double *px, double *pz, double *kappaw2, double *kappaw1, double *der2, double *der1, float *damp2a, float *damp1a, int nnz, int nnx);
__global__ void gpu_pml_v(double *vx, double *vz, double *buw2,    double *buw1,    double *der2, double *der1, float *damp2b, float *damp1b, int nnz, int nnx);
__global__ void gpu_transpose(float *inp, float *out, int n1, int n2);
__global__ void gpu_incr_p(double *px, double *wlt, double *kappa, float fd_dt, float dx, float dz, int isource, int it);

#endif
