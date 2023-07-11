/* Modifications Copyright (C) 2023 Intel Corporation
 *
 * This Program is subject to the terms of the GNU General Public License v3.0 only.​
 * If a copy of the license was not distributed with this file, ​
 * you can obtain one at https://spdx.org/licenses/GPL-3.0-only.html
 *​
 *
 * SPDX-License-Identifier: GPL-3.0-only
 */

#include "hip/hip_runtime.h"
#include <iostream>
#include <cmath>
#include <cstring>
// #include <omp.h>
#include "gpu_modelling_kernels.h"

#define PI 3.141592654

#define a0 1.125f
#define a1 -1.0f/24.0f

#define Block_size1 16
#define Block_size2 16

using namespace std;

// set the positions of sources and geophones in whole domain
void cpu_set_sg(int *sxz, int *sx_pos, int *sz_pos, int ns, int npml, int nnz)
{
    int id;

    for(id = 0; id < ns; id++)
        sxz[id] = nnz * (sx_pos[id] + npml) + (sz_pos[id] + npml);
}// End of cpu_set_sg function

void cpu_ricker(float fm, float dt, double* wlt, int nts)
{
    // iss is hardcoded to 2
    // this function calculates source wavelet at different time points
    int iss   = 2;
    double t0 = 1.5 * sqrt(6.0) / (PI * (double)fm);
    double da = PI * (double)fm;
    int i;
    double t, a, a2;
   
    for(i = 0; i < nts; i++)
    {
        t  = (double)(i * dt);
        a  = PI * (double)fm * (t - t0);
        a2 = a * a;

        // if(iss == 3)
        // {
        //     wlt[i] = (a / da) * exp(-a2);
        // }
        // else if(iss == 2)
        // {
            wlt[i] = (1.0f - 2.0f * a2) * exp(-a2);
        // }
        // else
        // {
        //     wlt[i] = -4 * a * da * exp(-a2) - 2.0f * a * da * (1.0f - 2.0f * a2) * exp(-a2);
        // }//end if

    }//End of for
}//End of cpu_ricker function

void cpu_stagger(float *vpe, float *rhoe, float *bue, float *bu2, float *bu1, double *kappa, int nnz, int nnx)
{
    // kappa = velocity * velocity * density
    // bue   = 1 / density [set to 10B when density is zero (or negative)]
    // bue1  = avg of bue of two consecutive vertical   points (i.e. z direction)
    // bue2  = avg of bue of two consecutive horizontal points (i.e. x direction)
    int i1, i2, id;
    double vp2;

    #pragma omp parallel for private(i1,i2,id,vp2) schedule(dynamic, 1)
    for(i2 = 0; i2 < nnx; i2++)
    {
        for(i1 = 0; i1 < nnz; i1++)
        {
            id = i1 + i2 * nnz;
            vp2 = (double)vpe[id] * (double)vpe[id];
            kappa[id] = vp2 * (double)rhoe[id];
            if(rhoe[id] > 0.0f)
            {
                bue[id] = 1.0f / rhoe[id];
            }
            else
            {
                bue[id] = 10000000000.0f;
            }
        }
    }

    //bu2 (grid coincident with Tauzy)
    #pragma omp parallel for private(i1,i2,id) schedule(dynamic, 1)
    for(i2 = 0; i2 < nnx; i2++)
    {
        for(i1 = 0; i1 < nnz - 1; i1++)
        {
            id = i1 + i2 * nnz;
            bu1[id] = 0.5f * (bue[id + 1] + bue[id]);
        }   
        
        bu1[i2 * nnz + nnz - 1] = bu2[i2 * nnz + nnz - 2];
    }

    //bu1 (grid coincident with Tauxy)
    #pragma omp parallel for private(i1,i2,id) schedule(dynamic, 1)
    for(i2 = 0; i2 < nnx - 1; i2++)
    {
        for(i1 = 0; i1 < nnz; i1++)
        {
            id = i1 + i2 * nnz;
            bu2[id] = 0.5f * (bue[id + nnz] + bue[id]);
        }
    }

    #pragma omp parallel for private(i1) schedule(dynamic, 1)
    for(i1 = 0; i1 < nnz; i1++)
    {
        bu2[(nnx - 1) * nnz + i1] = bu1[(nnx - 2) * nnz + i1];
    }
}// End of cpu_stagger function

void cpu_pml_coefficient_a(float fac, float *damp, float *damp1, float *spg, float *spg1, int npml, int n)
{
    // damp and damp1 are both 1.0f at the middle
    // they decrease exponentially towards the edge in the pml layer
    // same exponential decay is used for start and end
    int i;
    float x, temp;

    #pragma omp parallel for private(i,x,temp) schedule(dynamic, 1)
    for(i = 0; i < npml; i++)
    {
        x       = (float)(npml - (i + 1));
        temp    = (fac * x);
        spg[ i] = exp(      -(temp * temp));
        spg1[i] = exp(-0.5 * (temp * temp));
    }

    #pragma omp parallel for private(i) schedule(dynamic, 1)
    for(i = 0; i < npml; i++)
    {
        damp[ i]            = spg[ i];
        damp1[i]            = spg1[i];
        damp[ npml + n + i] = spg[ npml - i - 1];
        damp1[npml + n + i] = spg1[npml - i - 1];
    }

    #pragma omp parallel for private(i) schedule(dynamic, 1)
    for(i = 0; i < n; i++)
    {
        damp[ i + npml] = 1.0f;
        damp1[i + npml] = 1.0f;
    }
}// End of cpu_pml_coefficient_a function

void cpu_pml_coefficient_b(float fac,float *damp, float *damp1, float *spg, float *spg1, int npml, int n)
{
    // damp and damp1 are both 1.0f at the middle
    // they decrease exponentially towards the edge in the pml layer
    // separate exponential decays are used for start and end
    int i;
    float x, temp;

    #pragma omp parallel for private(i,x,temp) schedule(dynamic, 1)
    for(i = 0; i < npml - 1; i++)
    {
        x       = (float)(npml - 1 - (i + 1)) + 0.5f;
        temp    = (fac * x);
        spg[ i] = exp(      -(temp * temp));
        spg1[i] = exp(-0.5 * (temp * temp));
    }

    #pragma omp parallel for private(i) schedule(dynamic, 1)
    for(i = 0; i < npml - 1; i++)
    {
        damp[ i] = spg[ i];
        damp1[i] = spg1[i];
    }

    #pragma omp parallel for private(i) schedule(dynamic, 1)
    for(i = 0; i < n + 1; i++)
    {
        damp[ i + npml - 1] = 1.0f;
        damp1[i + npml - 1] = 1.0f;
    }

    #pragma omp parallel for private(i,x,temp) schedule(dynamic, 1)
    for(i = 0; i < npml; i++)
    {
        x       = (float)((i + 1) - 1) + 0.5f;
        temp    = (fac * x);
        spg[ i] = exp(      -(temp * temp));
        spg1[i] = exp(-0.5 * (temp * temp));
    }

    #pragma omp parallel for private(i) schedule(dynamic, 1)
    for(i = 0; i < npml; i++)
    {
       damp[ npml + n + i] = spg[ i];
       damp1[npml + n + i] = spg1[i];
    }
}// End of cpu_pml_coefficient_b function

__global__ void gpu_record(double *d_p, float *seis_kt, int *d_Gxz, int ng)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    
    if(id < ng)
        seis_kt[id] = (float)d_p[d_Gxz[id]];
}// End of gpu_record function

__global__ void gpu_fdoperator420(double *vx, double *der2, int nnz, int nnx)
{
    int i1, i2, id;
    i1 = threadIdx.x + blockIdx.x * blockDim.x;                 // horizontal (z) direction
    i2 = threadIdx.y + blockIdx.y * blockDim.y;                 // vertical   (x) direction
    id = i1 + i2 * nnz;

    __shared__ double s_vx[Block_size1][Block_size2 + 3];       // s_vx size: blockDim.x X (2 + blockDim.y + 1)
    s_vx[threadIdx.x][threadIdx.y + 2] = vx[id];                // copy global memory for this block to 0 to (blockDim.x - 1) rows and 2 to (blockDim.y + 1) cols
                                                                // note: all rows are filled up but three cols (0, 1, and blockDim.y + 2) are empty at this point
    if (threadIdx.y < 2)                                        // fill up cols 0 and 1
    {
        if(blockIdx.y)                                          // if blockIdx.y (vertical block) > 0
            s_vx[threadIdx.x][threadIdx.y] = vx[id - 2 * nnz];  // copy last two rows of previous vertical block to cols 0 and 1
        else                                                    // if blockIdx.y (vertical block) = 0
            s_vx[threadIdx.x][threadIdx.y] = 0.0;               // fill cols 0 and 1 with zeros
    }
    if (threadIdx.y > Block_size2 - 2)                          // fill up col blockDim.y + 2
    {
        if(blockIdx.y < gridDim.y - 1)                          // if blockIdx.y (vertical block) is not the last block
            s_vx[threadIdx.x][threadIdx.y + 3] = vx[id + nnz];  // copy first row of next vertical block to col blockDim.y + 2
        else                                                    // if blockIdx.y (vertical block) is the last block
            s_vx[threadIdx.x][threadIdx.y + 3] = 0.0;           // fill col blockDim.y + 2 with zeros
    }

    __syncthreads();

    der2[id] = a0 * (s_vx[threadIdx.x][threadIdx.y + 2] - s_vx[threadIdx.x][threadIdx.y + 1])
             + a1 * (s_vx[threadIdx.x][threadIdx.y + 3] - s_vx[threadIdx.x][threadIdx.y    ]);
}// End of gpu_fdoperator420 function

__global__ void gpu_fdoperator410(double *vz, double *der1, int nnz, int nnx)
{
    int i1, i2, id;
    i1 = threadIdx.x + blockIdx.x * blockDim.x;
    i2 = threadIdx.y + blockIdx.y * blockDim.y;
    id = i1 + i2 * nnz;

    __shared__ double s_vz[Block_size1 + 3][Block_size2];
    s_vz[threadIdx.x + 2][threadIdx.y] = vz[id];

    if (threadIdx.x < 2)
    {
        if (blockIdx.x)
            s_vz[threadIdx.x][threadIdx.y] = vz[id - 2];
        else
            s_vz[threadIdx.x][threadIdx.y] = 0.0;
    }
    if (threadIdx.x > Block_size1 - 2)
    {
        if (blockIdx.x < gridDim.x - 1)
            s_vz[threadIdx.x + 3][threadIdx.y] = vz[id + 1];
        else
            s_vz[threadIdx.x + 3][threadIdx.y] = 0.0;
    }

    __syncthreads();

    der1[id] = a0 * (s_vz[threadIdx.x + 2][threadIdx.y] - s_vz[threadIdx.x + 1][threadIdx.y])
             + a1 * (s_vz[threadIdx.x + 3][threadIdx.y] - s_vz[threadIdx.x    ][threadIdx.y]);
}// End of gpu_fdoperator410 function

__global__ void gpu_fdoperator421(double *p, double *der2, int nnz, int nnx)
{
    int i1, i2, id;
    i1 = threadIdx.x + blockIdx.x * blockDim.x;
    i2 = threadIdx.y + blockIdx.y * blockDim.y;
    id = i1 + i2 * nnz;

    __shared__ double s_p[Block_size1][Block_size2 + 3];
    s_p[threadIdx.x][threadIdx.y + 1] = p[id];

    if (threadIdx.y < 1)
    {
        if(blockIdx.y)
            s_p[threadIdx.x][threadIdx.y] = p[id - nnz];
        else
            s_p[threadIdx.x][threadIdx.y] = 0.0;
    }
    if (threadIdx.y > Block_size2 - 3)
    {
        if(blockIdx.y < gridDim.y - 1)
            s_p[threadIdx.x][threadIdx.y + 3] = p[id + 2 * nnz];
        else
            s_p[threadIdx.x][threadIdx.y + 3] = 0.0;
    }

    __syncthreads();

    der2[id] = a0 * (s_p[threadIdx.x][threadIdx.y + 2] - s_p[threadIdx.x][threadIdx.y + 1])
             + a1 * (s_p[threadIdx.x][threadIdx.y + 3] - s_p[threadIdx.x][threadIdx.y    ]);
}// End of gpu_fdoperator421 function

__global__ void gpu_fdoperator411(double *p, double *der1, int nnz, int nnx)
{
    int i1, i2, id;
    i1 = threadIdx.x + blockIdx.x * blockDim.x;
    i2 = threadIdx.y + blockIdx.y * blockDim.y;
    id = i1 + i2 * nnz;

    __shared__ double s_p[Block_size1 + 3][Block_size2];
    s_p[threadIdx.x + 1][threadIdx.y] = p[id];

    if (threadIdx.x < 1)
    {
        if(blockIdx.x)
            s_p[threadIdx.x][threadIdx.y] = p[id - 1];
        else
            s_p[threadIdx.x][threadIdx.y] = 0.0;
    }
    if (threadIdx.x > Block_size1 - 3)
    {
        if(blockIdx.x < gridDim.x - 1)
            s_p[threadIdx.x + 3][threadIdx.y] = p[id + 2];
        else
            s_p[threadIdx.x + 3][threadIdx.y] = 0.0;
    }

    __syncthreads();

    der1[id] = a0 * (s_p[threadIdx.x + 2][threadIdx.y] - s_p[threadIdx.x + 1][threadIdx.y])
             + a1 * (s_p[threadIdx.x + 3][threadIdx.y] - s_p[threadIdx.x    ][threadIdx.y]);
}// End of gpu_fdoperator411 function

__global__ void gpu_compute_p(double *p, double *px, double *pz, int nnz, int nnx)
{
    int i1, i2, id;
    i1 = threadIdx.x + blockIdx.x * blockDim.x;
    i2 = threadIdx.y + blockIdx.y * blockDim.y;
    id = i1 + i2 * nnz;

    p[id] = px[id] + pz[id];
}// End of gpu_compute_p function

__global__ void gpu_pml_p(double *px, double *pz, double *kappaw2, double *kappaw1, double *der2, double *der1, float *damp2a, float *damp1a, int nnz, int nnx)
{
    int i1, i2, id;
    i1 = threadIdx.x + blockIdx.x * blockDim.x;
    i2 = threadIdx.y + blockIdx.y * blockDim.y;
    id = i1 + i2 * nnz;

    px[id] = damp2a[i2] * px[id] + kappaw2[id] * der2[id];
    pz[id] = damp1a[i1] * pz[id] + kappaw1[id] * der1[id];
}// End of gpu_pml_p function

__global__ void gpu_pml_v(double *vx, double *vz, double *buw2,    double *buw1,    double *der2, double *der1, float *damp2b, float *damp1b, int nnz, int nnx)
{
    int i1, i2, id;
    i1 = threadIdx.x + blockIdx.x * blockDim.x;
    i2 = threadIdx.y + blockIdx.y * blockDim.y;
    id = i1 + i2 * nnz;

    vx[id] = damp2b[i2] * vx[id] + buw2[id] * der2[id];
    vz[id] = damp1b[i1] * vz[id] + buw1[id] * der1[id];
}// End of gpu_pml_v function

__global__ void gpu_transpose(float *inp, float *out, int n1, int n2)
{
    int i1, i2, id1, id2;

    i1 = threadIdx.x + blockIdx.x * blockDim.x;
    i2 = threadIdx.y + blockIdx.y * blockDim.y;

    if(i2 < n2 && i1 < n1)
    {
        id1 = i1 + i2 * n1;
        id2 = i2 + i1 * n2;
        out[id2] = inp[id1];
    }
}// End of gpu_transpose function

__global__ void gpu_incr_p(double *px, double *wlt, double *kappa, float fd_dt, float dx, float dz, int isource, int it)
{
   px[isource] = px[isource] + fd_dt * wlt[it] * kappa[isource] / (dx * dz); 
}// End of gpu_incr_p function
