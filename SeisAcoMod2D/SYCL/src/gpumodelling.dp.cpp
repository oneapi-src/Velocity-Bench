/* Modifications Copyright (C) 2023 Intel Corporation
 *
 * This Program is subject to the terms of the GNU General Public License v3.0 only.​
 * If a copy of the license was not distributed with this file, ​
 * you can obtain one at https://spdx.org/licenses/GPL-3.0-only.html
 *​
 *
 * SPDX-License-Identifier: GPL-3.0-only
 */

#include <sycl/sycl.hpp>
#include <mpi.h>
#include <iostream>
#include <cmath>
#include <cstring>
// #include <omp.h>
#include <chrono>

#include "modelling.h"
#include "gpu_modelling_kernels.h"
#define PI 3.141592654
#define Block_size1 16
#define Block_size2 16

using namespace std;

const int npml = 32;      /* thickness of PML boundary */

static int      nz1, nx1, nz, nx, nnz, nnx, N, NJ, ns, ng, fd_nt, real_nt, nts;
static float    fm, fd_dt, real_dt, dz, dx, _dz, _dx, rec_len;
static sycl::range<3> dimg0(1, 1, 1), dimb0(1, 1, 1);

float   *v0, *h_vel, *h_rho;
/* variables on device */
int     *sx_pos, *sz_pos;
int     *gx_pos, *gz_pos;
int     *h_Sxz, *h_Gxz;     /* set source and geophone position */
int     *d_Gxz;

double dtdx, dtdz;

float *bu1, *bu2, *bue;
float *spg, *spg1, *damp1a1, *damp2a1, *damp1b1, *damp2b1;
float *damp1a, *damp2a, *damp1b, *damp2b;
float *d_damp1a, *d_damp2a, *d_damp1b, *d_damp2b;

double *h_wlt;  //source wavelet
double *d_wlt;
float *h_dobs;           // seismogram
float *d_dobs, *d_temp;  // seismogram

double *d_p, *d_px, *d_pz, *d_vx, *d_vz, *d_der2, *d_der1;

double *kappa, *kappaw2, *kappaw1;
double *d_kappa, *d_kappaw2, *d_kappaw1;
double *buw2, *buw1;
double *d_buw2, *d_buw1;

void expand(float *b, float *a, int npml, int nnz, int nnx, int nz1, int nx1)
/*< expand domain of 'a' to 'b':  a, size=nz1*nx1; b, size=nnz*nnx;  >*/
{
    int iz, ix;
    for(ix = 0; ix < nx1; ix++)
        for (iz = 0; iz < nz1; iz++)
        {
            b[(npml+ix)*nnz+(npml+iz)] = a[ix*nz1+iz];
        }

    for(ix = 0; ix < nnx; ix++)
    {
        for (iz = 0; iz < npml; iz++)       b[ix*nnz+iz] = b[ix*nnz+npml];//top
        for (iz = nz1+npml; iz < nnz; iz++) b[ix*nnz+iz] = b[ix*nnz+npml+nz1-1];//bottom
    }

    for(iz = 0; iz < nnz; iz++)
    {
        for(ix = 0; ix < npml; ix++)        b[ix*nnz+iz] = b[npml*nnz+iz];//left
        for(ix = npml+nx1; ix < nnx; ix++)  b[ix*nnz+iz] = b[(npml+nx1-1)*nnz+iz];//right
    }
}// End of expand function

void host_alloc()
{
    // source wavelet
    h_wlt   = new double[nts];

    h_vel   = new float[N];
    h_rho   = new float[N];

    // staggered grid		
    bue     = new float[N];
    bu1     = new float[N];
    bu2     = new float[N];
    kappa   = new double[N];
   		
    // sponge
    spg     = new float[npml + 1];
    spg1    = new float[npml + 1];

    damp2a  = new float[nnx];
    damp2a1 = new float[nnx];
    damp2b  = new float[nnx];
    damp2b1 = new float[nnx];

    damp1a  = new float[nnz];
    damp1a1 = new float[nnz];
    damp1b  = new float[nnz];
    damp1b1 = new float[nnz];

    kappaw2 = new double[N];
    kappaw1 = new double[N];
    buw2    = new double[N];
    buw1    = new double[N];
}// End of host_alloc function

void host_free()
{
    // source wavelet
    delete[] h_wlt;
    delete[] h_vel;
    delete[] h_rho;
    
    // staggered grid
    delete[] bue;
    delete[] bu2;
    delete[] bu1;
    delete[] kappa;

    // sponge
    delete[] spg;
    delete[] spg1;

    delete[] damp2a;
    delete[] damp2b;
    delete[] damp1a;
    delete[] damp1b;

    delete[] damp2a1;
    delete[] damp2b1;
    delete[] damp1a1;
    delete[] damp1b1;

    delete[] kappaw2;
    delete[] kappaw1;
    delete[] buw2;
    delete[] buw1;
}// End of host_free function

auto async_exception_handler = [](sycl::exception_list elist) {
    for (auto& e : elist) {
        try{ std::rethrow_exception(e); }
        catch (sycl::exception& e) {
            std::cout << "ASYNC sycl exception!!\n";
            std::cout << e.what() << std::endl;
            std::terminate();
        }
    }
};

void device_alloc(sycl::queue& qsm)
{
    // source wavelet
    d_wlt     = sycl::malloc_device<double>(nts, qsm);

    // sponge
    d_damp2a  = sycl::malloc_device<float>(nnx,  qsm);
    d_damp2b  = sycl::malloc_device<float>(nnx,  qsm);
    d_damp1a  = sycl::malloc_device<float>(nnz,  qsm);
    d_damp1b  = sycl::malloc_device<float>(nnz,  qsm);

    // wavefields
    d_p       = sycl::malloc_device<double>(N,   qsm);
    d_px      = sycl::malloc_device<double>(N,   qsm);
    d_pz      = sycl::malloc_device<double>(N,   qsm);
    d_vx      = sycl::malloc_device<double>(N,   qsm);
    d_vz      = sycl::malloc_device<double>(N,   qsm);

    // diffrential operators
    d_der2    = sycl::malloc_device<double>(N,   qsm);
    d_der1    = sycl::malloc_device<double>(N,   qsm);

    d_kappa   = sycl::malloc_device<double>(N,   qsm);
    d_kappaw2 = sycl::malloc_device<double>(N,   qsm);
    d_kappaw1 = sycl::malloc_device<double>(N,   qsm);
    d_buw2    = sycl::malloc_device<double>(N,   qsm);
    d_buw1    = sycl::malloc_device<double>(N,   qsm);
}// End of device_alloc function

void device_free(sycl::queue& qsm)
{
    // source wavelet
    sycl::free(d_wlt,     qsm);

    // sponge
    sycl::free(d_damp2a,  qsm);
    sycl::free(d_damp2b,  qsm);
    sycl::free(d_damp1a,  qsm);
    sycl::free(d_damp1b,  qsm);

    // wavefields
    sycl::free(d_p,       qsm);
    sycl::free(d_px,      qsm);
    sycl::free(d_pz,      qsm);
    sycl::free(d_vx,      qsm);
    sycl::free(d_vz,      qsm);

    // diffrential operators
    sycl::free(d_der2,    qsm);
    sycl::free(d_der1,    qsm);

    sycl::free(d_kappa,   qsm);
    sycl::free(d_kappaw2, qsm);
    sycl::free(d_kappaw1, qsm);
    sycl::free(d_buw2,    qsm);
    sycl::free(d_buw1,    qsm);
}// End of device_free function

void FDOperator_4(int it, int isource, sycl::queue& qsm)
{
    if((it + 1) % 1000 == 0)
        cout << "\n it: " << it + 1;

    if(NJ == 4) // sycl events are used for synchronization. Please follow e\d{2} values.
    {
        // qsm.memset(d_der1, 0, N * sizeof(double));
        // qsm.memset(d_der2, 0, N * sizeof(double));
	    auto e01 = qsm.fill(d_der1, (double)0.0, N);
        auto e02 = qsm.fill(d_der2, (double)0.0, N);

        auto e03 = qsm.submit([&](sycl::handler &cgh) {
            sycl::local_accessor<double, 2> s_vz_acc_ct1(sycl::range<2>(19 /*Block_size1+3*/, 16 /*16*/), cgh);

            auto d_vz_ct0 = d_vz;
            auto d_der1_ct1 = d_der1;
            auto nnz_ct2 = nnz;
            auto nnx_ct3 = nnx;

            cgh.depends_on(e01);
            cgh.parallel_for(
                sycl::nd_range<3>(dimg0 * dimb0, dimb0),
                [=](sycl::nd_item<3> item_ct1) { // input: d_vz, nnz, nnx, output: d_der1
                    gpu_fdoperator410(d_vz_ct0, d_der1_ct1, nnz_ct2, nnx_ct3, item_ct1, std::move(s_vz_acc_ct1));
                }
            );
        });

        auto e04 = qsm.submit([&](sycl::handler &cgh) {
            sycl::local_accessor<double, 2> s_vx_acc_ct1(sycl::range<2>(16 /*16*/, 19 /*Block_size2+3*/), cgh);

            auto d_vx_ct0 = d_vx;
            auto d_der2_ct1 = d_der2;
            auto nnz_ct2 = nnz;
            auto nnx_ct3 = nnx;

            cgh.depends_on(e02);
            cgh.parallel_for(
                sycl::nd_range<3>(dimg0 * dimb0, dimb0),
                [=](sycl::nd_item<3> item_ct1) { // input: d_vx, nnz, nnx, output: d_der2
                    gpu_fdoperator420(d_vx_ct0, d_der2_ct1, nnz_ct2, nnx_ct3, item_ct1, std::move(s_vx_acc_ct1));
                }
            );
        });

        auto e05 = qsm.submit([&](sycl::handler &cgh) {
            auto d_px_ct0 = d_px;
            auto d_pz_ct1 = d_pz;
            auto d_kappaw2_ct2 = d_kappaw2;
            auto d_kappaw1_ct3 = d_kappaw1;
            auto d_der2_ct4 = d_der2;
            auto d_der1_ct5 = d_der1;
            auto d_damp2a_ct6 = d_damp2a;
            auto d_damp1a_ct7 = d_damp1a;
            auto nnz_ct8 = nnz;
            auto nnx_ct9 = nnx;

            cgh.depends_on({e03, e04});
            cgh.parallel_for(
                sycl::nd_range<3>(dimg0 * dimb0, dimb0),
                [=](sycl::nd_item<3> item_ct1) {  // calculates d_px and d_pz
                    gpu_pml_p(d_px_ct0, d_pz_ct1, d_kappaw2_ct2, d_kappaw1_ct3, d_der2_ct4, d_der1_ct5, d_damp2a_ct6, d_damp1a_ct7, nnz_ct8, nnx_ct9, item_ct1);
                });
        });
	    // qsm.wait();

        // increment source
        if(it <= nts)
        {
            auto e06 = qsm.submit([&](sycl::handler &cgh) {
                auto d_px_ct0 = d_px;
                auto d_wlt_ct1 = d_wlt;
                auto d_kappa_ct2 = d_kappa;
                auto fd_dt_ct3 = fd_dt;
                auto dx_ct4 = dx;
                auto dz_ct5 = dz;

                cgh.depends_on(e05);
                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
                    [=](sycl::nd_item<3> item_ct1) { // increments d_px
                        gpu_incr_p(d_px_ct0, d_wlt_ct1, d_kappa_ct2, fd_dt_ct3, dx_ct4, dz_ct5, isource, it);
                    }
                );
            });
            e06.wait();
        }

        sycl::range<2> threads(Block_size2, Block_size1);
        sycl::range<2> blocks(nnx / Block_size2, nnz / Block_size1);
        auto e07 = qsm.submit([&](sycl::handler &cgh) {
            auto d_p_ct0  = d_p;
            auto d_px_ct1 = d_px;
            auto d_pz_ct2 = d_pz;
            auto nnz_ct3  = nnz;
            auto nnx_ct4  = nnx;

            cgh.depends_on(e05);
            cgh.parallel_for(
                sycl::nd_range<2>(blocks * threads, threads),
                [=](sycl::nd_item<2> item) { // sets d_p = d_px + d_pz
                    gpu_compute_p(d_p_ct0, d_px_ct1, d_pz_ct2, nnz_ct3, nnx_ct4, item);
                }
            );
        });

        // qsm.memset(d_der1, 0, N * sizeof(double));
        // qsm.memset(d_der2, 0, N * sizeof(double));
        auto e11 = qsm.fill(d_der1, (double)0.0, N, e05);
        auto e12 = qsm.fill(d_der2, (double)0.0, N, e05);

        auto e13 = qsm.submit([&](sycl::handler &cgh) {
            sycl::local_accessor<double, 2> s_p_acc_ct1(sycl::range<2>(19 /*Block_size1+3*/, 16 /*16*/), cgh);

            auto d_p_ct0 = d_p;
            auto d_der1_ct1 = d_der1;
            auto nnz_ct2 = nnz;
            auto nnx_ct3 = nnx;

            cgh.depends_on({e07, e11});
            cgh.parallel_for(
                sycl::nd_range<3>(dimg0 * dimb0, dimb0),
                [=](sycl::nd_item<3> item_ct1) { // input: d_p, nnz, nnx, output: d_der1
                    gpu_fdoperator411(d_p_ct0, d_der1_ct1, nnz_ct2, nnx_ct3, item_ct1, std::move(s_p_acc_ct1));
                }
            );
        });

        auto e14 = qsm.submit([&](sycl::handler &cgh) {
            sycl::local_accessor<double, 2> s_p_acc_ct1(sycl::range<2>(16 /*16*/, 19 /*Block_size2+3*/), cgh);

            auto d_p_ct0 = d_p;
            auto d_der2_ct1 = d_der2;
            auto nnz_ct2 = nnz;
            auto nnx_ct3 = nnx;

            cgh.depends_on({e07, e12});
            cgh.parallel_for(
                sycl::nd_range<3>(dimg0 * dimb0, dimb0),
                [=](sycl::nd_item<3> item_ct1) { // input: d_p, nnz, nnx, output: d_der2
                    gpu_fdoperator421(d_p_ct0, d_der2_ct1, nnz_ct2, nnx_ct3, item_ct1, std::move(s_p_acc_ct1));
                }
            );
        });

        auto e15 = qsm.submit([&](sycl::handler &cgh) {
            auto d_vx_ct0 = d_vx;
            auto d_vz_ct1 = d_vz;
            auto d_buw2_ct2 = d_buw2;
            auto d_buw1_ct3 = d_buw1;
            auto d_der2_ct4 = d_der2;
            auto d_der1_ct5 = d_der1;
            auto d_damp2b_ct6 = d_damp2b;
            auto d_damp1b_ct7 = d_damp1b;
            auto nnz_ct8 = nnz;
            auto nnx_ct9 = nnx;

            cgh.depends_on({e13, e14});
            cgh.parallel_for(
                sycl::nd_range<3>(dimg0 * dimb0, dimb0),
                [=](sycl::nd_item<3> item_ct1) { // calculates d_vx and d_vz
                    gpu_pml_v(d_vx_ct0, d_vz_ct1, d_buw2_ct2, d_buw1_ct3, d_der2_ct4, d_der1_ct5, d_damp2b_ct6, d_damp1b_ct7, nnz_ct8, nnx_ct9, item_ct1);
                }
            );
        });
	    e15.wait();
    }// NJ=4
}//End of FDOperator function

void modelling_module(int my_nsrc, geo2d_t* mygeo2d_sp)
{
try {
    int rank = MPI::COMM_WORLD.Get_rank();

    int ix, iz, id, is, ig, kt, istart, dt_factor, indx;

    // variable init
    NJ        = job_sp->fdop;
    ns        = my_nsrc;
    nx1       = mod_sp->nx;
    nz1       = mod_sp->nz;

    dx        = mod_sp->dx;
    dz        = mod_sp->dz;
    _dx       = 1.0f / dx;
    _dz       = 1.0f / dz;

    rec_len   = wave_sp->rec_len;
    fd_dt     = wave_sp->fd_dt_sec;
    real_dt   = wave_sp->real_dt_sec;
    fm        = wave_sp->dom_freq;
    fd_nt     = (int)ceil(rec_len / fd_dt);
    real_nt   = (int)ceil(rec_len / real_dt);

    dtdx      = (double) (fd_dt * _dx);
    dtdz      = (double) (fd_dt * _dz);

    dt_factor = (int) (real_dt / fd_dt);

    double td = sqrt(6.0f) / (PI * fm);
    nts       = (int) (5.0f * td / fd_dt) + 1;
    if (nts > fd_nt)
    {
        nts   = fd_nt;
    }

    nx        = (int)((nx1 + Block_size2 - 1) / Block_size2) * Block_size2;
    nz        = (int)((nz1 + Block_size1 - 1) / Block_size1) * Block_size1;
    nnz       = 2 * npml + nz;
    nnx       = 2 * npml + nx;

    N         = nnz * nnx;

    dimb0     = sycl::range<3>(1, Block_size2, Block_size1);
    dimg0     = sycl::range<3>(1, nnx / Block_size2, nnz / Block_size1);

#ifdef DEBUG_TIME
    cout << "\n Rank: " << rank 
         << "\t nnx: "  << nnx 
         << "   nnz: "  << nnz 
         << "   nts: "  << nts;
    cout << endl;
    cout << "Rank: " << rank << "  " << "NJ        : " << NJ        << endl;
    cout << "Rank: " << rank << "  " << "ns        : " << ns        << endl;
    cout << "Rank: " << rank << "  " << "nx1       : " << nx1       << endl;
    cout << "Rank: " << rank << "  " << "nz1       : " << nz1       << endl;
    cout << "Rank: " << rank << "  " << "dx        : " << dx        << endl;
    cout << "Rank: " << rank << "  " << "dz        : " << dz        << endl;
    cout << "Rank: " << rank << "  " << "_dx       : " << _dx       << endl;
    cout << "Rank: " << rank << "  " << "_dz       : " << _dz       << endl;
    cout << "Rank: " << rank << "  " << "rec_len   : " << rec_len   << endl;
    cout << "Rank: " << rank << "  " << "fd_dt     : " << fd_dt     << endl;
    cout << "Rank: " << rank << "  " << "real_dt   : " << real_dt   << endl;
    cout << "Rank: " << rank << "  " << "fm        : " << fm        << endl;
    cout << "Rank: " << rank << "  " << "fd_nt     : " << fd_nt     << endl;
    cout << "Rank: " << rank << "  " << "real_nt   : " << real_nt   << endl;
    cout << "Rank: " << rank << "  " << "dtdx      : " << dtdx      << endl;
    cout << "Rank: " << rank << "  " << "dtdz      : " << dtdz      << endl;
    cout << "Rank: " << rank << "  " << "dt_factor : " << dt_factor << endl;
    cout << "Rank: " << rank << "  " << "td        : " << td        << endl;
    cout << "Rank: " << rank << "  " << "nts       : " << nts       << endl;
    cout << "Rank: " << rank << "  " << "nx        : " << nx        << endl;
    cout << "Rank: " << rank << "  " << "nz        : " << nz        << endl;
    cout << "Rank: " << rank << "  " << "nnx       : " << nnx       << endl;
    cout << "Rank: " << rank << "  " << "nnz       : " << nnz       << endl;
    cout << "Rank: " << rank << "  " << "N         : " << N         << endl;
    // cout << "Rank: " << rank << "  " << "dimb0.x   : " << dimb0.x   << endl;
    // cout << "Rank: " << rank << "  " << "dimb0.y   : " << dimb0.y   << endl;
    // cout << "Rank: " << rank << "  " << "dimg0.x   : " << dimg0.x   << endl;
    // cout << "Rank: " << rank << "  " << "dimg0.y   : " << dimg0.y   << endl;
#endif

    // allocate host memory
    host_alloc();

    v0 = new float[nx1*nz1];
    // expand velocity model
    memset(v0,    0, nz1*nx1 * sizeof(float));
    memset(h_vel, 0, N       * sizeof(float));
    for(ix = 0; ix < nx1; ix++)
        for(iz = 0; iz < nz1; iz++)
            v0[ix*nz1+iz] = mod_sp->vel2d[ix][iz];  // converts 2d data to 1d data (number of data points remains same)

    expand(h_vel, v0, npml, nnz, nnx, nz1, nx1);    // h_vel is v0 with pml boundary values added

    // expand density model
    memset(v0,    0, nz1*nx1 * sizeof(float));
    memset(h_rho, 0, N       * sizeof(float));
    for(ix = 0; ix < nx1; ix++)
        for(iz = 0; iz < nz1; iz++)
            v0[ix*nz1+iz] = mod_sp->rho2d[ix][iz];  // converts 2d data to 1d data (number of data points remains same)

    expand(h_rho, v0, npml, nnz, nnx, nz1, nx1);    // h_rho is v0 with pml boundary values added

    // build staggered grids
    memset(bue,   0, N * sizeof(float));
    memset(bu2,   0, N * sizeof(float));
    memset(bu1,   0, N * sizeof(float));
    memset(kappa, 0, N * sizeof(double));
    cpu_stagger(h_vel, h_rho, bue, bu2, bu1, kappa, nnz, nnx);  // calculates bue (as an intermediate value), bu2, bu1, kappa

    // compute source wavelet
    memset(h_wlt, 0, nts * sizeof(double));
    cpu_ricker(fm, fd_dt, h_wlt, nts);              // calculates h_wlt

    // build sponges
    memset(spg,     0, (npml+1) * sizeof(float));
    memset(spg1,    0, (npml+1) * sizeof(float));
    memset(damp2a,  0, nnx      * sizeof(float));
    memset(damp2a1, 0, nnx      * sizeof(float));
    memset(damp2b,  0, nnx      * sizeof(float));
    memset(damp2b1, 0, nnx      * sizeof(float));
    memset(damp1a,  0, nnz      * sizeof(float));
    memset(damp1a1, 0, nnz      * sizeof(float));
    memset(damp1b,  0, nnz      * sizeof(float));
    memset(damp1b1, 0, nnz      * sizeof(float));

    float xxfac = 0.05f;
    cpu_pml_coefficient_a(xxfac, damp2a, damp2a1, spg, spg1, npml, nx1);
    cpu_pml_coefficient_b(xxfac, damp2b, damp2b1, spg, spg1, npml, nx1);
    cpu_pml_coefficient_a(xxfac, damp1a, damp1a1, spg, spg1, npml, nz1);
    cpu_pml_coefficient_b(xxfac, damp1b, damp1b1, spg, spg1, npml, nz1);

    memset(kappaw2, 0, N * sizeof(double));
    memset(kappaw1, 0, N * sizeof(double));
    memset(buw2,    0, N * sizeof(double));
    memset(buw1,    0, N * sizeof(double));

    for(ix = 0; ix < nnx; ix++)
    {
       for(iz = 0; iz < nnz; iz++)
       {
            id = iz + ix * nnz;

            kappaw2[id] = dtdx * kappa[id] * damp2a1[ix];
            kappaw1[id] = dtdz * kappa[id] * damp1a1[iz];
            buw2[id]    = dtdx * bu2[id]   * damp2b1[ix];
            buw1[id]    = dtdz * bu1[id]   * damp1b1[iz];
        }
    }

    double duration = 0.0;

#ifdef DEBUG_TIME
    auto time11 = std::chrono::steady_clock::now();
#endif

    // Set Device and allocate memory
    sycl::device dsm = sycl::device(sycl::default_selector_v);


    sycl::queue qsm{ dsm, async_exception_handler };

    device_alloc(qsm);

#ifdef DEBUG_TIME
    auto time12 = std::chrono::steady_clock::now();
    double duration1 = std::chrono::duration<double, std::micro>(time12 - time11).count();
    duration += duration1;
    std::cout << "\ndevice alloc : duration1: " << duration1 << " us\n\n";

    auto time21 = std::chrono::steady_clock::now();
#endif

    qsm.memcpy(d_wlt,     h_wlt,   nts * sizeof(double)); // these memcpy calls are independent

    qsm.memcpy(d_damp2a,  damp2a,  nnx * sizeof(float));
    qsm.memcpy(d_damp2b,  damp2b,  nnx * sizeof(float));
    qsm.memcpy(d_damp1a,  damp1a,  nnz * sizeof(float));
    qsm.memcpy(d_damp1b,  damp1b,  nnz * sizeof(float));

    qsm.memcpy(d_kappa,   kappa,   N   * sizeof(double));
    qsm.memcpy(d_kappaw2, kappaw2, N   * sizeof(double));
    qsm.memcpy(d_kappaw1, kappaw1, N   * sizeof(double));
    qsm.memcpy(d_buw2,    buw2,    N   * sizeof(double));
    qsm.memcpy(d_buw1,    buw1,    N   * sizeof(double)).wait(); // this wait is needed as host data are freed next

#ifdef DEBUG_TIME
    auto time22 = std::chrono::steady_clock::now();
    double duration2 = std::chrono::duration<double, std::micro>(time22 - time21).count();
    duration += duration2;
    std::cout << "\nmemcopy : duration2: " << duration2 << " us\n\n";
#endif

    // free host memory
    host_free();

    // set source positions
    sz_pos = new int[ns];
    sx_pos = new int[ns];
    for(is = 0; is < ns; is++)
    {
        sz_pos[is] = (int)(mygeo2d_sp->src2d_sp[is].z * _dz);
        sx_pos[is] = (int)(mygeo2d_sp->src2d_sp[is].x * _dx);
    }
    h_Sxz = new int[ns];
    cpu_set_sg(h_Sxz, sx_pos, sz_pos, ns, npml, nnz);   // transform from ns number pair(s) to ns single number(s)

    // checking job status - New/Restart
    istart = 0;
    if(strcmp(job_sp->jbtype, "New") == 0)
    {
        istart = 0;
    }
    else if(strcmp(job_sp->jbtype, "Restart") == 0)
    {
        char rnk[5];
        char log_file[1024];
        FILE *fp_log;

        sprintf(rnk, "%d", rank);
        strcpy(log_file, job_sp->tmppath);
        strcat(log_file, job_sp->jobname);
        strcat(log_file, "checkpoint");
        strcat(log_file, "_rank");
        strcat(log_file, rnk);
        strcat(log_file, ".txt");

        fp_log = fopen(log_file, "r");
        if(fp_log == NULL)
        {
            cerr << "\n Rank: " << rank << "   Error !!! in openining log file: " << log_file;
            cerr << "\n Please make sure that same job is executed before with same resource \
                      configuration";
            MPI::COMM_WORLD.Abort(-10);
            return;
        }
        fscanf(fp_log, "%d", &istart);
        fclose(fp_log);
    }
    else
    {
        cerr << "\n Error !!! Rank: " << rank << "   Unknow job status, Please make respective changes in job card";
        MPI::COMM_WORLD.Abort(-11);
    }

    cout << "\n Rank: " << rank << "   Started Modelling.......";

    // shot loop (for a new job, istart starts at 0)
    for(is = istart; is < ns; is++) // for current input, this loop run only once!
    {
        // set geaophone positions
        ng     = mygeo2d_sp->nrec[is];
        gz_pos = new int[ng];
        gx_pos = new int[ng];
        for(ig = 0; ig < ng; ig++)
        {
            gz_pos[ig] = (int)(mygeo2d_sp->rec2d_sp[is][ig].z * _dz);
            gx_pos[ig] = (int)(mygeo2d_sp->rec2d_sp[is][ig].x * _dx);
        }
        h_Gxz = new int[ng];
        cpu_set_sg(h_Gxz, gx_pos, gz_pos, ng, npml, nnz);   // transform from ng number pairs to ng single numbers

        h_dobs = new float[ng * real_nt];   // ng receivers and real_nt time points (for device to host copy)
        memset(h_dobs, 0,  ng * real_nt * sizeof(float));

#ifdef DEBUG_TIME
        auto time31 = std::chrono::steady_clock::now();
#endif

        d_Gxz = sycl::malloc_device<int>(ng, qsm);
        qsm.memcpy(d_Gxz, h_Gxz, ng * sizeof(int)); // these memcpy / memset calls are independent

        d_temp = sycl::malloc_device<float>(ng * real_nt, qsm);
        qsm.memset(d_temp, 0, ng * real_nt * sizeof(float));

        d_dobs = sycl::malloc_device<float>(ng * real_nt, qsm);     // transpose of d_temp
        qsm.memset(d_dobs, 0, ng * real_nt * sizeof(float));

        qsm.memset(d_p,  0, N * sizeof(double));
        qsm.memset(d_px, 0, N * sizeof(double));
        qsm.memset(d_pz, 0, N * sizeof(double));
        qsm.memset(d_vx, 0, N * sizeof(double));
        qsm.memset(d_vz, 0, N * sizeof(double)).wait(); // synchronize before FDOperator_4()

        for(kt = 0; kt < fd_nt; kt++)   // loop 10000 times
        {
            FDOperator_4(kt, h_Sxz[is], qsm);

            // storing pressure seismograms [real_nt values for each kt]
            if( (kt + 1) % dt_factor == 0 )
            {
                indx = ((kt + 1) / dt_factor) - 1;
                auto e31 = qsm.submit([&](sycl::handler &cgh) {
                    auto d_p_ct0 = d_p;
                    auto d_temp_indx_ng_ct1 = &d_temp[indx * ng];
                    auto d_Gxz_ct2 = d_Gxz;
                    auto ng_ct3 = ng;

                    cgh.parallel_for(
                        sycl::nd_range<1>(((ng + 255) / 256) * 256, 256),
                        [=](sycl::nd_item<1> item) {
                            gpu_record(d_p_ct0, d_temp_indx_ng_ct1, d_Gxz_ct2, ng_ct3, item);
                        });
                });
                e31.wait();
            }
        } // End of NT loop

        // synchronized at this point since you can reach here either from FDOperator_4() or after gpu_record() both of which wait()
        auto e32 = qsm.submit([&](sycl::handler &cgh) {
            auto d_temp_ct0 = d_temp;   // input
            auto d_dobs_ct1 = d_dobs;   // output
            auto ng_ct2 = ng;
            auto real_nt_ct3 = real_nt;

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, (real_nt + 15) / 16, (ng + 15) / 16) * sycl::range<3>(1, 16, 16), sycl::range<3>(1, 16, 16)),
                [=](sycl::nd_item<3> item_ct1) {
                    gpu_transpose(d_temp_ct0, d_dobs_ct1, ng_ct2, real_nt_ct3, item_ct1);
                });
        });

        qsm.memcpy(h_dobs, d_dobs, ng * real_nt * sizeof(float), std::move(e32)).wait(); // synchronize

#ifdef DEBUG_TIME
        auto time32 = std::chrono::steady_clock::now();
        double duration3 = std::chrono::duration<double, std::micro>(time32 - time31).count();
        duration += duration3;
        std::cout << "\niter: " << is << ", memcopy+memset+FDOperator_4+gpu_record+gpu_transpose+memcpy : duration3: " << duration3 << " us\n\n";
#endif

        // write seismogram
        char sx_ch[15];
        char seis_file[1024];
        FILE* fp_seis;

        sprintf(sx_ch, "%.2f", sx_pos[is] * dx);
        strcpy(seis_file, job_sp->tmppath);
        strcat(seis_file, job_sp->jobname);
        strcat(seis_file, "sx");
        strcat(seis_file, sx_ch);
        strcat(seis_file, "_seismogram.bin");

        fp_seis = fopen(seis_file, "w");
        if(fp_seis == NULL)
        {
            cerr << "\n Error!!! Unable to open output file : " << seis_file << endl;
            MPI::COMM_WORLD.Abort(-27);
            return;
        }
        fwrite(h_dobs, sizeof(float), ng * real_nt, fp_seis);
        fclose(fp_seis);

        delete[] gz_pos;
        delete[] gx_pos;
        delete[] h_Gxz;
        delete[] h_dobs;
        sycl::free(d_Gxz,  qsm);
        sycl::free(d_dobs, qsm);
        sycl::free(d_temp, qsm);

        cout << "\n Rank: " << rank << "   Shot remaining: " << ns - (is + 1) << "\n";
    }// End of shot loop

    device_free(qsm);

    // std::cout << "\nTotal device time for whole calculation: " << duration / 1e6 << " s\n\n";
} catch (sycl::exception& e) {
    std::cout << "Caught SYNC sycl exception: " << e.what() << std::endl;
} catch (std::exception& e) {
    std::cout << "Caught std exception: " << e.what() << std::endl;
} catch (...) {
    std::cout << "Caught unknown exception\n";
}
}//End of modelling_module function
