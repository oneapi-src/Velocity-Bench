/*
MIT License

Copyright (c) 2015 University of West Bohemia

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

/*
MIT License

Modifications Copyright (C) 2023 Intel Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

SPDX-License-Identifier: MIT License
*/

#define NUM_ITERATIONS 100

#ifndef KERNEL_USE_PROFILE
#define KERNEL_USE_PROFILE 0
#endif

#include <sycl/sycl.hpp>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>
#include <ctype.h>
#include <float.h>
#include <algorithm>
#include <math.h>
#include "mex.h"
#include "cuSVMutil.h"
#include <vector>
#include <chrono>
#include "CommandLineParser.h"
//#include "SYCL.h"

#ifdef USE_CUBLAS
#include <sycl/backend/cuda.hpp>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#elif USE_HIPBLAS
//#include "hip/hip_runtime.h" 
#include "hipblas.h"
#else
#include "oneapi/mkl/blas.hpp"
#endif

float C;
float taumin;
float kernelwidth;


#ifdef USE_CUBLAS
#define CHECK_ERROR(FUNC) checkCudaErrorMsg(FUNC, " " #FUNC)

void inline checkCudaErrorMsg(cublasStatus_t status, const char *msg) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << "ERROR CUBLAS:" << msg << " - " << status << std::endl;
    exit(EXIT_FAILURE);
  }
}

void inline checkCudaErrorMsg(cudaError status, const char *msg) {
  if (status != cudaSuccess) {
    std::cout << "ERROR CUDA: " << msg << " - " << status << std::endl;
    exit(EXIT_FAILURE);
  }
}

void inline checkCudaErrorMsg(CUresult status, const char *msg) {
  if (status != CUDA_SUCCESS) {
    std::cout << "ERROR CUDA: " << msg << " - " << status << std::endl;
    exit(EXIT_FAILURE);
  }
}

#endif



template <unsigned int blockSize>
void FindBJ(float *d_F, float* d_y,float* d_alpha,float* d_KernelCol,float *g_odata,int* g_index,float BIValue, unsigned int n,
            sycl::nd_item<3> item_ct1, float C, float taumin, float *sdata,
            int *ind)
{

    unsigned int tid = item_ct1.get_local_id(2);
    unsigned int i = item_ct1.get_group(2) * (blockSize * 2) + tid;
    unsigned int gridSize = blockSize * 2 * item_ct1.get_group_range(2);
        sdata[tid]=-FLT_MAX;
	ind[tid]=0;

	float temp;
	float globaltemp;

	float LocalCloseY;
	float LocalFarY;
	float maxtemp;
	float denomclose;
	float denomfar=1.f;


	while (i < n) 
	{ 
		LocalCloseY=d_y[i];
		LocalFarY=(i+blockSize)<n ? d_y[i+blockSize]:0.f;
		denomclose=(2.f-2.f*d_KernelCol[i]);
		if(i+blockSize<n){denomfar=(2.f-2.f*d_KernelCol[i+blockSize]);}


		denomclose=denomclose<taumin?taumin:denomclose;
		denomfar=denomfar<taumin?taumin:denomfar;

        maxtemp = sycl::fmax(
        globaltemp =
          (LocalCloseY * d_alpha[i]) > (LocalCloseY == 1 ? 0 : -C)
              ? sycl::pow(BIValue + LocalCloseY * d_F[i], 2.f) / denomclose
              : -FLT_MAX,
        i + blockSize < n
          ? ((LocalFarY * d_alpha[i + blockSize]) > (LocalFarY == 1 ? 0 : -C)
                 ? sycl::pow(BIValue + LocalFarY * d_F[i + blockSize], 2.f) /
                       denomfar
                 : -FLT_MAX)
          : -FLT_MAX);

        sdata[tid] = sycl::fmax(temp = sdata[tid], maxtemp);

        if (sdata[tid]!=temp)
		{
			sdata[tid]== globaltemp ? ind[tid]=i : ind[tid]=i+blockSize;
		}

		i += gridSize; 
	}

    item_ct1.barrier();

    if (tid < 128) {
        if (sdata[tid] < sdata[tid + 128]) {
            ind[tid] = ind[tid + 128]; sdata[tid] = sdata[tid + 128];
        }
    } 
 
    item_ct1.barrier();

    if (tid < 64) {
        if (sdata[tid] < sdata[tid + 64]) {
        ind[tid] = ind[tid + 64]; sdata[tid] = sdata[tid + 64];
        }
    } 
 
    item_ct1.barrier();

  
    if ((tid < 32) && (sdata[tid] < sdata[tid + 32])) {
        ind[tid] = ind[tid + 32]; sdata[tid] = sdata[tid + 32];
    } 
        
    item_ct1.barrier();
  
    if ((tid < 32) && (sdata[tid] < sdata[tid + 16])) {
        ind[tid] = ind[tid + 16]; sdata[tid] = sdata[tid + 16];
    } 
        
    item_ct1.barrier();
  
    if ((tid < 32) && (sdata[tid] < sdata[tid + 8])) {
        ind[tid] = ind[tid + 8]; 
        sdata[tid] = sdata[tid + 8];
    } 
        
    item_ct1.barrier();
  
    if ((tid < 32) && (sdata[tid] < sdata[tid + 4])) {
        ind[tid] = ind[tid + 4]; 
        sdata[tid] = sdata[tid + 4];
    } 
        
    item_ct1.barrier();
  
    if ((tid < 32) && (sdata[tid] < sdata[tid + 2])) {
        ind[tid] = ind[tid + 2]; sdata[tid] = sdata[tid + 2];
    } 
        
        
    item_ct1.barrier();
        
    if ((tid < 32) && (sdata[tid] < sdata[tid + 1]))  {
        ind[tid] = ind[tid + 1]; sdata[tid] = sdata[tid + 1];
    } 
        
    item_ct1.barrier();
        
    if (tid == 0) g_odata[item_ct1.get_group(2)] = sdata[0];
    if (tid == 0) g_index[item_ct1.get_group(2)] = ind[0];
}

//float C problems
template <unsigned int blockSize>
void FindBI(float *d_F, float* d_y,float* d_alpha,float *g_odata,int* g_index,unsigned int n,
            sycl::nd_item<3> item_ct1, float C, float *sdata, int *ind)
{

 unsigned int tid = item_ct1.get_local_id(2);
 unsigned int i = item_ct1.get_group(2) * (blockSize * 2) + tid;
 unsigned int gridSize = blockSize * 2 * item_ct1.get_group_range(2);
 sdata[tid]=-FLT_MAX;
 ind[tid]=0;


   
	float temp;
	float globaltemp;

	float LocalCloseY;
	float LocalFarY;
	float maxtemp;


     
	while (i < n) 
    { 
		LocalCloseY=d_y[i];
		LocalFarY=(i+blockSize)<n ? d_y[i+blockSize]:0;

        maxtemp = sycl::fmax(
            globaltemp = (LocalCloseY * d_alpha[i]) < (LocalCloseY == 1 ? C : 0)
                       ? -(d_F[i] * LocalCloseY)
                       : -FLT_MAX,
            i + blockSize < n
            ? ((LocalFarY * d_alpha[i + blockSize]) < (LocalFarY == 1 ? C : 0)
                 ? -(d_F[i + blockSize] * LocalFarY)
                 : -FLT_MAX) : -FLT_MAX);

        sdata[tid] = sycl::fmax(temp = sdata[tid], maxtemp);

                if (sdata[tid]!=temp)
		        {
			        sdata[tid]== globaltemp ? ind[tid]=i : ind[tid]=i+blockSize;
		        }

		i += gridSize; 
	}

    item_ct1.barrier();
 
    
    if (tid < 128) {
        if (sdata[tid] < sdata[tid + 128]) {
            ind[tid] = ind[tid + 128]; 
            sdata[tid] = sdata[tid + 128];
        }
    } 
    item_ct1.barrier();

    
    if (tid < 64) {
        if (sdata[tid] < sdata[tid + 64]) {
            ind[tid] = ind[tid + 64]; 
            sdata[tid] = sdata[tid + 64];
        }
    } 
    
    item_ct1.barrier();

    
    if ((tid < 32) && (sdata[tid] < sdata[tid + 32])) 
	{
        
            ind[tid] = ind[tid + 32]; 
            sdata[tid] = sdata[tid + 32];
    }

    item_ct1.barrier();
  
    if ( (tid < 32) &&  (sdata[tid] < sdata[tid + 16])) {
            ind[tid] = ind[tid + 16]; 
            sdata[tid] = sdata[tid + 16];
    } 
    
    item_ct1.barrier();
  
    if ((tid < 32) && (sdata[tid] < sdata[tid + 8])) {
            ind[tid] = ind[tid + 8]; 
            sdata[tid] = sdata[tid + 8];
    } 
    
    item_ct1.barrier();
  
    if ((tid < 32) && (sdata[tid] < sdata[tid + 4])) {
            ind[tid] = ind[tid + 4]; 
            sdata[tid] = sdata[tid + 4];
    } 
    
    item_ct1.barrier();
  
  
    if ((tid < 32) && (sdata[tid] < sdata[tid + 2])) {
            ind[tid] = ind[tid + 2]; 
            sdata[tid] = sdata[tid + 2];
    } 
    
    item_ct1.barrier();
  
    if ((tid < 32) && (sdata[tid] < sdata[tid + 1])) {
            ind[tid] = ind[tid + 1]; 
            sdata[tid] = sdata[tid + 1];
    } 
    
    item_ct1.barrier();
    
    if (tid == 0) g_odata[item_ct1.get_group(2)] = sdata[0];
    if (tid == 0) g_index[item_ct1.get_group(2)] = ind[0]; 
}


template <unsigned int blockSize>
void FindStoppingJ(float *d_F, float* d_y,float* d_alpha,float *g_odata,unsigned int n,
                   sycl::nd_item<3> item_ct1, float C, float *sdata)
{

 unsigned int tid = item_ct1.get_local_id(2);
 unsigned int i = item_ct1.get_group(2) * (blockSize * 2) + tid;
 unsigned int gridSize = blockSize * 2 * item_ct1.get_group_range(2);
        sdata[tid]=FLT_MAX;


	float LocalCloseY;
	float LocalFarY;


	while (i < n) 
	{ 
		LocalCloseY=d_y[i];
		LocalFarY=(i+blockSize)<n ? d_y[i+blockSize]:0;

        sdata[tid] = sycl::fmin(
        sdata[tid],
        sycl::fmin((LocalCloseY * d_alpha[i]) > (LocalCloseY == 1 ? 0 : -C) ? -(d_F[i] * LocalCloseY)
                     : FLT_MAX,
                 i + blockSize < n ? ((LocalFarY * d_alpha[i + blockSize]) >
                                              (LocalFarY == 1 ? 0 : -C)
                                          ? -(d_F[i + blockSize] * LocalFarY)
                                          : FLT_MAX)
                                   : FLT_MAX));

        i += gridSize; 
	}

    item_ct1.barrier();

    if (tid < 128) {
        sdata[tid] = sycl::fmin(sdata[tid], sdata[tid + 128]);
    } 
 
    item_ct1.barrier();

    if (tid < 64) {
        sdata[tid] = sycl::fmin(sdata[tid], sdata[tid + 64]);
    } 
 
    item_ct1.barrier();

    if (tid < 32) { sdata[tid] = sycl::fmin(sdata[tid], sdata[tid + 32]); }
    item_ct1.barrier();
            
    if (tid < 32) { sdata[tid] = sycl::fmin(sdata[tid], sdata[tid + 16]); }
    item_ct1.barrier();
           
           
    if (tid < 32) {  sdata[tid] = sycl::fmin(sdata[tid], sdata[tid + 8]); }
    item_ct1.barrier();
    
    if (tid < 32) {  sdata[tid] = sycl::fmin(sdata[tid], sdata[tid + 4]); }
    item_ct1.barrier();
    
    if (tid < 32) {sdata[tid] = sycl::fmin(sdata[tid], sdata[tid + 2]); }
    item_ct1.barrier();
    
    
    if (tid < 32) {sdata[tid] = sycl::fmin(sdata[tid], sdata[tid + 1]); }
    item_ct1.barrier();
 

    if (tid == 0) {
        g_odata[item_ct1.get_group(2)] = sdata[0];
    }
}




void UpdateF(float * F,float *KernelColI,float* KernelColJ, float* d_y,float deltaalphai,float deltaalphaj,float yi,float yj,int n,
             sycl::nd_item<3> item_ct1)
{

	int totalThreads,ctaStart,tid;
 totalThreads = item_ct1.get_group_range(2) * item_ct1.get_local_range().get(2);
 ctaStart = item_ct1.get_local_range().get(2) * item_ct1.get_group(2);
 tid = item_ct1.get_local_id(2);
        int i;

	for (i = ctaStart + tid; i < n; i += totalThreads) 
	{  
		F[i] = F[i] + yi*d_y[i]*deltaalphai*KernelColI[i]+yj*d_y[i]*deltaalphaj*KernelColJ[i];
	}


}

void RBFFinish(float *KernelCol, const float * KernelDotProd,const float* DotProd,const float* DotProdRow,const int n,
               sycl::nd_item<3> item_ct1, float kernelwidth)
{

	int totalThreads,ctaStart,tid;
    totalThreads = item_ct1.get_group_range(2) * item_ct1.get_local_range().get(2);
    ctaStart = item_ct1.get_local_range().get(2) * item_ct1.get_group(2);
    tid = item_ct1.get_local_id(2);
    int i;
	float temp;

	for (i = ctaStart + tid; i < n; i += totalThreads) 
	{
        KernelCol[i] = sycl::exp(kernelwidth * (DotProd[i] + *DotProdRow - KernelDotProd[i] * 2.f));
    }


}


void RBFKernel(float *d_KernelJ,const int BJIndex,const float *d_x,const float * d_Kernel_InterRow, float *d_KernelDotProd, 
float *d_SelfDotProd,const int& m,const int& n,const int &nbrCtas,const int& threadsPerCta, sycl::queue &q_ct1, float &elapsed_kernel_time)
{
 

        const int k = 1; 
    
	    //const int MBLOCKSIZE = 32;
        const int MBLOCKSIZE = 16;

	    unsigned int grid_rows = (m + MBLOCKSIZE - 1) / MBLOCKSIZE;
        unsigned int grid_cols = (n + MBLOCKSIZE - 1) / MBLOCKSIZE;

        sycl::range<3> dimGrid(grid_cols, grid_rows, 1);
        sycl::range<3> dimBlock(MBLOCKSIZE, MBLOCKSIZE, 1);

        sycl::event queue_event;
        sycl::cl_ulong time_start, time_end;

    #if USE_CUBLAS
        cublasHandle_t handle;
        CHECK_ERROR(cublasCreate(&handle)); 

    q_ct1.submit([&](sycl::handler &cgh) {
        //auto d_A = b_A.get_access<sycl::access::mode::read_write>(cgh);
        cgh.host_task([=](sycl::interop_handle ih) {
            cuCtxSetCurrent(ih.get_native_context<sycl::backend::ext_oneapi_cuda>());
            auto cuStream = ih.get_native_queue<sycl::backend::ext_oneapi_cuda>();
            cublasSetStream(handle, cuStream);
            constexpr float ALPHA = 1.0f;
            constexpr float BETA = 0.0f;
            CHECK_ERROR(cublasSgemv (handle, CUBLAS_OP_N, m, n, &ALPHA, d_x, m, d_Kernel_InterRow, 1, &BETA, d_KernelDotProd, 1));
            cudaStreamSynchronize(cuStream);
            //cudaDeviceSynchronize(); 
        });
    });
    q_ct1.wait_and_throw();

    cublasDestroy(handle);

    #elif USE_HIPBLAS

    constexpr float ALPHA = 1.0f;
  	constexpr float BETA = 0.0f;  

  	hipblasHandle_t handle;
  	hipblasCreate(&handle); 

         
  	q_ct1.submit([&](sycl::handler &h){

                h.host_task([=](sycl::interop_handle ih) {
                      //hipCtxSetCurrent(ih.get_native_context<sycl::backend::ext_oneapi_hip>());
                      //hipblasSetStream(handle, ih.get_native_queue<sycl::backend::ext_oneapi_hip>());
                      hipblasSgemv (handle, HIPBLAS_OP_N, m, n, &ALPHA, d_x, m, d_Kernel_InterRow, 1, &BETA, d_KernelDotProd, 1); 
                });
        });
        q_ct1.wait_and_throw();
    
    #else
        oneapi::mkl::blas::column_major::gemv(q_ct1,  oneapi::mkl::transpose::nontrans, m, n, 1, d_x, m, d_Kernel_InterRow, 1, 0, d_KernelDotProd, 1);
        q_ct1.wait();
    #endif
    

    #if KERNEL_USE_PROFILE
        queue_event =  q_ct1.submit([&](sycl::handler &cgh) {
    #else
        q_ct1.submit([&](sycl::handler &cgh) {
    #endif
        
        auto kernelwidth_ptr_ct1 = kernelwidth;

        cgh.parallel_for<class RBFKernel2>(sycl::nd_range<3>(sycl::range<3>(1, 1, nbrCtas) *
                                         sycl::range<3>(1, 1, threadsPerCta),
                                     sycl::range<3>(1, 1, threadsPerCta)),
                   [=](sycl::nd_item<3> item_ct1) {
                    RBFFinish(d_KernelJ, d_KernelDotProd, d_SelfDotProd,
                              d_SelfDotProd + BJIndex, m, item_ct1,
                              kernelwidth_ptr_ct1);
                   });
    });

    q_ct1.wait_and_throw();
    
    #if KERNEL_USE_PROFILE
        time_start = queue_event.template get_profiling_info<sycl::info::event_profiling::command_start>();
        time_end = queue_event.template get_profiling_info<sycl::info::event_profiling::command_end>();
        elapsed_kernel_time += (time_end - time_start)/1e9;
    #endif 

}

void CpuMaxInd(float &BIValue, int &BIIndex,const float * value_inter,const  int * index_inter,const  int n)
{

	BIValue=value_inter[0];
	BIIndex=index_inter[0];

	for(int j=0;j<n;j++)
	{
		if (value_inter[j]>BIValue)
		{
			BIValue=value_inter[j];
			BIIndex=index_inter[j];

		}
	}   

}

void CpuMaxIndSvr(float &BIValue, int &BIIndex, const  float * value_inter,const  int * index_inter,int n,const  int m)
{

	BIValue=value_inter[0];
	BIIndex=index_inter[0];

	for(int j=0;j<n;j++)
	{
		if (value_inter[j]>BIValue)
		{
			BIValue=value_inter[j];
			BIIndex=j<n/2?index_inter[j]:index_inter[j]+m;

		}
	}

}




void CpuMin(float &SJValue, float * value_inter,int n)
{

	SJValue=value_inter[0];

	for(int j=0;j<n;j++)
	{
		if (value_inter[j]<SJValue)
		{
			SJValue=value_inter[j];

		}
	}

}



void DotProdVector(float * x, float* dotprod,int m, int n)
{

	for(int i=0;i<m;i++)
	{
		dotprod[i]=0;

		for(int j=0;j<n;j++)
			dotprod[i]+=(x[i+j*m])*(x[i+j*m]);

	}



}

void IncrementKernelCache(std::vector<int>& KernelCacheItersSinceUsed,const int &RowsInKernelCache)
{
	for(int k=0;k<RowsInKernelCache;k++)
	{
		KernelCacheItersSinceUsed[k]+=1;
	}
}

inline void UpdateAlphas(float& alphai,float& alphaj,const float& Kij,const float& yi,const float& yj,const float& Fi,const float& Fj,const float& C,const float& h_taumin)
{

	//This alpha update code is adapted from that in LIBSVM.  
	//Chih-Chung Chang and Chih-Jen Lin, LIBSVM : a library for support vector machines, 2001. Software available at http://www.csie.ntu.edu.tw/~cjlin/libsvm 

	float lambda;
	float lambda_denom;


	lambda_denom=2.0-2.0*Kij;
	if (lambda_denom<h_taumin) {lambda_denom=h_taumin;}

	if (yi!=yj)
	{
		lambda=(-Fi-Fj)/lambda_denom;
		float alphadiff=alphai-alphaj;

		alphai+=lambda;
		alphaj+=lambda;


		if(alphadiff > 0)
		{
			if(alphaj < 0)
			{
				alphaj = 0;
				alphai = alphadiff;
			}



		}
		else
		{
			if(alphai < 0)
			{
				alphai = 0;
				alphaj = -alphadiff;
			}
		}


		if(alphadiff > 0)
		{
			if(alphai > C)
			{
				alphai = C;
				alphaj = C - alphadiff;
			}
		}
		else
		{
			if(alphaj > C)
			{
				alphaj = C;
				alphai = C + alphadiff;
			}
		}


	}
	else
	{
		float alphasum=alphai+alphaj;
		lambda=(Fi-Fj)/lambda_denom;
		alphai-=lambda;
		alphaj+=lambda;

		if(alphasum > C)
		{
			if(alphai > C)
			{
				alphai = C;
				alphaj = alphasum - C;
			}
			if(alphaj > C)
			{
				alphaj = C;
				alphai = alphasum - C;
			}
		}
		else
		{
			if(alphaj < 0)
			{
				alphaj = 0;
				alphai = alphasum;
			}
			if(alphai < 0)
			{
				alphai = 0;
				alphaj = alphasum;
			}
		}

	}

}



sycl::device selectedDevice(int plateform_id, int device_id){

	std::vector<sycl::platform> platforms = sycl::platform::get_platforms();
	sycl::device selected_device;	

    std::cout << "Platform:" << std::endl;
    std::cout << "+--Device:" << std::endl;
    std::cout << "---------------------------------" << std::endl;
    
    bool device_found = false;

    for (int plateform_index = 0; plateform_index < platforms.size(); plateform_index++){
        
        std::string name = platforms[plateform_index].get_info<sycl::info::platform::name>();
        std::vector<sycl::device> devices = platforms[plateform_index].get_devices(sycl::info::device_type::all);   
        bool is_selected_plateform = 0;
        if (plateform_id != plateform_index){
	   		std::cout << "[" << plateform_index << "]" << name << std::endl;
            
		} else {
			std::cout << "[X]" << name << std::endl;
		}
    
        for (int device_index = 0; device_index < devices.size(); device_index++){
            sycl::device device = devices[device_index];
            std::string device_name = device.get_info<sycl::info::device::name>(); 
            
            if (!((device_id == device_index) && (plateform_id == plateform_index))) { 
                std::cout << "+--[" << device_index  << "]" << device_name << std::endl;
            } else {
                std::cout << "+--[X]" << device_name << std::endl;
                selected_device = device; 
                device_found = true;
            }         
        }
    }

	return selected_device; 
}   





extern "C" void SVMTrain(float *mexalpha, float *beta, float *y, float *x,
                         float _C, float _kernelwidth, int m, int n,
                         float StoppingCrit, int argc, const char *argv[]) {

//try {

    printf("_C %f\n", _C);
    float elapsed_kernel_time= 0;
    sycl::cl_ulong time_start, time_end;
    sycl::event queue_event;

    sycl::event start, stop;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_ct1;
    std::chrono::time_point<std::chrono::high_resolution_clock> stop_ct1;

    start_ct1 = std::chrono::high_resolution_clock::now();


    sycl::device selected_device = sycl::device(sycl::default_selector());
    sycl::context context({selected_device});

    auto max_wgroup_size = selected_device.get_info<sycl::info::device::max_work_group_size>();
    printf("Workgroup Size: %lu\n", max_wgroup_size);

    auto propList = sycl::property_list{
        #if IN_ORDER_QUEUE
        sycl::property::queue::in_order{},
        #ifdef SYCL_EXT_ONEAPI_DISCARD_QUEUE_EVENTS
        sycl::ext::oneapi::property::queue::discard_events{},
        #endif
        #endif
        #if KERNEL_USE_PROFILE
        sycl::property::queue::enable_profiling{},
        #endif
    };
    sycl::queue q_ct1(context, selected_device, propList);


 mxArray *mexelapsed =mxCreateNumericMatrix(1, 1,mxSINGLE_CLASS, mxREAL);
 float * elapsed=(float *)mxGetData(mexelapsed);


 int numBlocks=64;
 sycl::range<3> ReduceGrid(numBlocks, 1, 1);
 sycl::range<3> ReduceBlock(256, 1, 1);

 float h_taumin=0.0001;

 int error = 0;


 //mxCUDA_SAFE_CALL((q_ct1.memcpy(taumin.get_ptr(), &h_taumin, sizeof(float)).wait(), 0));
 taumin = h_taumin;    
 C=_C;
 //mxCUDA_SAFE_CALL((q_ct1.memcpy(C.get_ptr(), &h_C, sizeof(float)).wait(), 0));
_kernelwidth*=-1;
 //mxCUDA_SAFE_CALL((q_ct1.memcpy(kernelwidth.get_ptr(), &_kernelwidth, sizeof(float)).wait(), 0));
 kernelwidth = _kernelwidth; 
 //mxCUDA_SAFE_CALL((q_ct1.memcpy(C.get_ptr(), &_C, sizeof(float)).wait(), 0));
 

 float *h_alpha=new float [m];
 float *h_F=new float [m];

	for(int j=0;j<m;j++)
	{
		h_alpha[j]=0;
		h_F[j]=-1;
	}


	float *SelfDotProd=new float [m];
	DotProdVector(x, SelfDotProd,m, n);

	int nbrCtas;
	int elemsPerCta;
	int threadsPerCta;

	VectorSplay (m, SAXPY_THREAD_MIN, SAXPY_THREAD_MAX, SAXPY_CTAS_MAX, &nbrCtas, &elemsPerCta,&threadsPerCta);

    printf("nbrCtas %i \n", nbrCtas);
    printf("elemsPerCta %i \n", elemsPerCta);
    printf("threadsPerCta %i \n", threadsPerCta);

    float * d_x;
	float * d_xT;
	float * d_alpha;
	float* d_y;
	float* d_F;
	float *d_KernelDotProd;
	float *kernelDotProd = new float[m];
    float *d_SelfDotProd;
	float *d_KernelJ;
	float *d_KernelI;

    mxCUDA_SAFE_CALL((d_x = sycl::malloc_device<float>(m * n * sizeof(float), q_ct1), 0));
    mxCUDA_SAFE_CALL((d_xT = sycl::malloc_device<float>(m * n * sizeof(float), q_ct1), 0));
    mxCUDA_SAFE_CALL((q_ct1.memcpy(d_x, x, sizeof(float) * n * m), 0));
    q_ct1.wait();
 

    sycl::range<3> gridtranspose(ceil((float)m / TRANS_BLOCK_DIM),
                              ceil((float)n / TRANS_BLOCK_DIM), 1);
    sycl::range<3> threadstranspose(TRANS_BLOCK_DIM, TRANS_BLOCK_DIM, 1);
    
    

    #if KERNEL_USE_PROFILE
        queue_event = q_ct1.submit([&](sycl::handler &cgh) {
    #else
      q_ct1.submit([&](sycl::handler &cgh) {
    #endif


    sycl::range<2> block_range_ct1(16 /*TRANS_BLOCK_DIM*/,
                                 17 /*TRANS_BLOCK_DIM+1*/);

    sycl::accessor<float, 2, sycl::access::mode::read_write, sycl::access::target::local> block_acc_ct1(block_range_ct1, cgh);

    auto dpct_global_range = gridtranspose * threadstranspose;
    
    cgh.parallel_for<class TransposeKernel>( 
     sycl::nd_range<3>(
          sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1),
                         dpct_global_range.get(0)),
          sycl::range<3>(threadstranspose.get(2), threadstranspose.get(1),
                         threadstranspose.get(0))),
      [=](sycl::nd_item<3> item_ct1) {
               // read the matrix tile into shared memory
        unsigned int xIndex =   item_ct1.get_group(2) * TRANS_BLOCK_DIM + item_ct1.get_local_id(2);
        unsigned int yIndex = item_ct1.get_group(1) * TRANS_BLOCK_DIM + item_ct1.get_local_id(1);
        
        if((xIndex < m) && (yIndex < n))
	    {
		    unsigned int index_in = yIndex * m + xIndex;
            block_acc_ct1[item_ct1.get_local_id(1)][item_ct1.get_local_id(2)] = d_x[index_in];
        }

        item_ct1.barrier();

        // write the transposed matrix tile to global memory
        xIndex = item_ct1.get_group(1) * TRANS_BLOCK_DIM + item_ct1.get_local_id(2);
        yIndex = item_ct1.get_group(2) * TRANS_BLOCK_DIM + item_ct1.get_local_id(1);
        
        if((xIndex < n) && (yIndex < m))
	    {
		    unsigned int index_out = yIndex * n + xIndex;
            d_xT[index_out] = block_acc_ct1[item_ct1.get_local_id(2)][item_ct1.get_local_id(1)];
        }
      });
    });
    
    q_ct1.wait_and_throw();
     
    #if KERNEL_USE_PROFILE
        time_start = queue_event.template get_profiling_info<sycl::info::event_profiling::command_start>();
        time_end = queue_event.template get_profiling_info<sycl::info::event_profiling::command_end>();
        elapsed_kernel_time += (time_end - time_start)/1e9;
    #endif	    

    float *xT=new float [n*m];
 
    mxCUDA_SAFE_CALL((q_ct1.memcpy(xT, d_xT, sizeof(float) * m * n), 0));
    q_ct1.wait();
    (sycl::free(d_xT, q_ct1), 0);
 

    float* d_KernelInterRow;
    mxCUDA_SAFE_CALL((d_KernelInterRow = sycl::malloc_device<float>(n * sizeof(float), q_ct1), 0));
    mxCUDA_SAFE_CALL((d_alpha = sycl::malloc_device<float>(m * sizeof(float), q_ct1), 0));
    mxCUDA_SAFE_CALL((d_y = sycl::malloc_device<float>(m * sizeof(float), q_ct1), 0));
    mxCUDA_SAFE_CALL((d_F = sycl::malloc_device<float>(m * sizeof(float), q_ct1), 0));
    mxCUDA_SAFE_CALL((d_SelfDotProd = sycl::malloc_device<float>(m * sizeof(float), q_ct1), 0));
    mxCUDA_SAFE_CALL((d_KernelDotProd = sycl::malloc_device<float>(m * sizeof(float), q_ct1), 0));
    
    mxCUDA_SAFE_CALL((q_ct1.memcpy(d_y, y, sizeof(float) * m), 0));
    mxCUDA_SAFE_CALL((q_ct1.memcpy(d_alpha, h_alpha, sizeof(float) * m), 0));
    mxCUDA_SAFE_CALL((q_ct1.memcpy(d_F, h_F, sizeof(float) * m), 0));
    mxCUDA_SAFE_CALL((q_ct1.memcpy(d_SelfDotProd, SelfDotProd, sizeof(float) * m), 0));
    q_ct1.wait();
 

    delete [] SelfDotProd;


    float* value_inter;
    int* index_inter;
 
    value_inter = sycl::malloc_host<float>(numBlocks, q_ct1);
    index_inter = sycl::malloc_host<int>(numBlocks, q_ct1);



    float* d_value_inter;
    int* d_index_inter;

 
    mxCUDA_SAFE_CALL((d_value_inter = sycl::malloc_device<float>(numBlocks * sizeof(float), q_ct1), 0));
    mxCUDA_SAFE_CALL((d_index_inter = sycl::malloc_device<int>(numBlocks * sizeof(int), q_ct1), 0));
 

    size_t free_mem, total;
 
    //cuMemGetInfo(&free_mem, &total);
    free_mem = INT_MAX;
    //printf("free mem %i\n", free_mem);

	size_t KernelCacheSize=free_mem-MBtoLeave*1024*1024;
	int RowsInKernelCache=KernelCacheSize/(sizeof(float)*m);

	/* Do not use all memory available if not needed. */
	if (RowsInKernelCache > m) {
		RowsInKernelCache = m;
		KernelCacheSize = m * sizeof(float) * m;
	}

	float *d_Kernel_Cache;
 
    mxCUDA_SAFE_CALL((d_Kernel_Cache = (float *)sycl::malloc_device(KernelCacheSize, q_ct1), 0));

    std::vector<int> KernelCacheIndices(RowsInKernelCache,-1);
	std::vector<int> KernelCacheItersSinceUsed(RowsInKernelCache,0);
	std::vector<int>::iterator CachePosI;
	std::vector<int>::iterator CachePosJ;
	int CacheDiffI;
	int CacheDiffJ;

	int CheckStoppingCritEvery=255;
	int iter=0;

	float BIValue;
	int BIIndex;
	float SJValue;
	float BJSecondOrderValue;
	int BJIndex;
	float Kij;
	float yj;
	float yi;
	float alphai;
	float alphaj;
	float oldalphai;
	float oldalphaj;
	float Fi;
	float Fj;

    for(int index = 0; index < NUM_ITERATIONS; index++)
    {

        
        #if KERNEL_USE_PROFILE
            queue_event =  q_ct1.submit([&](sycl::handler &cgh) {
        #else
            q_ct1.submit([&](sycl::handler &cgh) {
        #endif
            
            auto  C_ptr_ct1 = C;
            

            sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> sdata_acc_ct1(sycl::range<1>(256), cgh);
            sycl::accessor<int, 1, sycl::access::mode::read_write, sycl::access::target::local> ind_acc_ct1(sycl::range<1>(256), cgh);


            auto dpct_global_range = ReduceGrid * ReduceBlock;

            cgh.parallel_for<class FindBIKernel>(
                        sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                        dpct_global_range.get(1),
                                        dpct_global_range.get(0)),
                        sycl::range<3>(ReduceBlock.get(2), ReduceBlock.get(1),
                                        ReduceBlock.get(0))),
                        [=](sycl::nd_item<3> item_ct1) {
                            FindBI<256>(d_F, d_y, d_alpha, d_value_inter, d_index_inter, m,
                             item_ct1, C_ptr_ct1, sdata_acc_ct1.get_pointer(),
                             ind_acc_ct1.get_pointer());
                        });
        });
        
        q_ct1.wait_and_throw();	

        #if KERNEL_USE_PROFILE
            time_start = queue_event.template get_profiling_info<sycl::info::event_profiling::command_start>();
            time_end = queue_event.template get_profiling_info<sycl::info::event_profiling::command_end>();
            elapsed_kernel_time += (time_end - time_start)/1e9;
        #endif
        
        mxCUDA_SAFE_CALL((q_ct1.memcpy(value_inter, d_value_inter, sizeof(float) * numBlocks), 0));
        mxCUDA_SAFE_CALL((q_ct1.memcpy(index_inter, d_index_inter, sizeof(int) * numBlocks), 0));
        q_ct1.wait();
  
        CpuMaxInd(BIValue,BIIndex,value_inter,index_inter,numBlocks);

        q_ct1.memcpy(&Fi, d_F + BIIndex, sizeof(float));
        q_ct1.wait();

        if (iter == (NUM_ITERATIONS - 1))
        {

            #if KERNEL_USE_PROFILE
                queue_event =  q_ct1.submit([&](sycl::handler &cgh) {
            #else
                q_ct1.submit([&](sycl::handler &cgh) {
            #endif

                auto C_ptr_ct1 = C;

                sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> sdata_acc_ct1(sycl::range<1>(256), cgh);

                auto dpct_global_range = ReduceGrid * ReduceBlock;

                cgh.parallel_for<class FindStoppingJKernel>(
                          sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                         dpct_global_range.get(1),
                                         dpct_global_range.get(0)),
                          sycl::range<3>(ReduceBlock.get(2), ReduceBlock.get(1),
                                         ReduceBlock.get(0))),
                         [=](sycl::nd_item<3> item_ct1) {
                            FindStoppingJ<256>(d_F, d_y, d_alpha, d_value_inter, m, item_ct1,
                            C_ptr_ct1, sdata_acc_ct1.get_pointer());
                });
            });
   
            q_ct1.wait_and_throw();     
	   	

            #if KERNEL_USE_PROFILE
                time_start = queue_event.template get_profiling_info<sycl::info::event_profiling::command_start>();
                time_end = queue_event.template get_profiling_info<sycl::info::event_profiling::command_end>();
                elapsed_kernel_time += (time_end - time_start)/1e9;
            #endif
            
            
            
            
            mxCUDA_SAFE_CALL((q_ct1.memcpy(value_inter, d_value_inter, sizeof(float) * numBlocks), 0));
            q_ct1.wait();
            
            CpuMin(SJValue,value_inter,numBlocks);

            //if(BIValue-SJValue<StoppingCrit) 
			//{
               
				*beta=(SJValue+BIValue)/2;
				if(BIValue-SJValue<StoppingCrit) {*beta=(SJValue+BIValue)/2; break;}

			//}
		}

        
        CachePosI=find(KernelCacheIndices.begin(),KernelCacheIndices.end(),BIIndex);
		
        if (CachePosI ==KernelCacheIndices.end())
		{
            
			CacheDiffI=max_element(KernelCacheItersSinceUsed.begin(),KernelCacheItersSinceUsed.end())-KernelCacheItersSinceUsed.begin();
			
            d_KernelI=d_Kernel_Cache+CacheDiffI*m;
   
            mxCUDA_SAFE_CALL((q_ct1.memcpy(d_KernelInterRow, xT + BIIndex * n, n * sizeof(float)), 0));
            q_ct1.wait();

            RBFKernel(d_KernelI,BIIndex,d_x,d_KernelInterRow,d_KernelDotProd,d_SelfDotProd, m,n,nbrCtas,threadsPerCta, q_ct1, elapsed_kernel_time);
            
            
            *(KernelCacheIndices.begin()+CacheDiffI)=BIIndex;
		}
		else
		{
            
			CacheDiffI=CachePosI-KernelCacheIndices.begin();
			d_KernelI=d_Kernel_Cache+m*CacheDiffI;
		}
		*(KernelCacheItersSinceUsed.begin()+CacheDiffI)=-1;

        #if KERNEL_USE_PROFILE
            queue_event = q_ct1.submit([&](sycl::handler &cgh) {
        #else
            q_ct1.submit([&](sycl::handler &cgh) {
        #endif   
            
            auto C_ptr_ct1 = C;
            auto taumin_ptr_ct1 = taumin;

            sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> sdata_acc_ct1(sycl::range<1>(256), cgh);
            sycl::accessor<int, 1, sycl::access::mode::read_write, sycl::access::target::local> ind_acc_ct1(sycl::range<1>(256), cgh);

            auto dpct_global_range = ReduceGrid * ReduceBlock;

            cgh.parallel_for<class FindBJKernel>(
                            sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                        dpct_global_range.get(1),
                                        dpct_global_range.get(0)),
                            sycl::range<3>(ReduceBlock.get(2), ReduceBlock.get(1),
                                        ReduceBlock.get(0))),
                            [=](sycl::nd_item<3> item_ct1) {
                                FindBJ<256>(d_F, d_y, d_alpha, d_KernelI, d_value_inter, d_index_inter,
                                BIValue, m, item_ct1, C_ptr_ct1,taumin_ptr_ct1,
                                sdata_acc_ct1.get_pointer(), ind_acc_ct1.get_pointer());
                            });
        });
        
        q_ct1.wait_and_throw();
        

         #if KERNEL_USE_PROFILE
            time_start = queue_event.template get_profiling_info<sycl::info::event_profiling::command_start>();
            time_end = queue_event.template get_profiling_info<sycl::info::event_profiling::command_end>();
            elapsed_kernel_time += (time_end - time_start)/1e9;
        #endif



        mxCUDA_SAFE_CALL((q_ct1.memcpy(value_inter, d_value_inter, sizeof(float) * numBlocks), 0));
        mxCUDA_SAFE_CALL((q_ct1.memcpy(index_inter, d_index_inter, sizeof(int) * numBlocks), 0));
        q_ct1.wait();
        
        CpuMaxInd(BJSecondOrderValue,BJIndex,value_inter,index_inter,numBlocks);

        mxCUDA_SAFE_CALL( (q_ct1.memcpy(&Kij, d_KernelI + BJIndex, sizeof(float)), 0));
        mxCUDA_SAFE_CALL( (q_ct1.memcpy(&alphai, d_alpha + BIIndex, sizeof(float)), 0));
        mxCUDA_SAFE_CALL( (q_ct1.memcpy(&alphaj, d_alpha + BJIndex, sizeof(float)), 0));
        mxCUDA_SAFE_CALL((q_ct1.memcpy(&yi, d_y + BIIndex, sizeof(float)), 0));
        mxCUDA_SAFE_CALL((q_ct1.memcpy(&yj, d_y + BJIndex, sizeof(float)), 0));
        mxCUDA_SAFE_CALL((q_ct1.memcpy(&Fj, d_F + BJIndex, sizeof(float)), 0));
        q_ct1.wait();

        oldalphai=alphai;
		
        oldalphaj=alphaj;


		UpdateAlphas(alphai,alphaj,Kij,yi,yj,Fi,Fj,_C,h_taumin);

        mxCUDA_SAFE_CALL((q_ct1.memcpy(d_alpha + BIIndex, &alphai, sizeof(float)), 0));
        mxCUDA_SAFE_CALL( (q_ct1.memcpy(d_alpha + BJIndex, &alphaj, sizeof(float)), 0));
        q_ct1.wait();

        float deltaalphai = alphai - oldalphai;
		float deltaalphaj = alphaj - oldalphaj;


		CachePosJ=find(KernelCacheIndices.begin(),KernelCacheIndices.end(),BJIndex);
		if (CachePosJ ==KernelCacheIndices.end())
		{
			CacheDiffJ=max_element(KernelCacheItersSinceUsed.begin(),KernelCacheItersSinceUsed.end())-KernelCacheItersSinceUsed.begin();
			d_KernelJ=d_Kernel_Cache+CacheDiffJ*m;
   

            mxCUDA_SAFE_CALL( (q_ct1.memcpy(d_KernelInterRow, xT + BJIndex * n, n * sizeof(float)), 0));
            q_ct1.wait();
          
            RBFKernel(d_KernelJ,BJIndex,d_x,d_KernelInterRow,d_KernelDotProd, d_SelfDotProd, m,n,nbrCtas,threadsPerCta, q_ct1, elapsed_kernel_time);
			*(KernelCacheIndices.begin()+CacheDiffJ)=BJIndex;
           

		}
		else
		{
			CacheDiffJ=CachePosJ-KernelCacheIndices.begin();
			d_KernelJ=d_Kernel_Cache+m*CacheDiffJ;

		}

        #if KERNEL_USE_PROFILE
            sycl::event queue_event =  q_ct1.submit([&](sycl::handler &cgh) {
        #else
            q_ct1.submit([&](sycl::handler &cgh) {
        #endif

            cgh.parallel_for<class UpdateFKernel>(sycl::nd_range<3>(sycl::range<3>(1, 1, nbrCtas) *
                                          sycl::range<3>(1, 1, threadsPerCta),
                                      sycl::range<3>(1, 1, threadsPerCta)),
                    [=](sycl::nd_item<3> item_ct1) {
                     UpdateF(d_F, d_KernelI, d_KernelJ, d_y, deltaalphai,
                             deltaalphaj, yi, yj, m, item_ct1);
                    });
        });


        q_ct1.wait_and_throw();

        #if KERNEL_USE_PROFILE
            time_start = queue_event.template get_profiling_info<sycl::info::event_profiling::command_start>();
            time_end = queue_event.template get_profiling_info<sycl::info::event_profiling::command_end>();
            elapsed_kernel_time += (time_end - time_start)/1e9;
        #endif



        IncrementKernelCache(KernelCacheItersSinceUsed,RowsInKernelCache);

		*(KernelCacheItersSinceUsed.begin()+CacheDiffI)=0;
		*(KernelCacheItersSinceUsed.begin()+CacheDiffJ)=0;



		iter++;

	}
    
    #if KERNEL_USE_PROFILE
        printf("Total Kernel Time: %f sec.\n", elapsed_kernel_time);
		//cout << "Kernel perf: " << ((noptions * 5)/ elapsed) << " Options per second\n";
		//cout << "Average Kernel Time per run: " << (elapsed/NUM_OF_RUNS) << " sec.\n";
	#endif
    
    q_ct1.memcpy(mexalpha, d_alpha, m * sizeof(float));
    q_ct1.wait();

    stop_ct1 = std::chrono::high_resolution_clock::now();
    
    //stop.wait_and_throw();
    float duration = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();
    printf("Total run time: %f seconds\n", duration/1000.00); 


    printf("Iter:%i\n", iter);
	printf("M:%i\n", m);
	printf("N:%i\n", n);


    mexPutVariable("base","cuSVMTrainTimeInMS",mexelapsed);
	
	delete [] xT;
	
	#ifdef DISABLE_CUDA_PINNED_ALLOC
	delete [] value_inter;
	delete [] index_inter;
	#else
    sycl::free(value_inter, q_ct1);
    sycl::free(index_inter, q_ct1);
	#endif

    delete[] kernelDotProd;
    mxCUDA_SAFE_CALL((sycl::free(d_x, q_ct1), 0));
    mxCUDA_SAFE_CALL((sycl::free(d_y, q_ct1), 0));
    mxCUDA_SAFE_CALL((sycl::free(d_alpha, q_ct1), 0));
    mxCUDA_SAFE_CALL((sycl::free(d_KernelInterRow, q_ct1), 0));
    mxCUDA_SAFE_CALL((sycl::free(d_Kernel_Cache, q_ct1), 0));
    mxCUDA_SAFE_CALL((sycl::free(d_F, q_ct1), 0));
    mxCUDA_SAFE_CALL((sycl::free(d_value_inter, q_ct1), 0));
    mxCUDA_SAFE_CALL((sycl::free(d_index_inter, q_ct1), 0));
    mxCUDA_SAFE_CALL((sycl::free(d_SelfDotProd, q_ct1), 0));
    mxCUDA_SAFE_CALL((sycl::free(d_KernelDotProd, q_ct1), 0));
 
    //return;
}