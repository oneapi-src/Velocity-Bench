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




#include "hip/hip_runtime.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>
#include <ctype.h>
#include <float.h>
#include <algorithm>
#include <math.h>
#include "mex.h" 
#include "hip/hip_runtime.h" 
#include <chrono>
#include "cuSVMutil.h"
#include <vector>
#include "hipblas.h"

__constant__ float C;
__constant__ float taumin;
__constant__ float kernelwidth;


#define NUM_ITERATIONS 100

template <unsigned int blockSize>
__global__ void FindBJ(float *d_F, float* d_y,float* d_alpha,float* d_KernelCol,float *g_odata,int* g_index,float BIValue, unsigned int n)
{

	__shared__ float sdata[blockSize];
	__shared__ int ind[blockSize];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize*2) + tid;
	unsigned int gridSize = blockSize*2*gridDim.x;
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


		maxtemp=
			fmaxf(
			globaltemp=
			(LocalCloseY*d_alpha[i])>(LocalCloseY==1?0:-C) ?
			__fdividef(__powf(BIValue+LocalCloseY*d_F[i],2.f),denomclose)
			:-FLT_MAX, 
			i+blockSize<n ? 
			((LocalFarY*d_alpha[i+blockSize])>(LocalFarY==1?0:-C)?      
			__fdividef(__powf(BIValue+LocalFarY*d_F[i+blockSize],2.f),denomfar)
			:-FLT_MAX)
			:-FLT_MAX);

		sdata[tid]=fmaxf(temp=sdata[tid],maxtemp);

		if (sdata[tid]!=temp)
		{
			sdata[tid]== globaltemp ? ind[tid]=i : ind[tid]=i+blockSize;
		}

		i += gridSize; 
	}


	__syncthreads();

	if (tid < 128){ if (sdata[tid] < sdata[tid + 128]){ ind[tid]=ind[tid+128];sdata[tid]=sdata[tid+128];  }} __syncthreads(); 

	if (tid < 64){ if (sdata[tid] < sdata[tid + 64]){ ind[tid]=ind[tid+64];sdata[tid]=sdata[tid+64];  }} __syncthreads();   

	//if (tid < 32) 
	//{
		if ((tid < 32) && (sdata[tid] <sdata[tid + 32])) {ind[tid]=ind[tid+32];sdata[tid]=sdata[tid+32];} __syncthreads();
		if ((tid < 32) && (sdata[tid] <sdata[tid + 16])) {ind[tid]=ind[tid+16];sdata[tid]=sdata[tid+16];} __syncthreads();
		if ((tid < 32) && (sdata[tid] <sdata[tid + 8])) {ind[tid]=ind[tid+8];sdata[tid]=sdata[tid+8];} __syncthreads();
		if ((tid < 32) && (sdata[tid] <sdata[tid + 4])) {ind[tid]=ind[tid+4];sdata[tid]=sdata[tid+4];} __syncthreads();
		if ((tid < 32) && (sdata[tid] <sdata[tid + 2])) {ind[tid]=ind[tid+2];sdata[tid]=sdata[tid+2];} __syncthreads();
		if ((tid < 32) && (sdata[tid] <sdata[tid + 1])) {ind[tid]=ind[tid+1];sdata[tid]=sdata[tid+1];} __syncthreads();
	//}
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
	if (tid == 0) g_index[blockIdx.x] = ind[0];
}

template <unsigned int blockSize>
__global__ void FindBI(float *d_F, float* d_y,float* d_alpha,float *g_odata,int* g_index,unsigned int n)
{

	__shared__ float sdata[blockSize];
	__shared__ int ind[blockSize];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize*2) + tid;
	unsigned int gridSize = blockSize*2*gridDim.x;
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

		maxtemp=
			fmaxf(
			globaltemp= 
			(LocalCloseY*d_alpha[i])<(LocalCloseY==1?C:0) ?  
			-(d_F[i]*LocalCloseY)  
			:-FLT_MAX, 
			i+blockSize<n ? 
			((LocalFarY*d_alpha[i+blockSize])<(LocalFarY==1?C:0) ?  
			-(d_F[i+blockSize]*LocalFarY)  
			:-FLT_MAX)
			:-FLT_MAX);

		sdata[tid]=fmaxf(temp=sdata[tid],maxtemp);

		if (sdata[tid]!=temp)
		{
			sdata[tid]== globaltemp ? ind[tid]=i : ind[tid]=i+blockSize;
		}

		i += gridSize; 
	}


	__syncthreads();

	if (tid < 128){ if (sdata[tid] < sdata[tid + 128]){ ind[tid]=ind[tid+128];sdata[tid]=sdata[tid+128];  }} __syncthreads(); 

	if (tid < 64){ if (sdata[tid] < sdata[tid + 64]){ ind[tid]=ind[tid+64];sdata[tid]=sdata[tid+64];  }} __syncthreads(); 

	//if (tid < 32) 
	//{
		if ((tid < 32) && (sdata[tid] <sdata[tid + 32])) {ind[tid]=ind[tid+32];sdata[tid]=sdata[tid+32];} __syncthreads();
		if ((tid < 32) && (sdata[tid] <sdata[tid + 16])) {ind[tid]=ind[tid+16];sdata[tid]=sdata[tid+16];} __syncthreads();
		if ((tid < 32) && (sdata[tid] <sdata[tid + 8])) {ind[tid]=ind[tid+8];sdata[tid]=sdata[tid+8];} __syncthreads();
		if ((tid < 32) && (sdata[tid] <sdata[tid + 4])) {ind[tid]=ind[tid+4];sdata[tid]=sdata[tid+4];} __syncthreads();
		if ((tid < 32) && (sdata[tid] <sdata[tid + 2])) {ind[tid]=ind[tid+2];sdata[tid]=sdata[tid+2];} __syncthreads();
		if ((tid < 32) && (sdata[tid] <sdata[tid + 1])) {ind[tid]=ind[tid+1];sdata[tid]=sdata[tid+1];} __syncthreads();

	//}
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
	if (tid == 0) g_index[blockIdx.x] = ind[0];
}


template <unsigned int blockSize>
__global__ void FindStoppingJ(float *d_F, float* d_y,float* d_alpha,float *g_odata,unsigned int n)
{

	__shared__ float sdata[blockSize];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize*2) + tid;
	unsigned int gridSize = blockSize*2*gridDim.x;
	sdata[tid]=FLT_MAX;


	float LocalCloseY;
	float LocalFarY;


	while (i < n) 
	{ 
		LocalCloseY=d_y[i];
		LocalFarY=(i+blockSize)<n ? d_y[i+blockSize]:0;

		sdata[tid]=
			fminf(
			sdata[tid],
			fminf( 
			(LocalCloseY*d_alpha[i])>(LocalCloseY==1?0:-C) ?  
			-(d_F[i]*LocalCloseY)  
			:FLT_MAX, 
			i+blockSize<n ? 
			((LocalFarY*d_alpha[i+blockSize])>(LocalFarY==1?0:-C)?  
			-(d_F[i+blockSize]*LocalFarY)  
			:FLT_MAX)
			:FLT_MAX));

		i += gridSize; 
	}   


	__syncthreads();

	if (tid < 128){ sdata[tid]=fminf(sdata[tid],sdata[tid+128]);} __syncthreads(); 

	if (tid < 64){ sdata[tid]=fminf(sdata[tid],sdata[tid+64]);} __syncthreads(); 

	if (tid < 32) {sdata[tid]=fminf(sdata[tid],sdata[tid+32]);}  __syncthreads();
	if (tid < 32) {sdata[tid]=fminf(sdata[tid],sdata[tid+16]);} __syncthreads();
	if (tid < 32) {sdata[tid]=fminf(sdata[tid],sdata[tid+8]);} __syncthreads();
	if (tid < 32) {sdata[tid]=fminf(sdata[tid],sdata[tid+4]);} __syncthreads();
	if (tid < 32) {sdata[tid]=fminf(sdata[tid],sdata[tid+2]);} __syncthreads();
	if (tid < 32) {sdata[tid]=fminf(sdata[tid],sdata[tid+1]);} __syncthreads();


	
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];

}




__global__ void UpdateF(float * F,float *KernelColI,float* KernelColJ, float* d_y,float deltaalphai,float deltaalphaj,float yi,float yj,int n)
{

	int totalThreads,ctaStart,tid;
	totalThreads = gridDim.x*blockDim.x;
	ctaStart = blockDim.x*blockIdx.x;
	tid = threadIdx.x;
	int i;

	for (i = ctaStart + tid; i < n; i += totalThreads) 
	{  
		F[i] = F[i] + yi*d_y[i]*deltaalphai*KernelColI[i]+yj*d_y[i]*deltaalphaj*KernelColJ[i];
	}


}

__global__ void RBFFinish(float *KernelCol, const float * KernelDotProd,const float* DotProd,const float* DotProdRow,const int n)
{

	int totalThreads,ctaStart,tid;
	totalThreads = gridDim.x*blockDim.x;
	ctaStart = blockDim.x*blockIdx.x;
	tid = threadIdx.x;
	int i;
	float temp;

	for (i = ctaStart + tid; i < n; i += totalThreads) 
	{
        KernelCol[i] = expf(kernelwidth*(DotProd[i]+*DotProdRow-KernelDotProd[i]*2.f));
	}


}


inline void RBFKernel(float *d_KernelJ,const int BJIndex,const float *d_x,const float * d_Kernel_InterRow,float *d_KernelDotProd, float *d_SelfDotProd,const int& m,const int& n,const int &nbrCtas,const int& threadsPerCta)
{

    //hipblasSgemv ('n', m, n, 1,d_x, m, d_Kernel_InterRow, 1, 0, d_KernelDotProd, 1);
    
    hipblasHandle_t handle = NULL;
    hipblasStatus_t status;
    status = hipblasCreate(&handle);
    const float alpha = 1.0, beta = 0.0;
    status = hipblasSgemv(handle, HIPBLAS_OP_N, m, n, &alpha, d_x, m, d_Kernel_InterRow, 1, &beta, d_KernelDotProd, 1); 	
    hipblasDestroy(handle);

    hipLaunchKernelGGL(RBFFinish, nbrCtas, threadsPerCta, 0, 0, d_KernelJ, d_KernelDotProd,d_SelfDotProd,d_SelfDotProd+BJIndex,m);
   
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

extern "C"
void SVMTrain(float *mexalpha,float* beta,float*y,float *x ,float _C, float _kernelwidth, int m, int n, float StoppingCrit)
{

	printf("_C %f\n", _C);

	hipEvent_t start, stop;
	hipEventCreate(&start);
	hipEventCreate(&stop);

	std::chrono::time_point<std::chrono::high_resolution_clock> start_ct1;
        std::chrono::time_point<std::chrono::high_resolution_clock> stop_ct1;

        start_ct1 = std::chrono::high_resolution_clock::now();	

	mxArray *mexelapsed =mxCreateNumericMatrix(1, 1,mxSINGLE_CLASS, mxREAL);
	float * elapsed=(float *)mxGetData(mexelapsed);


	hipEventRecord(start,0);

	int numBlocks=64;
	dim3 ReduceGrid(numBlocks, 1, 1);
	dim3 ReduceBlock(256, 1, 1);


	float h_taumin=0.0001;
	mxCUDA_SAFE_CALL(hipMemcpyToSymbol(HIP_SYMBOL(taumin), &h_taumin, sizeof(float)));

	_kernelwidth*=-1;
	mxCUDA_SAFE_CALL(hipMemcpyToSymbol(HIP_SYMBOL(kernelwidth), &_kernelwidth, sizeof(float)));

	mxCUDA_SAFE_CALL(hipMemcpyToSymbol(HIP_SYMBOL(C), &_C, sizeof(float)));



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

	mxCUDA_SAFE_CALL(hipMalloc( (void**) &d_x, m*n*sizeof(float)));
	mxCUDA_SAFE_CALL(hipMalloc( (void**) &d_xT, m*n*sizeof(float)));
	mxCUDA_SAFE_CALL(hipMemcpy(d_x, x, sizeof(float)*n*m,hipMemcpyHostToDevice));
	dim3 gridtranspose(ceil((float)m / TRANS_BLOCK_DIM), ceil((float)n / TRANS_BLOCK_DIM), 1);
	dim3 threadstranspose(TRANS_BLOCK_DIM, TRANS_BLOCK_DIM, 1);
	hipDeviceSynchronize();
	hipLaunchKernelGGL(transpose, gridtranspose, threadstranspose , 0, 0, d_xT, d_x, m, n);

	float *xT=new float [n*m];   
	mxCUDA_SAFE_CALL(hipMemcpy(xT, d_xT, sizeof(float)*m*n,hipMemcpyDeviceToHost));
	mxCUDA_SAFE_CALL(hipFree(d_xT));


	float* d_KernelInterRow;
	mxCUDA_SAFE_CALL(hipMalloc( (void**) &d_KernelInterRow, n*sizeof(float)));


	mxCUDA_SAFE_CALL(hipMalloc( (void**) &d_alpha, m*sizeof(float)));
	mxCUDA_SAFE_CALL(hipMalloc( (void**) &d_y, m*sizeof(float)));
	mxCUDA_SAFE_CALL(hipMalloc( (void**) &d_F, m*sizeof(float)));
	mxCUDA_SAFE_CALL(hipMalloc( (void**) &d_SelfDotProd, m*sizeof(float)));
	mxCUDA_SAFE_CALL(hipMalloc( (void**) &d_KernelDotProd, m*sizeof(float)));
    

    mxCUDA_SAFE_CALL(hipMemcpy(d_y, y, sizeof(float)*m,hipMemcpyHostToDevice));
	mxCUDA_SAFE_CALL(hipMemcpy(d_alpha, h_alpha, sizeof(float)*m,hipMemcpyHostToDevice));
	mxCUDA_SAFE_CALL(hipMemcpy(d_F, h_F, sizeof(float)*m,hipMemcpyHostToDevice));
	mxCUDA_SAFE_CALL(hipMemcpy(d_SelfDotProd, SelfDotProd, sizeof(float)*m,hipMemcpyHostToDevice));

    delete [] SelfDotProd;


	float* value_inter;
	int* index_inter;


	
	#ifdef DISABLE_CUDA_PINNED_ALLOC
	value_inter = new float[numBlocks];
	index_inter = new int[numBlocks];
	#else
	hipMallocHost( (void**)&value_inter, numBlocks*sizeof(float) );
	hipMallocHost( (void**)&index_inter, numBlocks*sizeof(int) );
	#endif


	float* d_value_inter;
	int* d_index_inter;


	mxCUDA_SAFE_CALL(hipMalloc( (void**) &d_value_inter, numBlocks*sizeof(float)));
	mxCUDA_SAFE_CALL(hipMalloc( (void**) &d_index_inter, numBlocks*sizeof(int)));

	size_t free_mem, total;
	//hipMemGetInfo(&free_mem, &total);
    	free_mem = INT_MAX;

	size_t KernelCacheSize=free_mem-MBtoLeave*1024*1024;
	int RowsInKernelCache=KernelCacheSize/(sizeof(float)*m);

	/* Do not use all memory available if not needed. */
	if (RowsInKernelCache > m) {
		RowsInKernelCache = m;
		KernelCacheSize = m * sizeof(float) * m;
	}

	float *d_Kernel_Cache;
	mxCUDA_SAFE_CALL(hipMalloc( (void**) &d_Kernel_Cache, KernelCacheSize));


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

		hipLaunchKernelGGL(HIP_KERNEL_NAME(FindBI<256>), ReduceGrid, ReduceBlock, 0, 0, d_F, d_y,d_alpha,d_value_inter,d_index_inter, m);
		mxCUDA_SAFE_CALL(hipMemcpy(value_inter, d_value_inter, sizeof(float)*numBlocks,hipMemcpyDeviceToHost));
		mxCUDA_SAFE_CALL(hipMemcpy(index_inter, d_index_inter, sizeof(int)*numBlocks,hipMemcpyDeviceToHost));
		hipDeviceSynchronize();
		CpuMaxInd(BIValue,BIIndex,value_inter,index_inter,numBlocks);

		hipMemcpy(&Fi, d_F+BIIndex, sizeof(float),hipMemcpyDeviceToHost);

		if (iter == (NUM_ITERATIONS -1))
		{
			hipLaunchKernelGGL(HIP_KERNEL_NAME(FindStoppingJ<256>), ReduceGrid, ReduceBlock, 0, 0, d_F, d_y,d_alpha,d_value_inter, m);
			mxCUDA_SAFE_CALL(hipMemcpy(value_inter, d_value_inter, sizeof(float)*numBlocks,hipMemcpyDeviceToHost));
			hipDeviceSynchronize();
			CpuMin(SJValue,value_inter,numBlocks);


			//if(BIValue-SJValue<StoppingCrit) 
			//{

				*beta=(SJValue+BIValue)/2;
				//if(BIValue-SJValue<StoppingCrit) {*beta=(SJValue+BIValue)/2; break;}

			//}
		}


        CachePosI=find(KernelCacheIndices.begin(),KernelCacheIndices.end(),BIIndex);
		if (CachePosI ==KernelCacheIndices.end())
		{
			CacheDiffI=max_element(KernelCacheItersSinceUsed.begin(),KernelCacheItersSinceUsed.end())-KernelCacheItersSinceUsed.begin();
			d_KernelI=d_Kernel_Cache+CacheDiffI*m;
			mxCUDA_SAFE_CALL(hipMemcpy(d_KernelInterRow, xT+BIIndex*n, n*sizeof(float),hipMemcpyHostToDevice));
            
            RBFKernel(d_KernelI,BIIndex,d_x,d_KernelInterRow,d_KernelDotProd,d_SelfDotProd, m,n,nbrCtas,threadsPerCta);
            hipDeviceSynchronize();
			*(KernelCacheIndices.begin()+CacheDiffI)=BIIndex;
		}
		else
		{
			CacheDiffI=CachePosI-KernelCacheIndices.begin();
			d_KernelI=d_Kernel_Cache+m*CacheDiffI;
		}
		*(KernelCacheItersSinceUsed.begin()+CacheDiffI)=-1;




		hipLaunchKernelGGL(HIP_KERNEL_NAME(FindBJ<256>), ReduceGrid, ReduceBlock, 0, 0, d_F, d_y,d_alpha,d_KernelI,d_value_inter,d_index_inter,BIValue, m);
		mxCUDA_SAFE_CALL(hipMemcpy(value_inter, d_value_inter, sizeof(float)*numBlocks,hipMemcpyDeviceToHost));
		mxCUDA_SAFE_CALL(hipMemcpy(index_inter, d_index_inter, sizeof(int)*numBlocks,hipMemcpyDeviceToHost));
		CpuMaxInd(BJSecondOrderValue,BJIndex,value_inter,index_inter,numBlocks);


		mxCUDA_SAFE_CALL(hipMemcpy(&Kij, d_KernelI+BJIndex, sizeof(float),hipMemcpyDeviceToHost));

		mxCUDA_SAFE_CALL(hipMemcpy(&alphai, d_alpha+BIIndex, sizeof(float),hipMemcpyDeviceToHost));
		mxCUDA_SAFE_CALL(hipMemcpy(&alphaj, d_alpha+BJIndex, sizeof(float),hipMemcpyDeviceToHost));

		mxCUDA_SAFE_CALL(hipMemcpy(&yi, d_y+BIIndex, sizeof(float),hipMemcpyDeviceToHost));
		mxCUDA_SAFE_CALL(hipMemcpy(&yj, d_y+BJIndex, sizeof(float),hipMemcpyDeviceToHost));
		mxCUDA_SAFE_CALL(hipMemcpy(&Fj, d_F+BJIndex, sizeof(float),hipMemcpyDeviceToHost));


		oldalphai=alphai;
		oldalphaj=alphaj;


		UpdateAlphas(alphai,alphaj,Kij,yi,yj,Fi,Fj,_C,h_taumin);



		mxCUDA_SAFE_CALL(hipMemcpy(d_alpha+BIIndex, &alphai, sizeof(float),hipMemcpyHostToDevice));
		mxCUDA_SAFE_CALL(hipMemcpy(d_alpha+BJIndex, &alphaj, sizeof(float),hipMemcpyHostToDevice));

		float deltaalphai = alphai - oldalphai;
		float deltaalphaj = alphaj - oldalphaj;


		CachePosJ=find(KernelCacheIndices.begin(),KernelCacheIndices.end(),BJIndex);
		if (CachePosJ ==KernelCacheIndices.end())
		{
			CacheDiffJ=max_element(KernelCacheItersSinceUsed.begin(),KernelCacheItersSinceUsed.end())-KernelCacheItersSinceUsed.begin();
			d_KernelJ=d_Kernel_Cache+CacheDiffJ*m;
			mxCUDA_SAFE_CALL(hipMemcpy(d_KernelInterRow, xT+BJIndex*n, n*sizeof(float),hipMemcpyHostToDevice));
            RBFKernel(d_KernelJ,BJIndex,d_x,d_KernelInterRow,d_KernelDotProd, d_SelfDotProd, m,n,nbrCtas,threadsPerCta);
			*(KernelCacheIndices.begin()+CacheDiffJ)=BJIndex;
		}
		else
		{
			CacheDiffJ=CachePosJ-KernelCacheIndices.begin();
			d_KernelJ=d_Kernel_Cache+m*CacheDiffJ;

		}



		hipLaunchKernelGGL(UpdateF, nbrCtas, threadsPerCta, 0, 0, d_F,d_KernelI,d_KernelJ,d_y,deltaalphai,deltaalphaj,yi,yj,m);

		IncrementKernelCache(KernelCacheItersSinceUsed,RowsInKernelCache);

		*(KernelCacheItersSinceUsed.begin()+CacheDiffI)=0;
		*(KernelCacheItersSinceUsed.begin()+CacheDiffJ)=0;



		iter++;

	}

	

	
	hipMemcpy(mexalpha, d_alpha, m*sizeof(float), hipMemcpyDeviceToHost);
	
	hipEventRecord(stop,0);
	hipEventSynchronize(stop);
	hipEventElapsedTime(elapsed, start, stop);


	stop_ct1 = std::chrono::high_resolution_clock::now();
    
    	//stop.wait_and_throw();
    	float duration = std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();
    	printf("Total run time: %f seconds\n", duration/1000.00); 

	printf("Iter:%i\n", iter);
	printf("M:%i\n", m);
	printf("N:%i\n", n);
        printf("Train done. Calulate Vector counts.\n"); 


	mexPutVariable("base","cuSVMTrainTimeInMS",mexelapsed);

	delete [] xT;
	delete [] h_alpha;
	delete [] h_F;
	delete[] kernelDotProd;
	
	#ifdef DISABLE_CUDA_PINNED_ALLOC
	delete [] value_inter;
	delete [] index_inter;
	#else
	hipHostFree(value_inter);
	hipHostFree(index_inter);
	#endif
	
	
	mxCUDA_SAFE_CALL(hipFree(d_x));
	mxCUDA_SAFE_CALL(hipFree(d_y));
	mxCUDA_SAFE_CALL(hipFree(d_alpha));
	mxCUDA_SAFE_CALL(hipFree(d_KernelInterRow));
	mxCUDA_SAFE_CALL(hipFree(d_Kernel_Cache));
	mxCUDA_SAFE_CALL(hipFree(d_F));
	mxCUDA_SAFE_CALL(hipFree(d_value_inter));
	mxCUDA_SAFE_CALL(hipFree(d_index_inter));
	mxCUDA_SAFE_CALL(hipFree(d_SelfDotProd));
    mxCUDA_SAFE_CALL(hipFree(d_KernelDotProd));
	mxCUDA_SAFE_CALL(hipDeviceReset());
	return;
}


