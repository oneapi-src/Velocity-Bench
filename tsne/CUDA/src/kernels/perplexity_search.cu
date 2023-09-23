/* Modifications Copyright (C) 2023 Intel Corporation
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *   may be used to endorse or promote products derived from this software
 *   without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
 * OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
 * OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 * OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

/*
    Compute the unnormalized pij matrix given a squared distance matrix and a target perplexity.
    pij = exp(-beta * dist ** 2)

    Note that FAISS returns the first row as the same point, with distance = 0. pii is defined as zero.
*/

#include "include/kernels/perplexity_search.h"

__global__
void ComputePijKernel(
    volatile float* __restrict__ pij,
    const    float* __restrict__ squared_dist,
    const    float* __restrict__ betas,
    const unsigned int num_points,
    const unsigned int num_neighbors)
{
    int tid, i, j;
    float dist, beta;

    tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_points * num_neighbors)
        return;

    i = tid / num_neighbors;
    j = tid % num_neighbors;

    beta = betas[i];
    dist = squared_dist[tid];

    // condition deals with evaluation of pii
    // FAISS neighbor zero is i so ignore it
    pij[tid] = (j == 0 & dist == 0.0f) ? 0.0f : __expf(-beta * dist); //TODO: This probably never evaluates to true
}

__global__
void PerplexitySearchKernel(
    volatile float* __restrict__ betas,
    volatile float* __restrict__ lower_bound,
    volatile float* __restrict__ upper_bound,
    volatile int*   __restrict__ found,
    const    float* __restrict__ neg_entropy,
    const    float* __restrict__ row_sum,
    const float perplexity_target,  // 50.0f
    const float epsilon,            // 1e-4
    const int num_points)
{
    int tid, is_found;
    float perplexity, neg_ent, sum_P, perplexity_diff, beta, min_beta, max_beta;

    tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_points)
        return;

    neg_ent  = neg_entropy[tid];
    sum_P    = row_sum[tid];
    beta     = betas[tid];
    min_beta = lower_bound[tid];
    max_beta = upper_bound[tid];

    perplexity      = (neg_ent / sum_P) + __logf(sum_P);
    perplexity_diff = perplexity - __logf(perplexity_target);
    is_found        = (perplexity_diff < epsilon && -perplexity_diff < epsilon);
    if (!is_found)
    {
        if (perplexity_diff > 0)
        {
            min_beta = beta;
            beta = (max_beta == FLT_MAX || max_beta == -FLT_MAX) ? beta * 2.0f : (beta + max_beta) / 2.0f;
        }
        else
        {
            max_beta = beta;
            beta = (min_beta == -FLT_MAX || min_beta == FLT_MAX) ? beta / 2.0f : (beta + min_beta) / 2.0f;
        }
        betas[tid] = beta;
        lower_bound[tid] = min_beta;
        upper_bound[tid] = max_beta;
    }
    found[tid] = is_found;
}

void tsnecuda::SearchPerplexity(
    thrust::device_vector<float>& pij,          // output array
    thrust::device_vector<float>& squared_dist, // input array
    const float perplexity_target,
    const float epsilon,
    const int num_points,
    const int num_neighbors,
    cublasHandle_t& handle)
{
    // use beta instead of sigma (this matches the bhtsne code but not the paper)
    // beta is just multiplicative instead of divisive (changes the way binary search works)
    thrust::device_vector<float> betas(num_points, 1.0f);
    thrust::device_vector<float> lower_bound_beta(num_points, -FLT_MAX);
    thrust::device_vector<float> upper_bound_beta(num_points,  FLT_MAX);
    thrust::device_vector<float> entropy(num_points * num_neighbors);
    thrust::device_vector<int>   found(num_points);

    // TODO: this doesn't really fit with the style
    const int BLOCKSIZE1 = 1024;
    const int NBLOCKS1 = iDivUp(num_points * num_neighbors, BLOCKSIZE1);

    const int BLOCKSIZE2 = 128;
    const int NBLOCKS2 = iDivUp(num_points, BLOCKSIZE2);

    size_t iters  = 0;
    int all_found = 0;

    thrust::device_vector<float> row_sum;
    thrust::device_vector<float> neg_entropy;

    do {
        // compute Gaussian Kernel row
        ComputePijKernel<<<NBLOCKS1, BLOCKSIZE1>>>(
            thrust::raw_pointer_cast(pij.data()),           // output
            thrust::raw_pointer_cast(squared_dist.data()),
            thrust::raw_pointer_cast(betas.data()),
            num_points,
            num_neighbors
        );
        GpuErrorCheck(cudaDeviceSynchronize());

        // compute entropy of current row
        row_sum = tsnecuda::utils::ReduceSum(handle, pij, num_neighbors, num_points, 0);
        // GpuErrorCheck(cudaDeviceSynchronize());

        // compute negative entropy
        thrust::transform(pij.begin(), pij.end(), entropy.begin(), tsnecuda::utils::FunctionalEntropy());
        neg_entropy = tsnecuda::utils::ReduceAlpha(handle, entropy, num_neighbors, num_points, -1.0f, 0);
        // GpuErrorCheck(cudaDeviceSynchronize());

        // binary search for beta
        PerplexitySearchKernel<<<NBLOCKS2, BLOCKSIZE2>>>(
            thrust::raw_pointer_cast(betas.data()),
            thrust::raw_pointer_cast(lower_bound_beta.data()),
            thrust::raw_pointer_cast(upper_bound_beta.data()),
            thrust::raw_pointer_cast(found.data()),
            thrust::raw_pointer_cast(neg_entropy.data()),
            thrust::raw_pointer_cast(row_sum.data()),
            perplexity_target,
            epsilon,
            num_points
        );
        GpuErrorCheck(cudaDeviceSynchronize());

        // Check if searching is done
        all_found = thrust::reduce(found.begin(), found.end(), 1, thrust::minimum<int>());

        iters++;
    } while (!all_found && iters < 200);
    // TODO: Warn if iters == 200 because perplexity not found?

    tsnecuda::utils::BroadcastMatrixVector(pij, row_sum, num_neighbors, num_points, thrust::divides<float>(), 1, 1.0f);
}
