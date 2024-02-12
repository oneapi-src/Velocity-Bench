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

#include "hip/hip_runtime.h"
#include "include/kernels/nbodyfft.h"

const float PI = 3.14159265358979f;
// const float twoPI = 3.14159265358979f * 2;

// #define HIP_CHECK_LAST_ERROR() {                                          \
//     if( hipSuccess != hipGetLastError()) {                                \
//         fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",     \
//             __FILE__, __LINE__, hipGetErrorString( hipGetLastError() ) ); \
//         exit(EXIT_FAILURE);                                               \
//     } }

#define BS1 16
#define BS2 16
/*
#define TWIDDLE()                                   \
    sinf = sin(angle * k);                          \
    cosf = cos(angle * k);                          \
    twiddle = thrust::complex<float>(cosf, sinf);
*/

/*
#define TWIDDLE()                                   \
    sincosf(angle * k, &sinf, &cosf);               \
    twiddle = thrust::complex<float>(cosf, sinf);
*/

#define TWIDDLE()                                   \
    sinf = __sinf(angle * k);                       \
    cosf = __cosf(angle * k);                       \
    twiddle = thrust::complex<float>(cosf, sinf);

__global__
void copy_to_fft_input(
    volatile float* __restrict__ fft_input,
    const    float* w_coefficients_device,
    const int n_fft_coeffs,
    const int n_fft_coeffs_half,
    const int n_terms)
{
    int i, j;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n_terms * n_fft_coeffs_half * n_fft_coeffs_half)
        return;

    int current_term = tid / (n_fft_coeffs_half * n_fft_coeffs_half);
    int current_loc  = tid % (n_fft_coeffs_half * n_fft_coeffs_half);

    i = current_loc / n_fft_coeffs_half;
    j = current_loc % n_fft_coeffs_half;

    fft_input[current_term * (n_fft_coeffs * n_fft_coeffs) + i * n_fft_coeffs + j] = w_coefficients_device[current_term + current_loc * n_terms];
}

__global__
void copy_from_fft_output(
    volatile float* __restrict__ y_tilde_values,
    const    float* fft_output,
    const int n_fft_coeffs,
    const int n_fft_coeffs_half,
    const int n_terms)
{
    int i, j;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n_terms * n_fft_coeffs_half * n_fft_coeffs_half)
        return;

    int current_term = tid / (n_fft_coeffs_half * n_fft_coeffs_half);
    int current_loc  = tid % (n_fft_coeffs_half * n_fft_coeffs_half);

    i = current_loc / n_fft_coeffs_half + n_fft_coeffs_half;
    j = current_loc % n_fft_coeffs_half + n_fft_coeffs_half;

    y_tilde_values[current_term + n_terms * current_loc] = fft_output[current_term * (n_fft_coeffs * n_fft_coeffs) + i * n_fft_coeffs + j] / (float)(n_fft_coeffs * n_fft_coeffs);
}

__global__
void compute_point_box_idx(
    volatile int*   __restrict__ point_box_idx,
    volatile float* __restrict__ x_in_box,
    volatile float* __restrict__ y_in_box,
    const float* const xs,
    const float* const ys,
    const float* const box_lower_bounds,
    const float min_coord,
    const float box_width,
    const int n_boxes,
    const int n_total_boxes,
    const int N)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= N)
        return;

    int x_idx = (int)((xs[tid] - min_coord) / box_width);
    int y_idx = (int)((ys[tid] - min_coord) / box_width);

    x_idx = max(0, x_idx);
    x_idx = min((int)(n_boxes - 1), x_idx);

    y_idx = max(0, y_idx);
    y_idx = min((int)(n_boxes - 1), y_idx);

    int box_idx = y_idx * n_boxes + x_idx;
    point_box_idx[tid] = box_idx;

    x_in_box[tid] = (xs[tid] - box_lower_bounds[box_idx])                 / box_width;
    y_in_box[tid] = (ys[tid] - box_lower_bounds[n_total_boxes + box_idx]) / box_width;
}

__global__
void interpolate_device(
    volatile float* __restrict__ interpolated_values,
    const    float* const y_in_box,
    const    float* const y_tilde_spacings,
    const    float* const denominator,
    const int n_interpolation_points,
    const int N)
{
    int tid, i, j, k;
    float value, ybox_i;

    tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= N * n_interpolation_points)
        return;

    i = tid % N;
    j = tid / N;

    value = 1;
    ybox_i = y_in_box[i];

    for (k = 0; k < n_interpolation_points; k++) {
        if (j != k) {
            value *= ybox_i - y_tilde_spacings[k];
        }
    }

    interpolated_values[j * N + i] = value / denominator[j];
}

__global__
void compute_interpolated_indices(
          float* __restrict__ w_coefficients_device,
    const int*   const point_box_indices,
    const float* const chargesQij,
    const float* const x_interpolated_values,
    const float* const y_interpolated_values,
    const int N,
    const int n_interpolation_points,
    const int n_boxes,
    const int n_terms)
{
    int tid, current_term, i, interp_i, interp_j, box_idx, box_i, box_j, idx;
    tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n_terms * n_interpolation_points * n_interpolation_points * N)
        return;

    current_term = tid % n_terms;
    i = (tid / n_terms) % N;
    interp_j = ((tid / n_terms) / N) % n_interpolation_points;
    interp_i = ((tid / n_terms) / N) / n_interpolation_points;

    box_idx = point_box_indices[i];
    box_i = box_idx % n_boxes;
    box_j = box_idx / n_boxes;

    idx = (box_i * n_interpolation_points  + interp_i) * (n_boxes * n_interpolation_points) +
          (box_j * n_interpolation_points) + interp_j;

    atomicAdd(
        w_coefficients_device + idx * n_terms + current_term,
        x_interpolated_values[i + interp_i * N] * y_interpolated_values[i + interp_j * N] * chargesQij[i * n_terms + current_term]);
}

__global__
void compute_potential_indices(
          float* __restrict__ potentialsQij,
    const int*   const point_box_indices,
    const float* const y_tilde_values,
    const float* const x_interpolated_values,
    const float* const y_interpolated_values,
    const int N,
    const int n_interpolation_points,
    const int n_boxes,
    const int n_terms)
{
    int tid, current_term, i, interp_i, interp_j, box_idx, box_i, box_j, idx;
    tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n_terms * n_interpolation_points * n_interpolation_points * N)
        return;

    current_term = tid % n_terms;
    i = (tid / n_terms) % N;
    interp_j = ((tid / n_terms) / N) % n_interpolation_points;
    interp_i = ((tid / n_terms) / N) / n_interpolation_points;

    box_idx = point_box_indices[i];
    box_i = box_idx % n_boxes;
    box_j = box_idx / n_boxes;

    idx = (box_i * n_interpolation_points + interp_i) * (n_boxes * n_interpolation_points) +
          (box_j * n_interpolation_points) + interp_j;

    atomicAdd(
        potentialsQij + i * n_terms + current_term,
        x_interpolated_values[i + interp_i * N] * y_interpolated_values[i + interp_j * N] * y_tilde_values[idx * n_terms + current_term]);
}

__host__ __device__
float squared_cauchy_2d(float x1, float x2, float y1, float y2)
{
    return pow(1.0f + pow(x1 - y1, 2) + pow(x2 - y2, 2), -2);
}

__global__
void compute_kernel_tilde(
    volatile float* __restrict__ kernel_tilde,   // 780 x 780
    const    float x_min,
    const    float y_min,
    const    float h,
    const    int   n_interpolation_points_1d,    // 390
    const    int   n_fft_coeffs)                 // 390 x 2 = 780
{
    int tid, i, j;
    float tmp;
    tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n_interpolation_points_1d * n_interpolation_points_1d)
        return;

    i = tid / n_interpolation_points_1d;
    j = tid % n_interpolation_points_1d;

    // TODO: Possibly issuing a memory pre-fetch here could help the code.
    tmp = squared_cauchy_2d(y_min + h / 2, x_min + h / 2, y_min + h / 2 + i * h, x_min + h / 2 + j * h);
    kernel_tilde[(n_interpolation_points_1d + i) * n_fft_coeffs + (n_interpolation_points_1d + j)] = tmp;
    kernel_tilde[(n_interpolation_points_1d - i) * n_fft_coeffs + (n_interpolation_points_1d + j)] = tmp;
    kernel_tilde[(n_interpolation_points_1d + i) * n_fft_coeffs + (n_interpolation_points_1d - j)] = tmp;
    kernel_tilde[(n_interpolation_points_1d - i) * n_fft_coeffs + (n_interpolation_points_1d - j)] = tmp;
}

__global__
void compute_upper_and_lower_bounds(
    volatile float* __restrict__ box_upper_bounds,
    volatile float* __restrict__ box_lower_bounds,
    const    float box_width,
    const    float x_min,
    const    float y_min,
    const    int   n_boxes,
    const    int   n_total_boxes)
{
    int tid, i, j;
    tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n_boxes * n_boxes)
        return;

    i = tid / n_boxes;
    j = tid % n_boxes;

    box_lower_bounds[i * n_boxes + j] =  j      * box_width + x_min;
    box_upper_bounds[i * n_boxes + j] = (j + 1) * box_width + x_min;

    box_lower_bounds[n_total_boxes + i * n_boxes + j] =  i      * box_width + y_min;
    box_upper_bounds[n_total_boxes + i * n_boxes + j] = (i + 1) * box_width + y_min;
}

// real to complex
__global__
void DFT2D1gpu(float* din, thrust::complex<float>* dout, int num_rows, int num_cols)
{
	int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= num_rows || i >= num_cols) {
        return;
    }
    
    float angle, cosf, sinf; 
    thrust::complex<float> sum, twiddle;
    angle = -2.0f * PI * fdividef((float)i, (float)num_cols);
    sum = 0.0f;
#pragma unroll
    for (int k = 0; k < num_cols; ++k) {
        // sincosf(angle * k, &sinf, &cosf);
        // twiddle = thrust::complex<float>(cosf, sinf);
        TWIDDLE();
        sum = sum + din[j * num_cols + k] * twiddle;
    }

    dout[i * num_rows + j] = sum;
}

// complex to complex
__global__
void DFT2D2gpu(thrust::complex<float>* din, thrust::complex<float>* dout, int num_rows, int num_cols)
{
	int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= num_rows || i >= num_cols) {
        return;
    }
    
    float angle, cosf, sinf;
    thrust::complex<float> sum, twiddle;
    angle = -2.0f * PI * fdividef((float)i, (float)num_cols);
    sum = 0.0f;
#pragma unroll
    for (int k = 0; k < num_cols; ++k) {
        // sincosf(angle * k, &sinf, &cosf);
        // twiddle = thrust::complex<float>(cosf, sinf);
        TWIDDLE();
        sum = sum + din[j * num_cols + k] * twiddle;
    }

    dout[i * num_rows + j] = sum;
}

// complex to complex
__global__
void iDFT2D1gpu(thrust::complex<float>* din, thrust::complex<float>* dout, int num_rows, int num_cols)
{
	int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= num_rows || i >= num_cols) {
        return;
    }
    
    float angle, cosf, sinf; 
    thrust::complex<float> sum, twiddle;
    angle = 2.0f * PI * fdividef((float)i, (float)num_cols);
    sum = 0.0f;
#pragma unroll
    for (int k = 0; k < num_cols/2 + 1; ++k) {
        // sincosf(angle * k, &sinf, &cosf);
        // twiddle = thrust::complex<float>(cosf, sinf);
        TWIDDLE();
        sum = sum + din[j * (num_cols/2+1) + k] * twiddle;
    }
#pragma unroll
    for (int k = num_cols/2 + 1; k < num_cols; ++k) {
        TWIDDLE();
        sum = sum + thrust::conj(din[((num_rows-j)%num_rows) * (num_cols/2+1) + ((num_cols-k)%num_cols)]) * twiddle;
    }

    dout[i * num_rows + j] = sum;
}

// complex to real
__global__
void iDFT2D2gpu(thrust::complex<float>* din, float* dout, int num_rows, int num_cols)
{
	int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= num_rows || i >= num_cols) {
        return;
    }
    
    float angle, sum, cosf, sinf;
    thrust::complex<float> twiddle;
    angle = 2.0f * PI * fdividef((float)i, (float)num_cols);
    sum = 0.0f;
#pragma unroll
    for (int k = 0; k < num_cols; ++k) {
        // sincosf(angle * k, &sinf, &cosf);
        // twiddle = thrust::complex<float>(cosf, sinf);
        TWIDDLE();
        sum = sum + (din[j * num_cols + k] * twiddle).real();
    }

    dout[i * num_rows + j] = sum;
}

void tsnecuda::PrecomputeFFT2D(
    // hipfftHandle& plan_kernel_tilde,
    float  x_max,
    float  x_min,
    float  y_max,
    float  y_min,
    int    n_boxes,
    int    n_interpolation_points,
    thrust::device_vector<float>& box_lower_bounds_device,
    thrust::device_vector<float>& box_upper_bounds_device,
    thrust::device_vector<float>& kernel_tilde_device,
    thrust::device_vector<thrust::complex<float>>& fft_kernel_tilde_device,
    thrust::device_vector<thrust::complex<float>>& fft_scratchpad_device, double& duration)   // added
{
    const int num_threads = 32;
    int num_blocks = (n_boxes * n_boxes + num_threads - 1) / num_threads;
    /*
     * Set up the boxes
     */
    int n_total_boxes = n_boxes * n_boxes;
    float box_width   = (x_max - x_min) / (float)n_boxes;

    // Left and right bounds of each box, first the lower bounds in the x direction, then in the y direction
    hipLaunchKernelGGL(compute_upper_and_lower_bounds, num_blocks, num_threads, 0, 0, 
        thrust::raw_pointer_cast(box_upper_bounds_device.data()),
        thrust::raw_pointer_cast(box_lower_bounds_device.data()),
        box_width,
        x_min,
        y_min,
        n_boxes,
        n_total_boxes
    );  // wait() not needed as this and next kernels can run simultaneously
    HIP_CHECK_LAST_ERROR()

    // Coordinates of all the equispaced interpolation points
    int n_interpolation_points_1d = n_interpolation_points    * n_boxes;
    int n_fft_coeffs              = n_interpolation_points_1d * 2;

    float h = box_width / (float)n_interpolation_points;

    /*
     * Evaluate the kernel at the interpolation nodes and form the embedded generating kernel vector for a circulant
     * matrix
     */
    num_blocks = (n_interpolation_points_1d * n_interpolation_points_1d + num_threads - 1) / num_threads;
    hipLaunchKernelGGL(compute_kernel_tilde, num_blocks, num_threads, 0, 0, 
        thrust::raw_pointer_cast(kernel_tilde_device.data()),
        x_min,
        y_min,
        h,
        n_interpolation_points_1d,
        n_fft_coeffs
    );
    HIP_CHECK_LAST_ERROR()
    GpuErrorCheck(hipDeviceSynchronize());

    // // Precompute the FFT of the kernel generating matrix
    // hipfftExecR2C(plan_kernel_tilde,
    //              reinterpret_cast<hipfftReal *>(thrust::raw_pointer_cast(kernel_tilde_device.data())),
    //              reinterpret_cast<hipfftComplex *>(thrust::raw_pointer_cast(fft_kernel_tilde_device.data())));

    int num_rows = n_interpolation_points * n_boxes * 2;
    int num_cols = num_rows;

    dim3 block_size(BS1, BS2);
    dim3 grid_size1((num_cols + block_size.x - 1) / block_size.x, (num_rows       + block_size.y - 1) / block_size.y);
    dim3 grid_size2((num_rows + block_size.x - 1) / block_size.x, ((num_cols/2+1) + block_size.y - 1) / block_size.y);

    DFT2D1gpu <<<grid_size1, block_size>>> (
        reinterpret_cast<float*>(thrust::raw_pointer_cast(kernel_tilde_device.data())),
        reinterpret_cast<thrust::complex<float>*>(thrust::raw_pointer_cast(fft_scratchpad_device.data())),
        num_rows,
        num_cols);
    HIP_CHECK_LAST_ERROR();
    GpuErrorCheck(hipDeviceSynchronize());

    DFT2D2gpu <<<grid_size2, block_size>>> (
        reinterpret_cast<thrust::complex<float>*>(thrust::raw_pointer_cast(fft_scratchpad_device.data())),
        reinterpret_cast<thrust::complex<float>*>(thrust::raw_pointer_cast(fft_kernel_tilde_device.data())),
        (num_cols/2+1),
        num_rows);
    HIP_CHECK_LAST_ERROR();
    GpuErrorCheck(hipDeviceSynchronize());
}

void tsnecuda::NbodyFFT2D(
    // hipfftHandle& plan_dft,
    // hipfftHandle& plan_idft,
    thrust::device_vector<thrust::complex<float>>& fft_kernel_tilde_device,
    thrust::device_vector<thrust::complex<float>>& fft_w_coefficients,
    int    N,
    int    n_terms,
    int    n_boxes,
    int    n_interpolation_points,
    int    n_total_boxes,
    int    total_interpolation_points,
    float  min_coord,
    float  box_width,
    int    n_fft_coeffs_half,
    int    n_fft_coeffs,
    thrust::device_vector<float>& fft_input,
    thrust::device_vector<float>& fft_output,
    thrust::device_vector<int>&   point_box_idx_device,
    thrust::device_vector<float>& x_in_box_device,
    thrust::device_vector<float>& y_in_box_device,
    thrust::device_vector<float>& points_device,
    thrust::device_vector<float>& box_lower_bounds_device,
    thrust::device_vector<float>& y_tilde_spacings_device,
    thrust::device_vector<float>& denominator_device,
    thrust::device_vector<float>& y_tilde_values,
    // thrust::device_vector<float>& all_interpolated_values_device,
    // thrust::device_vector<float>& output_values,
    // thrust::device_vector<int>&   all_interpolated_indices,
    // thrust::device_vector<int>&   output_indices,
    thrust::device_vector<float>& w_coefficients_device,
    thrust::device_vector<float>& chargesQij_device,
    thrust::device_vector<float>& x_interpolated_values_device,
    thrust::device_vector<float>& y_interpolated_values_device,
    thrust::device_vector<float>& potentialsQij_device,
    thrust::device_vector<thrust::complex<float>>& fft_scratchpad_device, double& duration) // added
{
    const int num_threads = 128;
    int num_blocks = (N + num_threads - 1) / num_threads;

    // Compute box indices and the relative position of each point in its box in the interval [0, 1]
    hipLaunchKernelGGL(compute_point_box_idx, num_blocks, num_threads, 0, 0, 
        thrust::raw_pointer_cast(point_box_idx_device.data()),
        thrust::raw_pointer_cast(x_in_box_device.data()),
        thrust::raw_pointer_cast(y_in_box_device.data()),
        thrust::raw_pointer_cast(points_device.data()),
        thrust::raw_pointer_cast(points_device.data() + N),
        thrust::raw_pointer_cast(box_lower_bounds_device.data()),
        min_coord,
        box_width,
        n_boxes,
        n_total_boxes,
        N
    );
    HIP_CHECK_LAST_ERROR()
    GpuErrorCheck(hipDeviceSynchronize());

    /*
     * Step 1: Interpolate kernel using Lagrange polynomials and compute the w coefficients
     */
    // TODO: We can stream-parallelize these two interpolation functions
    // Compute the interpolated values at each real point with each Lagrange polynomial in the `x` direction
    num_blocks = (N * n_interpolation_points + num_threads - 1) / num_threads;
    hipLaunchKernelGGL(interpolate_device, num_blocks, num_threads, 0, 0, 
        thrust::raw_pointer_cast(x_interpolated_values_device.data()),
        thrust::raw_pointer_cast(x_in_box_device.data()),
        thrust::raw_pointer_cast(y_tilde_spacings_device.data()),
        thrust::raw_pointer_cast(denominator_device.data()),
        n_interpolation_points,
        N
    );
    HIP_CHECK_LAST_ERROR()
    GpuErrorCheck(hipDeviceSynchronize()); // TODO: Remove the synchronization here

    // Compute the interpolated values at each real point with each Lagrange polynomial in the `y` direction
    hipLaunchKernelGGL(interpolate_device, num_blocks, num_threads, 0, 0, 
        thrust::raw_pointer_cast(y_interpolated_values_device.data()),
        thrust::raw_pointer_cast(y_in_box_device.data()),
        thrust::raw_pointer_cast(y_tilde_spacings_device.data()),
        thrust::raw_pointer_cast(denominator_device.data()),
        n_interpolation_points,
        N
    );
    HIP_CHECK_LAST_ERROR()
    GpuErrorCheck(hipDeviceSynchronize());

    //TODO: Synchronization required here

    // TODO: This section has an atomic-add, can we remove it?
    num_blocks = (n_terms * n_interpolation_points * n_interpolation_points * N + num_threads - 1) / num_threads;
    hipLaunchKernelGGL(compute_interpolated_indices, num_blocks, num_threads, 0, 0,
        thrust::raw_pointer_cast(w_coefficients_device.data()),
        thrust::raw_pointer_cast(point_box_idx_device.data()),
        thrust::raw_pointer_cast(chargesQij_device.data()),
        thrust::raw_pointer_cast(x_interpolated_values_device.data()),
        thrust::raw_pointer_cast(y_interpolated_values_device.data()),
        N,
        n_interpolation_points,
        n_boxes,
        n_terms
    );
    HIP_CHECK_LAST_ERROR()
    GpuErrorCheck(hipDeviceSynchronize());

    /*
     * Step 2: Compute the values v_{m, n} at the equispaced nodes, multiply the kernel matrix with the coefficients w
     */
    num_blocks = ((n_terms * n_fft_coeffs_half * n_fft_coeffs_half) + num_threads - 1) / num_threads;
    hipLaunchKernelGGL(copy_to_fft_input, num_blocks, num_threads, 0, 0, 
        thrust::raw_pointer_cast(fft_input.data()),
        thrust::raw_pointer_cast(w_coefficients_device.data()),
        n_fft_coeffs,
        n_fft_coeffs_half,
        n_terms
    );
    HIP_CHECK_LAST_ERROR()
    GpuErrorCheck(hipDeviceSynchronize());

    // Compute fft values at interpolated nodes
    // hipfftExecR2C(plan_dft,
    //              reinterpret_cast<hipfftReal *>(thrust::raw_pointer_cast(fft_input.data())),
    //              reinterpret_cast<hipfftComplex *>(thrust::raw_pointer_cast(fft_w_coefficients.data())));
    // GpuErrorCheck(hipDeviceSynchronize());

    int num_rows = n_fft_coeffs;
    int num_cols = n_fft_coeffs;

    dim3 block_size(BS1, BS2);
    dim3 grid_size1((num_cols + block_size.x - 1) / block_size.x, (num_rows       + block_size.y - 1) / block_size.y);
    dim3 grid_size2((num_rows + block_size.x - 1) / block_size.x, ((num_cols/2+1) + block_size.y - 1) / block_size.y);

    float* din  = reinterpret_cast<float*>(thrust::raw_pointer_cast(fft_input.data()));
    thrust::complex<float>* temp = reinterpret_cast<thrust::complex<float>*>(thrust::raw_pointer_cast(fft_scratchpad_device.data()));
    thrust::complex<float>* dout = reinterpret_cast<thrust::complex<float>*>(thrust::raw_pointer_cast(fft_w_coefficients.data()));

    for (int f = 0; f < n_terms; ++f) {
        DFT2D1gpu <<<grid_size1, block_size>>> (
            din  + f * num_rows * num_cols,
            temp + f * num_rows * num_cols,
            num_rows,
            num_cols
        );
        HIP_CHECK_LAST_ERROR();
        GpuErrorCheck(hipDeviceSynchronize());

        DFT2D2gpu <<<grid_size2, block_size>>> (
            temp + f * num_rows * num_cols,
            dout + f * num_rows * (num_cols/2+1),
            (num_cols/2+1),
            num_rows
        );
        HIP_CHECK_LAST_ERROR();
        GpuErrorCheck(hipDeviceSynchronize());
    }

    // Take the broadcasted Hadamard product of a complex matrix and a complex vector
    // TODO: Check timing on this kernel
    tsnecuda::utils::BroadcastMatrixVector(
        fft_w_coefficients,                     // 4 x 780 x (780 / 2 + 1) = 4 x 304980 = 1219920 (input/output)
        fft_kernel_tilde_device,                //     780 x  780 = 608400                        (input)
        n_fft_coeffs * (n_fft_coeffs / 2 + 1),  //     780 x (780 / 2 + 1) = 304980
        n_terms,                                // 4
        thrust::multiplies<thrust::complex<float>>(),
        0,
        thrust::complex<float>(1.0f));

    // Invert the computed values at the interpolated nodes
    // hipfftExecC2R(plan_idft,
    //              reinterpret_cast<hipfftComplex *>(thrust::raw_pointer_cast(fft_w_coefficients.data())),
    //              reinterpret_cast<hipfftReal *>(thrust::raw_pointer_cast(fft_output.data())));
    // GpuErrorCheck(hipDeviceSynchronize());

    din  = reinterpret_cast<float*>(thrust::raw_pointer_cast(fft_output.data()));

    for (int f = 0; f < n_terms; ++f) {
        iDFT2D1gpu <<<grid_size1, block_size>>> (
            dout + f * num_rows * (num_cols/2+1),
            temp + f * num_rows * num_cols,
            num_rows,
            num_cols
        );
        HIP_CHECK_LAST_ERROR();
        GpuErrorCheck(hipDeviceSynchronize());

        iDFT2D2gpu <<<grid_size1, block_size>>> (
            temp + f * num_rows * num_cols,
            din  + f * num_rows * num_cols,
            num_cols,
            num_rows
        );
        HIP_CHECK_LAST_ERROR();
        GpuErrorCheck(hipDeviceSynchronize());
    }

    hipLaunchKernelGGL(copy_from_fft_output, num_blocks, num_threads, 0, 0, 
        thrust::raw_pointer_cast(y_tilde_values.data()),    // output
        thrust::raw_pointer_cast(fft_output.data()),
        n_fft_coeffs,
        n_fft_coeffs_half,
        n_terms
    );
    HIP_CHECK_LAST_ERROR()
    GpuErrorCheck(hipDeviceSynchronize());

    /*
     * Step 3: Compute the potentials \tilde{\phi}
     */
    // TODO: Depending on the profiling here, we should check to see if we can split this code
    num_blocks = (n_terms * n_interpolation_points * n_interpolation_points * N + num_threads - 1) / num_threads;
    hipLaunchKernelGGL(compute_potential_indices, num_blocks, num_threads, 0, 0,
        thrust::raw_pointer_cast(potentialsQij_device.data()),          // output
        thrust::raw_pointer_cast(point_box_idx_device.data()),
        thrust::raw_pointer_cast(y_tilde_values.data()),
        thrust::raw_pointer_cast(x_interpolated_values_device.data()),
        thrust::raw_pointer_cast(y_interpolated_values_device.data()),
        N,
        n_interpolation_points,
        n_boxes,
        n_terms
    );
    HIP_CHECK_LAST_ERROR()
    GpuErrorCheck(hipDeviceSynchronize());
}
