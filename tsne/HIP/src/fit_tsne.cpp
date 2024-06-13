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
    Compute t-SNE via Barnes-Hut for NlogN time.
*/

#include <chrono>
#include "include/fit_tsne.h"
#include "verify.hpp"

// #ifndef DEBUG_TIME
// #define DEBUG_TIME
// #endif

#define TIMER_START_() time_start_ = std::chrono::steady_clock::now();
#define TIMER_END_()                                                                         \
    time_end_ = std::chrono::steady_clock::now();                                            \
    time_total_  += std::chrono::duration<double, std::milli>(time_end_ - time_start_).count();
#define TIMER_PRINT_(name) std::cout << name <<": " << time_total_ / 1e3 << " s\n";

#ifdef DEBUG_TIME
// #define TIME_START() time_start = std::chrono::steady_clock::now();
// #define TIME_SINCE(x)                                                       \
//     std::cout << "\nTime passed: "                                          \
//               << std::chrono::duration_cast<std::chrono::milliseconds>(     \
//                   std::chrono::steady_clock::now() - x).count()             \
//               << " ms\n";

#define START_IL_TIMER() start = std::chrono::steady_clock::now();
#define END_IL_TIMER(x)                                                             \
    stop = std::chrono::steady_clock::now();                                        \
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start); \
    x += duration;                                                                  \
    total_time += duration;
#define PRINT_IL_TIMER(x) std::cout << #x << ": " << ((float)x.count()) / 1000000.0 << "s" << std::endl
#endif

double tsnecuda::RunTsne(tsnecuda::Options& opt, int& success)
{
    std::chrono::steady_clock::time_point time_start_;
    std::chrono::steady_clock::time_point time_end_;
    double time_total_ = 0.0;

#ifdef DEBUG_TIME
    auto start = std::chrono::steady_clock::now();
    auto stop  = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    auto total_time             = duration;
    auto _time_initialization   = duration;
    auto _time_knn              = duration;
    auto _time_symmetry         = duration;
    auto _time_init_low_dim     = duration;
    auto _time_init_fft         = duration;
    auto _time_precompute_2d    = duration;
    auto _time_nbodyfft         = duration;
    auto _time_compute_charges  = duration;
    auto _time_other            = duration;
    auto _time_repl             = duration;
    auto _time_attr             = duration;
    auto _time_apply_forces     = duration;
#endif

    // auto time_start             = std::chrono::steady_clock::now();

    // Check the validity of the options file
    if (!opt.validate()) {
        std::cout << "E: Invalid options file. Terminating." << std::endl;
        return 0.0;
    }

    if (opt.verbosity > 0) {
        std::cout << "Initializing hip handles... " << std::flush;
    }

    // Construct the handles
    // TODO: Move this outside of the timing code, since RAPIDs is cheating by pre-initializing the handle.
    // TODO: Allow for multi-stream on the computation, since we can overlap portions of our computation to be quicker.

    // // Setup some return information if we're working on snapshots
    // // TODO: Add compile flag to remove snapshotting for timing parity
    // int snap_num = 0;
    // int snap_interval = 1;
    // if (opt.return_style == tsnecuda::RETURN_STYLE::SNAPSHOT)
    // {
    //     snap_interval = opt.iterations / (opt.num_snapshots - 1);
    // }

    // Get constants from options
    const int num_points            = opt.num_points;   // number of images
    const int high_dim              = opt.num_dims;     // number of pixels per image

    // TODO: Warn if the number of neighbors is more than the number of points
    const int num_neighbors                 = (opt.num_neighbors < num_points) ? opt.num_neighbors : num_points;
    const float perplexity                  = opt.perplexity;
    const float perplexity_search_epsilon   = opt.perplexity_search_epsilon;
    const float eta                         = opt.learning_rate;
    float momentum                          = opt.pre_exaggeration_momentum;
    float attr_exaggeration                 = opt.early_exaggeration;
    float normalization                     = 0.0f;

    // Allocate host memory
    // TODO: Pre-determine GPU/CPU memory requirements, since we will know them ahead of time, and can estimate
    // if you're going to run out of GPU memory
    // TODO: Investigate what it takes to use unified memory + Async fetch and execution
    long*  knn_indices      = new long[ num_points * num_neighbors];
    float* knn_distances    = new float[num_points * num_neighbors];
    memset(knn_distances, 0.0f, num_points * num_neighbors * sizeof(float));

    if (opt.verbosity > 0) {
        std::cout << "done.\nKNN Load...\n" << std::flush;
    }

    TIMER_START_()
    // Compute approximate K Nearest Neighbors and squared distances
    // TODO: See if we can gain some time here by updating FAISS, and building better indicies
    // TODO: Add suport for arbitrary metrics on GPU (Introduced by recent FAISS computation)
    // TODO: Expose Multi-GPU computation (+ Add streaming memory support for GPU optimization)
    std::string data_folder = "../../data/mnist_faissed/";
    // std::string data_folder = "../../data/cifar10_faissed/";
    tsnecuda::utils::KNearestNeighbors(
        std::move(data_folder), // folder containing input files
        knn_indices,            // *** output indices   ***
        knn_distances,          // *** output distances ***
        high_dim,               // number of pixels per image = 784
        num_points,             // number of images
        num_neighbors);
    TIMER_END_()

#ifdef DEBUG_TIME
    START_IL_TIMER();
#endif

    GpuErrorCheck(hipSetDevice(0));

    hipblasHandle_t dense_handle;
    CublasSafeCall(hipblasCreate(&dense_handle));

#ifdef DEBUG_TIME
    END_IL_TIMER(_time_initialization);
#endif

#ifdef DEBUG_TIME
    START_IL_TIMER();
#endif

    thrust::device_vector<long> knn_indices_device(knn_indices, knn_indices + num_points * num_neighbors);
    thrust::device_vector<int>  pij_indices_device(num_points * num_neighbors);

    tsnecuda::utils::PostprocessNeighborIndices(    // pij_indices_device[i] = (int)knn_indices_device
        pij_indices_device,     // output
        knn_indices_device,     // input
        num_points,
        num_neighbors);

    // Max-norm the distances to avoid exponentiating by large numbers
    thrust::device_vector<float> knn_distances_device(knn_distances, knn_distances + num_points * num_neighbors);
    tsnecuda::utils::MaxNormalizeDeviceVector(      // divide by max abs value
        knn_distances_device);           // input and output

#ifdef DEBUG_TIME
    END_IL_TIMER(_time_knn);
#endif

    if (opt.verbosity > 0) {
        std::cout << "done.\nComputing Pij matrix... " << std::endl;
    }

#ifdef DEBUG_TIME
    START_IL_TIMER();
#endif

    // Search Perplexity
    thrust::device_vector<float> pij_nonsymmetric_device(num_points * num_neighbors);
    tsnecuda::SearchPerplexity(
        pij_nonsymmetric_device,    // output
        knn_distances_device,       // input
        perplexity,
        perplexity_search_epsilon,
        num_points,
        num_neighbors,
        dense_handle);
    GpuErrorCheck(hipDeviceSynchronize());

#ifdef DEBUG_TIME
    END_IL_TIMER(_time_symmetry);
#endif

    // Clean up memory
    knn_indices_device.clear();
    knn_indices_device.shrink_to_fit();
    knn_distances_device.clear();
    knn_distances_device.shrink_to_fit();
    delete[] knn_indices;
    delete[] knn_distances;

#ifdef DEBUG_TIME
    START_IL_TIMER();
#endif

    // Symmetrize the pij matrix
    thrust::device_vector<float> pij_symmetric_device(num_points * num_neighbors);
    tsnecuda::utils::SymmetrizeMatrixV2(
        pij_symmetric_device,       // output
        pij_nonsymmetric_device,    // input
        pij_indices_device,         // input
        num_points,
        num_neighbors);
    GpuErrorCheck(hipDeviceSynchronize());

#ifdef DEBUG_TIME
    END_IL_TIMER(_time_symmetry);
#endif

    // Clean up memory
    pij_nonsymmetric_device.clear();
    pij_nonsymmetric_device.shrink_to_fit();

#ifdef DEBUG_TIME
    START_IL_TIMER();
#endif

    // Declare memory
    thrust::device_vector<float> attractive_forces_device(opt.num_points * 2);
    thrust::device_vector<float> repulsive_forces_device( opt.num_points * 2);
    thrust::device_vector<float> gains_device(            opt.num_points * 2);
    thrust::device_vector<float> old_forces_device(       opt.num_points * 2);
    thrust::device_vector<float> ones_device(             opt.num_points * 2);
    thrust::device_vector<float> normalization_vec_device(opt.num_points);
    thrust::device_vector<float> pij_workspace_device(num_points * num_neighbors * 2);

    thrust::fill(gains_device.begin(),      gains_device.begin()      + opt.num_points * 2, 1.0f);
    thrust::fill(old_forces_device.begin(), old_forces_device.begin() + opt.num_points * 2, 0.0f);
    thrust::fill(ones_device.begin(),       ones_device.begin()       + opt.num_points * 2, 1.0f);

#ifdef DEBUG_TIME
    END_IL_TIMER(_time_symmetry);
#endif

    if (opt.verbosity > 0) {
        std::cout << "done.\nInitializing low dim points... " << std::flush;
    }

    thrust::host_vector<float> points_host(num_points * 2);

    TIMER_START_()
    if (opt.initialization == tsnecuda::TSNE_INIT::GAUSSIAN) { // Random gaussian initialization
        std::ifstream points_file;
        points_file.open("../../data/points.txt");
        if (!points_file) std::cerr << "Can't open points.txt!";
        if (points_file.is_open()) {
            int i = 0;
            while (!points_file.eof()) {
                points_file >> points_host[i++];
            }
            points_file.close();
            i--;
            if (i != num_points * 2) {
                std::cout << "Number of data points read: " << i << std::endl;
                std::cout << "That is incorrect\n";
                exit(1);
            }
        } else {
            std::cout << "Can't read points file\n";
            exit(1);
        }
    } else { // Invalid initialization
        std::cerr << "E: Invalid initialization type specified." << std::endl;
        exit(1);
    }
    TIMER_END_()

#ifdef DEBUG_TIME
    START_IL_TIMER();
#endif

    // Initialize Low-Dim Points
    thrust::device_vector<float> points_device(num_points * 2);
    thrust::copy(points_host.begin(), points_host.end(), points_device.begin());

#ifdef DEBUG_TIME
    END_IL_TIMER(_time_init_low_dim);
#endif

    points_host.clear();
    points_host.shrink_to_fit();

    if (opt.verbosity > 0) {
        std::cout << "done.\nInitializing HIP memory... " << std::flush;
    }

    // FIT-TNSE Parameters
    int n_terms = 4;
    int n_interp_points = 3;
    int n_boxes_per_dim = 125;

    // FFTW works faster on numbers that can be written as  2^a 3^b 5^c 7^d
    // 11^e 13^f, where e+f is either 0 or 1, and the other exponents are
    // arbitrary
    int allowed_n_boxes_per_dim[21] = {25, 36, 50, 55, 60, 65, 70, 75, 80, 85, 90, 96, 100, 110, 120, 130, 140, 150, 175, 200, 1125};
    if (n_boxes_per_dim < allowed_n_boxes_per_dim[20]) { //Round up to nearest grid point
        int chosen_i;
        for (chosen_i = 0; allowed_n_boxes_per_dim[chosen_i] < n_boxes_per_dim; chosen_i++)
            ;
        n_boxes_per_dim = allowed_n_boxes_per_dim[chosen_i];
    }

    int n_total_boxes       = n_boxes_per_dim * n_boxes_per_dim;
    int total_interp_points = n_interp_points * n_interp_points * n_total_boxes;
    int n_fft_coeffs_half   = n_interp_points * n_boxes_per_dim;
    int n_fft_coeffs        = n_interp_points * n_boxes_per_dim * 2;
    // int n_interp_points_1d  = n_interp_points * n_boxes_per_dim;
    int N = num_points;

#ifdef DEBUG_TIME
    START_IL_TIMER();
#endif

    // FIT-TSNE Device Vectors
    thrust::device_vector<int>   point_box_idx_device(N);
    thrust::device_vector<float> x_in_box_device(N);
    thrust::device_vector<float> y_in_box_device(N);
    thrust::device_vector<float> y_tilde_values(total_interp_points * n_terms);
    thrust::device_vector<float> x_interpolated_values_device(N * n_interp_points);
    thrust::device_vector<float> y_interpolated_values_device(N * n_interp_points);
    thrust::device_vector<float> potentialsQij_device(N * n_terms);
    // thrust::device_vector<float> all_interpolated_values_device(n_terms * n_interp_points * n_interp_points * N);
    // thrust::device_vector<float> output_values(n_terms * n_interp_points * n_interp_points * N);
    // thrust::device_vector<int>   all_interpolated_indices(n_terms * n_interp_points * n_interp_points * N);
    // thrust::device_vector<int>   output_indices(n_terms * n_interp_points * n_interp_points * N);
    thrust::device_vector<float> w_coefficients_device(total_interp_points * n_terms);
    thrust::device_vector<float> chargesQij_device(N * n_terms);
    thrust::device_vector<float> box_lower_bounds_device(2 * n_total_boxes);
    thrust::device_vector<float> box_upper_bounds_device(2 * n_total_boxes);

    thrust::device_vector<float> kernel_tilde_device(                     n_fft_coeffs * n_fft_coeffs);
    thrust::device_vector<thrust::complex<float>> fft_kernel_tilde_device(n_fft_coeffs * n_fft_coeffs);

    thrust::device_vector<thrust::complex<float>> fft_scratchpad_device(  n_fft_coeffs * n_fft_coeffs       * n_terms); // added

    thrust::device_vector<float> fft_input(                               n_fft_coeffs * n_fft_coeffs       * n_terms);
    thrust::device_vector<thrust::complex<float>> fft_w_coefficients(     n_fft_coeffs * (n_fft_coeffs/2+1) * n_terms);
    thrust::device_vector<float> fft_output(                              n_fft_coeffs * n_fft_coeffs       * n_terms);

#ifdef DEBUG_TIME
    END_IL_TIMER(_time_init_fft);
#endif

    // Easier to compute denominator on CPU, so we should just calculate y_tilde_spacing on CPU also
    float h = 1 / (float)n_interp_points;
    float y_tilde_spacings[n_interp_points];
    y_tilde_spacings[0] = h / 2;
    for (int i = 1; i < n_interp_points; i++) {
        y_tilde_spacings[i] = y_tilde_spacings[i - 1] + h;
    }
    float denominator[n_interp_points];
    for (int i = 0; i < n_interp_points; i++) {
        denominator[i] = 1;
        for (int j = 0; j < n_interp_points; j++) {
            if (i != j) {
                denominator[i] *= y_tilde_spacings[i] - y_tilde_spacings[j];
            }
        }
    }

#ifdef DEBUG_TIME
    START_IL_TIMER();
#endif

    thrust::device_vector<float> y_tilde_spacings_device(y_tilde_spacings, y_tilde_spacings + n_interp_points);
    thrust::device_vector<float> denominator_device(denominator, denominator + n_interp_points);
    thrust::fill(fft_input.begin(), fft_input.begin() + n_fft_coeffs * n_fft_coeffs * n_terms, 0.0f);

#ifdef DEBUG_TIME
    END_IL_TIMER(_time_init_fft);
#endif

    if (opt.verbosity > 0) {
        std::cout << "done." << std::endl;
    }

    int fft_dimensions[2] = {n_fft_coeffs, n_fft_coeffs};        // {780, 780}
    size_t work_size_idft, work_size_dft;
    // size_t work_size;

    // std::cout << "Setting up dft plans...\n";
    // // *** TIMED SEPARATELY. NOT ADDED TO PERF TIME ***
    // TIME_START();
    // hipfftHandle plan_tilde;
    // CufftSafeCall(hipfftCreate(&plan_tilde));
    // CufftSafeCall(hipfftMakePlan2d(
    //     plan_tilde,
    //     fft_dimensions[0],
    //     fft_dimensions[1],
    //     HIPFFT_R2C,
    //     &work_size)
    // );
    // TIME_SINCE(time_start);

    // TIME_START();
    hipfftHandle plan_dft;
    CufftSafeCall(hipfftCreate(&plan_dft));
    CufftSafeCall(hipfftMakePlanMany(
        plan_dft,
        2,
        fft_dimensions,
        NULL,
        1,
        n_fft_coeffs * n_fft_coeffs,
        NULL,
        1,
        n_fft_coeffs * (n_fft_coeffs / 2 + 1),
        HIPFFT_R2C,
        n_terms,
        &work_size_dft)
    );
    // TIME_SINCE(time_start);

    // TIME_START();
    hipfftHandle plan_idft;
    CufftSafeCall(hipfftCreate(&plan_idft));
    CufftSafeCall(hipfftMakePlanMany(
        plan_idft,
        2,
        fft_dimensions,
        NULL,
        1,
        n_fft_coeffs * (n_fft_coeffs / 2 + 1),
        NULL,
        1,
        n_fft_coeffs * n_fft_coeffs,
        HIPFFT_C2R,
        n_terms,
        &work_size_idft)
    );
    // TIME_SINCE(time_start);
    // std::cout << "done.\n";

    // hipfftHandle plan_tilde;
    // hipfftHandle plan_dft;
    // hipfftHandle plan_idft;
    double duration_fft1 = 0.0, duration_fft2 = 0.0;

    // Support for infinite iteration
    for (size_t step = 0; step != opt.iterations; step++) {

#ifdef DEBUG_TIME
        START_IL_TIMER();
#endif
        // TODO: We might be able to write a kernel which does this more efficiently. It probably doesn't require much
        // TODO: but it could be done.
        thrust::fill(w_coefficients_device.begin(), w_coefficients_device.end(), 0.0f);
        thrust::fill(potentialsQij_device.begin(),  potentialsQij_device.end(),  0.0f);
#ifdef DEBUG_TIME
        END_IL_TIMER(_time_other);
#endif

        // Setup learning rate schedule
        if (step == opt.force_magnify_iters) {
            momentum = opt.post_exaggeration_momentum;
            attr_exaggeration = 1.0f;
        }

        // Prepare the terms that we'll use to compute the sum i.e. the repulsive forces
#ifdef DEBUG_TIME
        START_IL_TIMER();
#endif
        tsnecuda::ComputeChargesQij(
            chargesQij_device,      // output
            points_device,          // input
            num_points,
            n_terms);
#ifdef DEBUG_TIME
        END_IL_TIMER(_time_compute_charges);
#endif

        // Compute Minimax elements
#ifdef DEBUG_TIME
        START_IL_TIMER();
#endif
        auto minimax_iter = thrust::minmax_element(points_device.begin(), points_device.end());
        float min_coord = minimax_iter.first[0];
        float max_coord = minimax_iter.second[0];
#ifdef DEBUG_TIME
        END_IL_TIMER(_time_precompute_2d);
#endif

        float box_width = (max_coord - min_coord) / (float)n_boxes_per_dim;
        if (step < 30) {
            std::cout << "step: " << step << " min_coord: " << min_coord << " max_coord: " << max_coord << std::endl;
            std::cout << " box_width: " << box_width << std::endl;
        }

#ifdef DEBUG_TIME
        START_IL_TIMER();
#endif

        // Compute the number of boxes in a single dimension and the total number of boxes in 2d
        tsnecuda::PrecomputeFFT2D(
            // plan_tilde,
            max_coord,                  // input
            min_coord,                  // input
            max_coord,                  // input
            min_coord,                  // input
            n_boxes_per_dim,            // 130
            n_interp_points,            // 3
            box_lower_bounds_device,    // output: 2 * n_total_boxes size buffer [where n_total_boxes = 130 x 130]
            box_upper_bounds_device,    // output: 2 * n_total_boxes size buffer [where n_total_boxes = 130 x 130]
            kernel_tilde_device,        // output?: n_fft_coeffs * n_fft_coeffs size buffer [n_fft_coeffs = 2 x 3 x 130]
            fft_kernel_tilde_device,    // output:  n_fft_coeffs * n_fft_coeffs size buffer
            fft_scratchpad_device, duration_fft1);

#ifdef DEBUG_TIME
        END_IL_TIMER(_time_precompute_2d);
#endif

#ifdef DEBUG_TIME
        START_IL_TIMER();
#endif

        tsnecuda::NbodyFFT2D(
            plan_dft,
            plan_idft,
            fft_kernel_tilde_device,            // input
            fft_w_coefficients,                 // intermediate value
            N,
            n_terms,
            n_boxes_per_dim,
            n_interp_points,
            n_total_boxes,
            total_interp_points,
            min_coord,
            box_width,
            n_fft_coeffs_half,
            n_fft_coeffs,
            fft_input,                          // intermediate value
            fft_output,                         // intermediate value
            point_box_idx_device,               // intermediate value
            x_in_box_device,                    // intermediate value
            y_in_box_device,                    // intermediate value
            points_device,                      // input
            box_lower_bounds_device,            // input
            y_tilde_spacings_device,            // input (calculated outside the loop)
            denominator_device,                 // input (calculated outside the loop)
            y_tilde_values,                     // intermediate value
            // all_interpolated_values_device,
            // output_values,
            // all_interpolated_indices,
            // output_indices,
            w_coefficients_device,
            chargesQij_device,                  // input
            x_interpolated_values_device,       // intermediate value
            y_interpolated_values_device,       // intermediate value
            potentialsQij_device,               // intermediate value
            fft_scratchpad_device, duration_fft2);

#ifdef DEBUG_TIME
        END_IL_TIMER(_time_nbodyfft);
#endif

#ifdef DEBUG_TIME
        START_IL_TIMER();
#endif

        // TODO: We can overlap the computation of the attractive and repulsive forces, this requires changing the
        // TODO: default streams of the code in both of these methods
        // TODO: See: https://stackoverflow.com/questions/24368197/getting-cuda-thrust-to-use-a-cuda-stream-of-your-choice
        // Make the negative term, or F_rep in the equation 3 of the paper

        // Calculate Repulsive Forces
        normalization = tsnecuda::ComputeRepulsiveForces(
            repulsive_forces_device,    // num_points * 2                   (output: uninitialized)
            normalization_vec_device,   // num_points                       (output: uninitialized)
            points_device,              // num_points * 2                   (input: initially randomly generated)
            potentialsQij_device,       // N * n_terms = num_points * 4     (input)
            num_points,
            n_terms);

#ifdef DEBUG_TIME
        END_IL_TIMER(_time_repl);
#endif

#ifdef DEBUG_TIME
        START_IL_TIMER();
#endif

        // Calculate Attractive Forces
        tsnecuda::ComputeAttractiveForcesV3(
            attractive_forces_device,   // num_points * 2                   (output: uninitialized)
            pij_symmetric_device,       // num_points * num_neighbors       (input: calculated using SymmetrizeMatrixV2)
            pij_indices_device,         // num_points * num_neighbors       (input: calculated using PostprocessNeighborIndices)
            pij_workspace_device,       // num_points * num_neighbors * 2   (output)
            points_device,              // num_points * 2                   (input: initially randomly generated)
            ones_device,                // num_points * 2                   (input: all 1.0f)
            num_points,
            num_neighbors);

        GpuErrorCheck(hipDeviceSynchronize());

#ifdef DEBUG_TIME
        END_IL_TIMER(_time_attr);
#endif

#ifdef DEBUG_TIME
        START_IL_TIMER();
#endif

        // TODO: Add stream synchronization here.

        // Apply Forces
        tsnecuda::ApplyForces(
            points_device,              // num_points * 2                   (input/output: initially randomly generated)
            attractive_forces_device,   // num_points * 2                   (input/output: calculated using ComputeAttractiveForcesV3)
            repulsive_forces_device,    // num_points * 2                   (input/output: calculated using ComputeRepulsiveForces)
            gains_device,               // num_points * 2                   (input/output: all 1.0f)
            old_forces_device,          // num_points * 2                   (input/output: all 0.0f)
            eta,                        // scalar                           (learning rate)
            normalization,              // scalar                           (return value of ComputeRepulsiveForces)
            momentum,
            attr_exaggeration,
            num_points);

        // Compute the gradient norm
        float grad_norm = tsnecuda::utils::L2NormDeviceVector(old_forces_device);

#ifdef DEBUG_TIME
        END_IL_TIMER(_time_apply_forces);
#endif

        if (grad_norm < opt.min_gradient_norm) {
            if (opt.verbosity >= 1) {
                std::cout << "Reached minimum gradient norm: " << grad_norm << std::endl;
            }
            break;
        }

        if (opt.verbosity >= 1 && step % opt.print_interval == 0) {
            std::cout << "[Step " << step << "] Avg. Gradient Norm: " << grad_norm << std::endl;
        }
    } // End for loop
    // std::cout << "DFT2D1gpu : duration_fft2: " << duration_fft2 << " ms" << std::endl;
#ifdef DEBUG_TIME
    if (opt.verbosity > 0) {
        PRINT_IL_TIMER(_time_initialization);
        PRINT_IL_TIMER(_time_knn);
        PRINT_IL_TIMER(_time_symmetry);
        PRINT_IL_TIMER(_time_init_low_dim);
        PRINT_IL_TIMER(_time_init_fft);
        PRINT_IL_TIMER(_time_compute_charges);
        PRINT_IL_TIMER(_time_precompute_2d);
        PRINT_IL_TIMER(_time_nbodyfft);
        PRINT_IL_TIMER(_time_repl);
        PRINT_IL_TIMER(_time_attr);
        PRINT_IL_TIMER(_time_apply_forces);
        PRINT_IL_TIMER(_time_other);
        // PRINT_IL_TIMER(total_time);
    }
#endif

    // Write output - not timed
    if (opt.get_dump_points()) {
        thrust::host_vector<float> host_ys(num_points * 2);
        thrust::copy(points_device.begin(), points_device.end(), host_ys.begin());

        TIMER_START_()
        std::ofstream dump_file;
        dump_file.open(opt.get_dump_file());
        dump_file << num_points << " " << 2 << std::endl;
        
        for (int i = 0; i < opt.num_points; i++) {
            dump_file << host_ys[i] << " " << host_ys[i + num_points] << std::endl;
        }
        dump_file.close();

        std::string golden_file = "../../data/tsne_mnist_output_golden.txt";
        success = verify(golden_file, opt.get_dump_file(), 0.2, 10.0);
        TIMER_END_()

        host_ys.clear();
        host_ys.shrink_to_fit();
    }

    // Return some final values
    opt.trained = true;
    opt.trained_norm = normalization;

    // CufftSafeCall(hipfftDestroy(plan_tilde));
    // CufftSafeCall(hipfftDestroy(plan_dft));
    // CufftSafeCall(hipfftDestroy(plan_idft));

    TIMER_PRINT_("time to subtract from total")
    return time_total_;
}
