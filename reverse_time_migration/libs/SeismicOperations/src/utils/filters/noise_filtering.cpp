/*
 * Modifications Copyright (C) 2023 Intel Corporation
 *
 * This Program is subject to the terms of the GNU Lesser General Public License v3.0 or later
 * 
 * If a copy of the license was not distributed with this file, you can obtain one at 
 * https://www.gnu.org/licenses/lgpl-3.0-standalone.html
 * 
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */


//
// Created by amr on 24/06/2020.
//

#include "operations/utils/filters/noise_filtering.h"

#include <cmath>
#include <iostream>
#include <cmath>

namespace operations {
    namespace utils {
        namespace filters {

            float laplace_3x3[9] = {
                    1, 1, 1,
                    1, -8, 1,
                    1, 1, 1};
            float laplace_9x9[81] = {
                    0, 1, 1, 2, 2, 2, 1, 1, 0,
                    1, 2, 4, 5, 5, 5, 4, 2, 1,
                    1, 4, 5, 3, 0, 3, 5, 4, 1,
                    2, 5, 3, -12, -24, -12, 3, 5, 2,
                    2, 5, 0, -24, -40, -24, 0, 5, 2,
                    2, 5, 3, -12, -24, -12, 3, 5, 2,
                    1, 4, 5, 3, 0, 3, 5, 4, 1,
                    1, 2, 4, 5, 5, 5, 4, 2, 1,
                    0, 1, 1, 2, 2, 2, 1, 1, 0,
            };
            float sobel_x[9] = {
                    1, 0, -1,
                    2, 0, -2,
                    1, 0, -1};

            float sobel_y[9] = {
                    1, 2, 1,
                    0, 0, 0,
                    -1, -2, -1};

            float bandpass_3x3[9] = {
                    -1, 2, -1,
                    2, -4, 2,
                    -1, 2, -1,
            };

            float gauss_5x5[25] = {
                    1, 4, 7, 4, 1,
                    4, 16, 26, 16, 4,
                    7, 25, 41, 25, 7,
                    4, 16, 26, 16, 4,
                    1, 4, 7, 4, 1,
            };

            void convolution(const float *mat, float *dst,
                             const float *kernel, int kernel_size, uint nz, uint nx) {
                int hl = kernel_size / 2;
                for (uint row = hl; row < nz - hl; row++) {
                    for (uint col = hl; col < nx - hl; col++) {
                        float value = 0;
                        for (int i = 0; i < kernel_size; i++) {
                            for (int j = 0; j < kernel_size; j++) {
                                value += kernel[i * kernel_size + j]
                                         * mat[(row + i - hl)
                                               * nx + (col + j - hl)];
                            }
                        }
                        dst[row * nx + col] = value;
                    }
                }
            }

            void magnitude(const float *Gx, const float *Gy, float *G, uint size) {
                for (uint idx = 0; idx < size; idx++) {
                    G[idx] = std::sqrt((Gx[idx] * Gx[idx]) + (Gy[idx] * Gy[idx]));
                }
            }

            void row_normalize(float *data, float *output, size_t nx, size_t nz) {
                /* normalizes each row so that the new value is ranging from 0 to 1, it skips the boundaries in seek for min and max value */
                for (int iz = 0; iz < nz; iz++) { // normalize for each row (i.e. for each depth)
                    float min = data[iz * nx];
                    float max = min;
                    for (int ix = 0; ix < nx; ix++) { // finds the max and min per row
                        min = fminf(min, data[iz * nx + ix]);
                        max = fmaxf(max, data[iz * nx + ix]);
                    }
                    float range = max - min;
                    for (int ix = 0; ix <
                                     nx; ix++) { // subtract min value and divide by max-min, so that new value ranges from 0 to 1 (except for boundaries)
                        if (range != 0) {
                            output[iz * nx + ix] = (data[iz * nx + ix] - min) / range;
                        } else {
                            output[iz * nx + ix] = 0;
                        }
                    }
                }
            }

            void z_normalize(float *input, float *output, uint nz, uint nx) {
                float sum = 0.0;
                float mean;
                float standardDeviation = 0.0;
                uint item_count = nx * nz;

                for (int i = 0; i < item_count; ++i) {
                    sum += input[i];
                }

                mean = sum / item_count;
                printf("mean = %f\n", mean);

                for (int i = 0; i < item_count; ++i) {
                    standardDeviation += std::pow(input[i] - mean, 2);
                }

                standardDeviation = std::sqrt(standardDeviation / item_count);

                printf("Standard Deviation\t: %f\n", standardDeviation);
                for (int i = 0; i < item_count; i++) {
                    output[i] = (input[i] - mean) / standardDeviation;
                }

            }

            void normalize(float *input, uint size, float max, float min) {
                for (int i = 0; i < size; i++) {
                    input[i] = 1023 * ((input[i] - min) / (max - min));
                }
            }

            void denormalize(float *input, uint size, float max, float min) {
                for (int i = 0; i < size; i++) {
                    input[i] = (input[i] / 1023) * (max - min) + min;
                }
            }

            void histogram_equalize(float *input, float *output, uint nz, uint nx) {
                uint item_count = nx * nz;
                float min = input[0];
                float max = min;

                float hist[1024] = {0};
                float new_gray_level[1024] = {0};
                float cumulative_frequency = 0;


                for (int i = 0; i < item_count; ++i) {
                    min = fminf(min, input[i]);
                    max = fmaxf(max, input[i]);
                }

                // scale to 0-1023
                normalize(input, item_count, max, min);

                for (int i = 0; i < item_count; i++) {
                    hist[(int) input[i]]++;
                }

                for (int i = 0; i < 1024; i++) {
                    cumulative_frequency += hist[i];
                    new_gray_level[i] = std::round(
                            ((cumulative_frequency) * 1023) / float(item_count));
                }

                for (int i = 0; i < item_count; i++) {
                    output[i] = new_gray_level[(int) input[i]];
                }

                denormalize(output, item_count, max, min);

            }

            // AGC inspired by the work done on the old RTM code by Karim/Essam/Bassiony.
            void apply_agc(const float *original, float *agc_trace,
                           int nx, int ny, int nz, int window) {
                // do signal gain using a window to the left of each sample, and to the right of each sample.
                // (left,right) here refer to each trace.
                window = std::min(window, nz - 1);
                // For each trace.
                for (unsigned int iy = 0; iy < ny; iy++) {
                    for (unsigned int ix = 0; ix < nx; ix++) {
                        uint trace_offset = iy * nx * nz + ix;
                        // Actual window size.
                        int window_size = (window + 1);
                        float sum = 0.0f;
                        // Initial start.
                        for (int iz = 0; iz <= window; iz++) {
                            float value = original[trace_offset + iz * nx];
                            sum += (value * value);
                        }
                        float rms = sum / float(window_size);
                        if (rms > 0.0f) {
                            agc_trace[trace_offset] = original[trace_offset] / sqrtf(rms);
                        } else {
                            agc_trace[trace_offset] = 0.0f;
                        }
                        // Window not full at left side.
                        for (int iz = 1; iz <= window; iz++) {
                            if (iz + window < nz) {
                                uint new_trace_offset = trace_offset + ((iz + window) * nx);
                                float value = original[new_trace_offset];
                                window_size++;
                                sum += (value * value);
                                rms = sum / float(window_size);
                            }
                            if (rms > 0.0f) {
                                agc_trace[trace_offset + iz * nx] = original[trace_offset + iz * nx] / sqrtf(rms);
                            } else {
                                agc_trace[trace_offset + iz * nx] = 0.0f;
                            }
                        }
                        // Window on both sides.
                        for (int iz = window + 1; iz <= nz - window; iz++) {
                            float value = original[trace_offset + (iz + window) * nx];
                            sum += (value * value);
                            value = original[trace_offset + (iz - window - 1) * nx];
                            sum -= (value * value);
                            rms = sum / float(window_size);
                            if (rms > 0.0f) {
                                agc_trace[trace_offset + iz * nx] = original[trace_offset + iz * nx] / sqrtf(rms);
                            } else {
                                agc_trace[trace_offset + iz * nx] = 0.0f;
                            }
                        }
                        if (nz > window + 1) {
                            // Window not full at right side.
                            for (int iz = nz - window + 1; iz < nz; iz++) {
                                int before_iz = iz - window - 1;
                                if (before_iz < 0) {
                                    float value = original[trace_offset + (iz - window - 1) * nx];
                                    sum -= (value * value);
                                    window_size--;
                                    rms = sum / float(window_size);
                                }
                                if (rms > 0.0f) {
                                    agc_trace[trace_offset + iz * nx] = original[trace_offset + iz * nx] / sqrtf(rms);
                                } else {
                                    agc_trace[trace_offset + iz * nx] = 0.0f;
                                }
                            }
                        }
                    }
                }
            }

            void filter_stacked_correlation(float *input_frame, float *output_frame,
                                            uint nx, uint ny, uint nz,
                                            float dx, float dz, float dy) {
                //auto *temp = new float[nx * nz * ny];
                convolution(input_frame, output_frame, laplace_9x9, 9, nz, nx);
                //apply_agc(temp, output_frame, nx, ny, nz, 500);
                //delete[] temp;
            }

            void apply_laplace_filter(float *input_frame, float *output_frame,
                                      unsigned int nx, unsigned int ny, unsigned int nz) {
                convolution(input_frame, output_frame, laplace_9x9, 9, nz, nx);
            }
        } //namespace filters
    } //namespace utils
} //namespace operations