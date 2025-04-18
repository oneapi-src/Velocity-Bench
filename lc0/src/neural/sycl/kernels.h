/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2019 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

/*   This file is part of Leela Chess Zero.
    Modifications Copyright (C) 2023 Intel Corporation

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
   along with this program.  If not, see <https://www.gnu.org/licenses/>. 
   
   SPDX license identifier to be GNU General Public License v3.0 or later
*/

#include <sycl/sycl.hpp>
#include "sycl_common.h"

namespace lczero {
namespace cudnn_backend {

// Adds two vectors (possibly of different sizes), also do optional
// activation (relu, tanh or sigmoid).
template <typename T>
void addVectors(T* c, T* a, T* b, int size, int asize, int bsize, ActivationFunction activation, sycl::queue &mqueue);

// Optimized kernel to add bias to innermost dimension
// and perform optional activation (to be used with GEMMs/fully connected)
template <typename T>
void addBiasBatched(T* output, const T* input, const T* bias, int Batch, int N, int C, ActivationFunction activation, sycl::queue &mqueue);

// Add bias to convolution's output.
template <typename T>
void addBias_NCHW(T* c, T* a, T* b, int N, int C, int H, int W, ActivationFunction activation, sycl::queue &mqueue);

// Conversion from NCHW to NHWC, can also change datatype depending on template
// params, also pad/un-pad elements from Batch or Channel dimensions
template <typename DstType, typename SrcType>
void convertNCHWtoNHWC(DstType* output_tensor, const SrcType* input_tensor, int Nin, int Cin, int Nout, int Cout, int H, int W, sycl::queue &mqueue);

// Plain data-type conversion (no layout conversion).
template <typename DstType, typename SrcType> void copyTypeConverted(DstType* op, SrcType* ip, int N, sycl::queue &mqueue);

// Perform batch normilization.
template <typename T>
void batchNorm(T* output, const T* input, const T* skipInput, int N, int C, int H, int W, float* means, float* var_multipliers, ActivationFunction activation, sycl::queue &mqueue);

// Unpack planes (input to network).
void expandPlanes_Fp32_NCHW(float* output, const uint64_t* masks, const float* values, int n, sycl::queue &mqueue);

void expandPlanes_Fp16_NHWC(sycl::half* output, const uint64_t* masks, const float* values, int n, sycl::queue &mqueue);

void expandPlanes_Fp16_NCHW(sycl::half* output, const uint64_t* masks, const float* values, int n, sycl::queue &mqueue);

// Perform global avg pool.
template <typename T>
void globalAvgPool(int N, int C, T* output, const T* input, const T* prevLayerBias, bool nhwc, sycl::queue &q_ct1);

// Perform global scale.
template <typename T>
void globalScale(int N, int C, T* output, const T* input, const T* scaleBias, const T* prevLayerBias, bool nhwc, ActivationFunction activation, sycl::queue &q_ct1);

// Perform Squeeze-and-Excitation (SE) in a single fused kernel.
// Returns false if the fused kernel can't handle the sizes.
bool Se_Fp16_NHWC(int N, int C, int numFc1Out, sycl::half* output,
                  const sycl::half* skip, const sycl::half* input,
                  const sycl::half* w1, const sycl::half* b1,
                  const sycl::half* w2, const sycl::half* b2,
                  const sycl::half* bPrev, ActivationFunction activation);

template <typename T>
void PolicyMap(int N, T* output, const T* input, const short* indices,
               int inputSize, int usedSize, int outputSize,
               sycl::queue &q_ct1);

// Custom winograd helper functions
template <typename T>
void FilterTransform(int N, int C, T* transformedFilter, const T* filter, sycl::queue &mqueue);

template <typename T, bool nhcw>
void InputTransform(int N, int C, T* transformedInput, const T* input, sycl::queue &mqueue);

template <typename T, bool use_se, ActivationFunction activation, bool use_bias,
          bool use_skip, bool skipInput_nhcw, bool output_nhcw>
void OutputTransform(int N, int C, int se_K, T* output, const T* input,
                     const T* skip, const T* bias, const T* w1, const T* b1,
                     const T* w2, const T* b2, sycl::queue &stream);

template <typename T, bool use_se, ActivationFunction activation, bool use_bias,
          bool use_skip>
void OutputInputTransform(int N, int C, int se_K, T* output, const T* input,
                          const T* skip, const T* bias, const T* w1,
                          const T* b1, const T* w2, const T* b2,
                          sycl::queue &stream);

template <typename T>
void Softmax(int N, int C, T* output, const T* input, sycl::queue &mqueue);

template <typename T>
void LayerNorm(int N, int C, T* output, const T* input, const T* bias,
               const T* skip, const T* gammas, const T* betas, float ep,
               sycl::queue &mqueue);

template <typename T>
void ComputePromotionLogits(int N, int C, T* output, const T* keys,
                            const T* ppo, const T* policy_attn_logits,
                            sycl::queue &mqueue);

}  // namespace cudnn_backend
}  // namespace lczero
