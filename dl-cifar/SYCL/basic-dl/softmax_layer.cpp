/* Copyright (C) 2023 Intel Corporation
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom
 * the Software is furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
 * OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
 * OR OTHER DEALINGS IN THE SOFTWARE.
 * 
 * SPDX-License-Identifier: MIT
 */

#include "softmax_layer.h"

SoftmaxLayer::SoftmaxLayer(LangHandle *langHandle, Timer* timer,
     int softmaxInputTensorDims[4], int softmaxOutputTensorDims[4],
     float *d_input, float *d_d_input, float *d_output, float *d_d_output)
     : langHandle_(langHandle), timer_(timer), d_input_(d_input), d_d_input_(d_d_input), 
       d_output_(d_output), d_d_output_(d_d_output) {

#if defined(USE_CUBLAS)
    softmaxInputTensorDims_ = softmaxInputTensorDims;
    softmaxOutputTensorDims_ = softmaxOutputTensorDims;

    softmaxInputStrideDims_[3] = 1;
    softmaxInputStrideDims_[2] = softmaxInputTensorDims_[3];
    softmaxInputStrideDims_[1] = softmaxInputTensorDims_[2] * softmaxInputTensorDims_[3];
    softmaxInputStrideDims_[0] = softmaxInputTensorDims_[1] * softmaxInputTensorDims_[2] * softmaxInputTensorDims_[3];

    softmaxOutputStrideDims_[3] = 1;
    softmaxOutputStrideDims_[2] = softmaxOutputTensorDims_[3];
    softmaxOutputStrideDims_[1] = softmaxOutputTensorDims_[2] * softmaxOutputTensorDims_[3];
    softmaxOutputStrideDims_[0] = softmaxOutputTensorDims_[1] * softmaxOutputTensorDims_[2] * softmaxOutputTensorDims_[3];

    assertDlApiInvar(cudnnCreateTensorDescriptor(&softMaxInputDesc_));
    assertDlApiInvar(cudnnSetTensorNdDescriptor(softMaxInputDesc_, CUDNN_DATA_FLOAT, 4, 
                                                    softmaxInputTensorDims_, softmaxInputStrideDims_));

    assertDlApiInvar(cudnnCreateTensorDescriptor(&softMaxOutputDesc_));
    assertDlApiInvar(cudnnSetTensorNdDescriptor(softMaxOutputDesc_, CUDNN_DATA_FLOAT, 4, 
                                                    softmaxOutputTensorDims_, softmaxOutputStrideDims_));
#elif defined(USE_ROCBLAS) 
    softmaxInputTensorDims_ = softmaxInputTensorDims;
    softmaxOutputTensorDims_ = softmaxOutputTensorDims;

    softmaxInputStrideDims_[3] = 1;
    softmaxInputStrideDims_[2] = softmaxInputTensorDims_[3];
    softmaxInputStrideDims_[1] = softmaxInputTensorDims_[2] * softmaxInputTensorDims_[3];
    softmaxInputStrideDims_[0] = softmaxInputTensorDims_[1] * softmaxInputTensorDims_[2] * softmaxInputTensorDims_[3];

    softmaxOutputStrideDims_[3] = 1;
    softmaxOutputStrideDims_[2] = softmaxOutputTensorDims_[3];
    softmaxOutputStrideDims_[1] = softmaxOutputTensorDims_[2] * softmaxOutputTensorDims_[3];
    softmaxOutputStrideDims_[0] = softmaxOutputTensorDims_[1] * softmaxOutputTensorDims_[2] * softmaxOutputTensorDims_[3];

    assertDlApiInvar(miopenCreateTensorDescriptor(&softMaxInputDesc_));
    assertDlApiInvar(miopenSetTensorDescriptor(softMaxInputDesc_, miopenFloat, 4, 
                                                    softmaxInputTensorDims_, softmaxInputStrideDims_));

    assertDlApiInvar(miopenCreateTensorDescriptor(&softMaxOutputDesc_));
    assertDlApiInvar(miopenSetTensorDescriptor(softMaxOutputDesc_, miopenFloat, 4, 
                                                    softmaxOutputTensorDims_, softmaxOutputStrideDims_));

#else
    inputTensorDims   = softmaxInputTensorDims;   
    outputTensorDims  = softmaxOutputTensorDims;

    memSize   = inputTensorDims[0]   * inputTensorDims[1]   * inputTensorDims[2]   * inputTensorDims[3];


    dnnl::memory::format_tag formatTag = dnnl::memory::format_tag::nchw;
    using dt = memory::data_type;

    src_md = memory::desc({inputTensorDims[0], inputTensorDims[1], inputTensorDims[2], inputTensorDims[3]}, 
                                                                                dt::f32, formatTag /*tag::nc*/);
    dst_md = memory::desc({inputTensorDims[0], inputTensorDims[1], inputTensorDims[2], inputTensorDims[3]}, 
                                                                                dt::f32, formatTag /*tag::nc*/);
    src_mem = memory(src_md, *(langHandle_->getEngine()), d_input_);
    dy_mem = memory(src_md, *(langHandle_->getEngine()), d_d_output_);

    const int axis = 1;

    softmax_fw_pd = softmax_forward::primitive_desc(*(langHandle_->getEngine()),
            prop_kind::forward_training, algorithm::softmax_accurate, src_md,
            dst_md, axis);

    softmax_fw_prim = softmax_forward(softmax_fw_pd);

    softmax_bw_pd = softmax_backward::primitive_desc(*(langHandle_->getEngine()),
            algorithm::softmax_accurate, src_md,
            dst_md, dst_md, axis, softmax_fw_pd);

    softmax_bw_prim = softmax_backward(softmax_bw_pd);
#endif
}

void SoftmaxLayer::doFw() {  
#if defined(USE_CUBLAS)

    SYCL::ExecNativeCommand(*langHandle_->getSyclQueue(), [=](sycl::interop_handle ih) {
            cuCtxSetCurrent(ih.get_native_context<sycl::backend::ext_oneapi_cuda>());
            cublasSetStream(*(langHandle_->getCublasHandle()), ih.get_native_queue<sycl::backend::ext_oneapi_cuda>());

            //auto cudaStreamHandle = sycl::get_native<sycl::backend::ext_oneapi_cuda>(*(langHandle->getSyclQueue()));
            //cublasSetStream(*(langHandle->getCublasHandle()), cudaStreamHandle);
            //auto cuA = reinterpret_cast<float *>(ih.get_mem<sycl::backend::ext_oneapi_cuda>(d_A));
            //constexpr int INCX = 1;
            assertDlApiInvar(cudnnSoftmaxForward(*(langHandle_->getCudnnHandle()),
                                    CUDNN_SOFTMAX_ACCURATE,
                                    CUDNN_SOFTMAX_MODE_CHANNEL,
                                    (void*)(&alpha),
                                    softMaxInputDesc_,
                                    d_input_,
                                    (void*)(&beta),
                                    softMaxOutputDesc_,
                                    d_output_));
            //cublasDestroy(handle);
            //cudaStreamSynchronize(cudaStreamHandle);
        }, []{assertDevApiInvar(cudaDeviceSynchronize())});
#elif defined(USE_ROCBLAS) 
    SYCL::ExecNativeCommand(*langHandle_->getSyclQueue(), [=](sycl::interop_handle ih) {
            //cuCtxSetCurrent(ih.get_native_context<sycl::backend::ext_oneapi_cuda>());
            //cublasSetStream(*(langHandle_->getCublasHandle()), ih.get_native_queue<sycl::backend::ext_oneapi_cuda>());

            //auto cudaStreamHandle = sycl::get_native<sycl::backend::ext_oneapi_cuda>(*(langHandle->getSyclQueue()));
            //cublasSetStream(*(langHandle->getCublasHandle()), cudaStreamHandle);
            //auto cuA = reinterpret_cast<float *>(ih.get_mem<sycl::backend::ext_oneapi_cuda>(d_A));
            //constexpr int INCX = 1;
            assertDlApiInvar(miopenSoftmaxForward(*(langHandle_->getMiopenHandle()),
                                    (void*)(&alpha),
                                    softMaxInputDesc_,
                                    d_input_,
                                    (void*)(&beta),
                                    softMaxOutputDesc_,
                                    d_output_));
            //cublasDestroy(handle);
            //cudaStreamSynchronize(cudaStreamHandle);
        }, []{assertDevApiInvar(hipDeviceSynchronize())});
#else
    std::unordered_map<int, memory> softmax_args;
    softmax_args.insert({DNNL_ARG_SRC, src_mem});
    softmax_args.insert({DNNL_ARG_DST, src_mem});

    softmax_fw_prim.execute(*(langHandle_->getStream()), softmax_args);
    langHandle_->getStream()->wait();
#endif    
}

void SoftmaxLayer::doBw() {
#if defined(USE_CUBLAS)

    SYCL::ExecNativeCommand(*langHandle_->getSyclQueue(), [=](sycl::interop_handle ih) {
            cuCtxSetCurrent(ih.get_native_context<sycl::backend::ext_oneapi_cuda>());
            cublasSetStream(*(langHandle_->getCublasHandle()), ih.get_native_queue<sycl::backend::ext_oneapi_cuda>());

            //auto cudaStreamHandle = sycl::get_native<sycl::backend::ext_oneapi_cuda>(*(langHandle->getSyclQueue()));
            //cublasSetStream(*(langHandle->getCublasHandle()), cudaStreamHandle);
            //auto cuA = reinterpret_cast<float *>(ih.get_mem<sycl::backend::ext_oneapi_cuda>(d_A));
            //constexpr int INCX = 1;
            assertDlApiInvar(cudnnSoftmaxBackward(*(langHandle_->getCudnnHandle()),
                                    CUDNN_SOFTMAX_ACCURATE,
                                    CUDNN_SOFTMAX_MODE_CHANNEL,
                                    (void*)(&alpha),
                                    softMaxOutputDesc_,
                                    d_output_,
                                    softMaxOutputDesc_,
                                    d_d_output_,
                                    (void*)(&beta),
                                    softMaxInputDesc_,
                                    d_d_input_));
            //cublasDestroy(handle);
            //cudaStreamSynchronize(cudaStreamHandle);
        }, []{assertDevApiInvar(cudaDeviceSynchronize())});
#elif defined(USE_ROCBLAS)
    SYCL::ExecNativeCommand(*langHandle_->getSyclQueue(), [=](sycl::interop_handle ih) {
            //cuCtxSetCurrent(ih.get_native_context<sycl::backend::ext_oneapi_cuda>());
            //cublasSetStream(*(langHandle_->getCublasHandle()), ih.get_native_queue<sycl::backend::ext_oneapi_cuda>());

            //auto cudaStreamHandle = sycl::get_native<sycl::backend::ext_oneapi_cuda>(*(langHandle->getSyclQueue()));
            //cublasSetStream(*(langHandle->getCublasHandle()), cudaStreamHandle);
            //auto cuA = reinterpret_cast<float *>(ih.get_mem<sycl::backend::ext_oneapi_cuda>(d_A));
            //constexpr int INCX = 1;
            assertDlApiInvar(miopenSoftmaxBackward(*(langHandle_->getMiopenHandle()),
                                    (void*)(&alpha),
                                    softMaxOutputDesc_,
                                    d_output_,
                                    softMaxOutputDesc_,
                                    d_d_output_,
                                    (void*)(&beta),
                                    softMaxInputDesc_,
                                    d_d_input_));
            //cublasDestroy(handle);
            //cudaStreamSynchronize(cudaStreamHandle);
        }, []{assertDevApiInvar(hipDeviceSynchronize())});
#else    
    std::unordered_map<int, memory> softmax_args;
    softmax_args.insert({DNNL_ARG_SRC, src_mem});
    softmax_args.insert({DNNL_ARG_DST, src_mem});
    softmax_args.insert({DNNL_ARG_DIFF_SRC, dy_mem});
    softmax_args.insert({DNNL_ARG_DIFF_DST, dy_mem});

    softmax_bw_prim.execute(*(langHandle_->getStream()), softmax_args);
    langHandle_->getStream()->wait();
#endif       
}

SoftmaxLayer::~SoftmaxLayer() {

}
