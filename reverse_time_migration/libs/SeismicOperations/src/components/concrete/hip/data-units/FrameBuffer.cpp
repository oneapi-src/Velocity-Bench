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
// Created by ahmed-ayyad on 16/11/2020.
//

#include <operations/common/DataTypes.h>
#include <operations/data-units/concrete/holders/FrameBuffer.hpp>
/////#include <operations/backend/OneAPIBackend.hpp>
#include <hip/hip_runtime.h>
// #include <hip/hip_runtime.h>
#include "Logging.h"

using namespace operations::dataunits;
/////using namespace operations::backend;

template
class operations::dataunits::FrameBuffer<float>;

template
class operations::dataunits::FrameBuffer<int>;

template
class operations::dataunits::FrameBuffer<uint>;

template FrameBuffer<float>::FrameBuffer();

template FrameBuffer<float>::FrameBuffer(uint aSize);

template FrameBuffer<float>::~FrameBuffer();

template void FrameBuffer<float>::Allocate(uint size, const std::string &aName);

template void FrameBuffer<float>::Allocate(uint aSize, HALF_LENGTH aHalfLength, const std::string &aName);

template void FrameBuffer<float>::Free();

template float *FrameBuffer<float>::GetNativePointer();

template float *FrameBuffer<float>::GetHostPointer();

template void FrameBuffer<float>::SetNativePointer(float *ptr);

template void FrameBuffer<float>::ReflectOnNative();

template FrameBuffer<int>::FrameBuffer();

template FrameBuffer<int>::FrameBuffer(uint aSize);

template FrameBuffer<int>::~FrameBuffer();

template void FrameBuffer<int>::Allocate(uint size, const std::string &aName);

template void FrameBuffer<int>::Allocate(uint aSize, HALF_LENGTH aHalfLength, const std::string &aName);

template void FrameBuffer<int>::Free();

template int *FrameBuffer<int>::GetNativePointer();

template int *FrameBuffer<int>::GetHostPointer();

template void FrameBuffer<int>::SetNativePointer(int *ptr);

template void FrameBuffer<int>::ReflectOnNative();

template FrameBuffer<uint>::FrameBuffer();

template FrameBuffer<uint>::FrameBuffer(uint aSize);

template FrameBuffer<uint>::~FrameBuffer();

template void FrameBuffer<uint>::Allocate(uint size, const std::string &aName);

template void FrameBuffer<uint>::Allocate(uint aSize, HALF_LENGTH aHalfLength, const std::string &aName);

template void FrameBuffer<uint>::Free();

template uint *FrameBuffer<uint>::GetNativePointer();

template uint *FrameBuffer<uint>::GetHostPointer();

template void FrameBuffer<uint>::SetNativePointer(uint *ptr);

template void FrameBuffer<uint>::ReflectOnNative();

template<typename T>
FrameBuffer<T>::FrameBuffer() {
    mpDataPointer = nullptr;
    mpHostDataPointer = nullptr;
    mAllocatedBytes = 0;
}

template<typename T>
FrameBuffer<T>::FrameBuffer(uint aSize) {
    Allocate(aSize);
    mpHostDataPointer = nullptr;
}

template<typename T>
FrameBuffer<T>::~FrameBuffer() {
    Free();
}

template<typename T>
void FrameBuffer<T>::Allocate(uint aSize, const std::string &aName) {
    mAllocatedBytes = sizeof(T) * aSize;
    checkHIP(hipMalloc(reinterpret_cast<void**>(&mpDataPointer), sizeof(T) * aSize));
}

template<typename T>
void FrameBuffer<T>::Allocate(uint aSize, HALF_LENGTH aHalfLength, const std::string &aName) {
    mAllocatedBytes = sizeof(T) * aSize;
    checkHIP(hipMalloc(reinterpret_cast<void**>(&mpDataPointer), sizeof(T) * aSize));
}

template<typename T>
void FrameBuffer<T>::Free() {
    if (mpDataPointer != nullptr) {
        hipFree(mpDataPointer);
        mpDataPointer = nullptr;
        if (mpHostDataPointer != nullptr) {
            delete mpHostDataPointer;
            mpHostDataPointer = nullptr;
        }
    }
    mAllocatedBytes = 0;
}

template<typename T>
T *FrameBuffer<T>::GetNativePointer() {
    return mpDataPointer;
}

template<typename T>
T *FrameBuffer<T>::GetHostPointer() {
    if (mAllocatedBytes > 0) {
        if (mpHostDataPointer != nullptr) {
            delete(mpHostDataPointer);
            mpHostDataPointer = new T [mAllocatedBytes / sizeof(T)];
            assert(mpHostDataPointer != nullptr);
        }
        mpHostDataPointer = new T [mAllocatedBytes / sizeof(T)];
        assert(mpHostDataPointer != nullptr);
        Device::MemCpy(mpHostDataPointer, mpDataPointer, mAllocatedBytes, Device::COPY_DEVICE_TO_HOST);
        return mpHostDataPointer;
    }
    return mpHostDataPointer;
}

template<typename T>
void FrameBuffer<T>::SetNativePointer(T *pT) {
    mpDataPointer = pT;
}

template<typename T>
void FrameBuffer<T>::ReflectOnNative() {
    Device::MemCpy(mpDataPointer, mpHostDataPointer, mAllocatedBytes, Device::COPY_HOST_TO_DEVICE);
}


void Device::MemSet(void *apDst, int aVal, uint aSize) {
    checkHIP(hipMemset(reinterpret_cast<void*>(apDst), aVal, aSize));
}

void Device::MemCpy(void *apDst, const void *apSrc, uint aSize, CopyDirection aCopyDirection) {
    if (aCopyDirection == Device::COPY_HOST_TO_DEVICE) {
        checkHIP(hipMemcpy(apDst, apSrc, aSize, hipMemcpyHostToDevice)); 
    } else if (aCopyDirection == Device::COPY_DEVICE_TO_HOST) {
        checkHIP(hipMemcpy(apDst, apSrc, aSize, hipMemcpyDeviceToHost));
    } else if (aCopyDirection == Device::COPY_DEVICE_TO_DEVICE) {
        checkHIP(hipMemcpy(apDst, apSrc, aSize, hipMemcpyDeviceToDevice)); // Same device
    } else {
        std::cout << "Unknown copy direction: " << aCopyDirection << std::endl;
        assert(0);
    }
}
