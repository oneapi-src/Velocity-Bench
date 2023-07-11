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

#ifndef OPERATIONS_LIB_DATA_UNITS_FRAMEBUFFER_HPP
#define OPERATIONS_LIB_DATA_UNITS_FRAMEBUFFER_HPP

#include <operations/data-units/interface/DataUnit.hpp>

#include <operations/common/DataTypes.h>
#include <string>

namespace operations {
    namespace dataunits {

        template<typename T>
        class FrameBuffer : public DataUnit {
        public:
            FrameBuffer();

            explicit FrameBuffer(uint aSize);

            ~FrameBuffer() override;

            void
            Allocate(uint aSize, const std::string &aName = "");

            void
            Allocate(uint aSize, HALF_LENGTH aHalfLength, const std::string &aName = "");

            void
            Free();

            T *
            GetNativePointer();

            T *
            GetHostPointer();

            void
            SetNativePointer(T *pT);

            void
            ReflectOnNative();

        private:
            T *mpDataPointer;
            T *mpHostDataPointer;
            uint mAllocatedBytes;

            FrameBuffer(FrameBuffer const &RHS) = delete;
            FrameBuffer &operator=(FrameBuffer const &RHS) = delete;
        };

        namespace Device {
            enum CopyDirection : short {
                COPY_HOST_TO_HOST,
                COPY_HOST_TO_DEVICE,
                COPY_DEVICE_TO_HOST,
                COPY_DEVICE_TO_DEVICE,
                COPY_DEFAULT
            };

            void MemSet(void *apDst, int aVal, uint aSize);

            void MemCpy(void *apDst, const void *apSrc, uint aSize, CopyDirection aCopyDirection = COPY_DEFAULT);
        }
    } //namespace dataunits
} //namespace operations

#endif //OPERATIONS_LIB_DATA_UNITS_FRAMEBUFFER_HPP
