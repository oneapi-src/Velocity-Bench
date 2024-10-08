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
// Created by amr on 03/01/2021.
//

#ifndef OPERATIONS_LIB_BACKEND_ONEAPI_BACKEND_HPP
#define OPERATIONS_LIB_BACKEND_ONEAPI_BACKEND_HPP

#include <operations/common/Singleton.tpp>
#include <sycl/sycl.hpp>

namespace operations {
    namespace backend {
        /**
         * @brief
         * The algorithm in effect to be used.
         */
        enum class SYCL_ALGORITHM {
            CPU, GPU, GPU_SHARED, GPU_SEMI_SHARED
        };

        /**
         * @brief
         * The backend used for all computations in the oneAPI technology.
         */
        class OneAPIBackend : public common::Singleton<OneAPIBackend> {
        public:
            friend class Singleton<OneAPIBackend>;

        public:
            /**
             * @brief
             * Get the Device queue in use for the DPC++ computations.
             *
             * @return
             * A pointer to the Device queue in use.
             */
            inline sycl::queue *GetDeviceQueue() {
                return mDeviceQueue;
            }

            /**
             * @brief
             * Setter for the Device queue being in use.
             *
             * @param[in] aDeviceQueue
             */
            void SetDeviceQueue(sycl::queue *aDeviceQueue);

            /**
             * @brief
             * Get the algorithm to be used in dpc++ computations.
             *
             * @return
             * The algorithm to be used in all dpc++ computations.
             */
            inline SYCL_ALGORITHM GetAlgorithm() {
                return mOneAPIAlgorithm;
            }

            /**
             * @brief
             * Setter for the algorithm to be used.
             *
             * @param[in] aOneAPIAlgorithm
             * The algorithm to be used.
             */
            void SetAlgorithm(SYCL_ALGORITHM aOneAPIAlgorithm);

        private:
            /**
             * @brief
             * Default Constructor.
             */
            OneAPIBackend();

            /**
             * @brief
             * Default Destructor.
             */
            ~OneAPIBackend();

            OneAPIBackend(OneAPIBackend const &RHS) = delete;
            OneAPIBackend &operator=(OneAPIBackend const &RHS) = delete;

        private:
            /// The Device queue.
            sycl::queue *mDeviceQueue;
            /// The DPC++ underlying algorithm being used.
            SYCL_ALGORITHM mOneAPIAlgorithm;
        };
    } //namespace backend
} //namespace operations

#endif //OPERATIONS_LIB_BACKEND_ONEAPI_BACKEND_HPP
