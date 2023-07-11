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
// Created by zeyad-osama on 02/11/2020.
//

#ifndef THOTH_DATA_UNITS_TRACE_HPP
#define THOTH_DATA_UNITS_TRACE_HPP

#include <thoth/data-units/data-types/TraceHeaderKey.hpp>
#include <thoth/common/assertions.h>

#include <string>
#include <unordered_map>
#include <memory>
#include <sstream>
#include <vector>

namespace thoth {
    namespace dataunits {
        /**
         * @brief Trace Object contains TraceHeaderKey represented as a map,
         */
        class Trace {
        public:
            /**
             * @brief Constructor
             */
            explicit Trace(const unsigned short aNS) {
                this->mTraceHeaderMap[TraceHeaderKey(TraceHeaderKey::NS)] = std::to_string(aNS);
            };

            /**
             * @brief Move constructor, to move ownership of TraceData.
             *
             * @param aTrace
             */
            Trace(Trace &&aTrace) noexcept
                    : mpTraceData(std::move(aTrace.mpTraceData)) {}

            /**
             * @brief Copy constructor.
             */
            Trace(const Trace &) {};

            /**
             * @brief Destructor
             */
            ~Trace() = default;

            /**
             * @brief Equality operator overloading.
             */
            Trace &operator=(const Trace &) = delete;

            /**
             * @brief Getter for the Trace Header Value.
             *
             * @param[in] aTraceHeaderKey.
             *
             * @return Trace Header Value if available, else find() will
             * point to the end of map.
             */
            template<typename T>
            inline T GetTraceHeaderKeyValue(TraceHeaderKey aTraceHeaderKey) {
                ASSERT_T_TEMPLATE(T);
                std::stringstream ss;
                ss << this->mTraceHeaderMap[aTraceHeaderKey];
                T trace_header_value;
                ss >> trace_header_value;
                return trace_header_value;
            }

            const std::unordered_map<TraceHeaderKey, std::string> *GetTraceHeaders() {
                return &this->mTraceHeaderMap;
            }

            /**
             * @brief Setter for the Trace Header Value.
             *
             * @param[in] aTraceHeaderKey
             * Trace Header Key argument.
             *
             * @param[in] aValue
             * Value to set for the corresponding key
             */
            template<typename T>
            inline void SetTraceHeaderKeyValue(TraceHeaderKey aTraceHeaderKey, T aValue) {
                ASSERT_T_TEMPLATE(T);
                this->mTraceHeaderMap[aTraceHeaderKey] = std::to_string(aValue);
            }

            /**
             * @brief Setter for trace data.
             */
            inline void SetTraceData(float *apTraceData) {
                return this->mpTraceData.reset(apTraceData);
            }


            /**
             * @brief Getter for trace data.
             *
             * @return Pointer to trace data.
             */
            inline float *GetTraceData() {
                return this->mpTraceData.get();
            }

            /**
             * @brief Getter of number of samples trace header.
             *
             * @return Number of samples value.
             */
            inline unsigned short GetNumberOfSamples() {
                TraceHeaderKey trace_header_key(TraceHeaderKey::NS);
                return this->GetTraceHeaderKeyValue<unsigned short>(trace_header_key);
            }

        private:
            /// @brief TraceHeaderKey, containing trace headers and their corresponding value.
            /// @datastructure Unordered Ma
            /// @key TraceHeaderKey
            /// @value string
            std::unordered_map<TraceHeaderKey, std::string> mTraceHeaderMap;

            /// @brief TraceData, containing trace data.
            /// @datastructure 1D array, controlled by a unique pointer.
            std::unique_ptr<float[]> mpTraceData;
        };
    }//namespace dataunits
} //namespace thoth

#endif //THOTH_DATA_UNITS_TRACE_HPP
