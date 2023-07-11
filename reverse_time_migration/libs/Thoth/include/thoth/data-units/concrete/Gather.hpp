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

#ifndef THOTH_DATA_UNITS_GATHER_HPP
#define THOTH_DATA_UNITS_GATHER_HPP

#include <thoth/data-units/concrete/Trace.hpp>
#include <thoth/data-units/data-types/TraceHeaderKey.hpp>
#include <thoth/common/assertions.h>

#include <vector>
#include <utility>
#include <unordered_map>
#include <algorithm>

namespace thoth {
    namespace dataunits {
        /**
         * @brief
         */
        class Gather {
        public:
            /**
             * @brief Sorting Direction enum. Either ascending or descending.
             */
            enum SortDirection : char {
                ASC,
                DES
            };

        public:
            explicit Gather();

            /**
             * @brief Constructor.
             *
             * @param[in] TraceHeaderKey
             * map (key value pairs) gathered on, and vector of gather traces.
             */
            explicit Gather(std::unordered_map<TraceHeaderKey, std::string> &aUniqueKeys,
                            const std::vector<Trace *> &aTraces);

            /**
             *
             * @brief Constructor.
             *
             * @param[in] aUniqueKey
             * Unique Key gathered on, its value, and vector of gather traces.
             *
             */
            explicit Gather(TraceHeaderKey aUniqueKey,
                            const std::string &aUniqueKeyValue,
                            const std::vector<Trace *> &aTraces);

            /**
             *
             * @brief Constructor.
             *
             * @param aUniqueKeys map.
             *
             */
            explicit Gather(std::unordered_map<TraceHeaderKey, std::string> &aUniqueKeys);

            /**
             * @brief Destructor.
             */
            ~Gather() = default;

            /**
             *@brief Unique Key value getter.
             *
             * @param aTraceHeaderKey.
             */
            template<typename T>
            inline T GetUniqueKeyValue(TraceHeaderKey aTraceHeaderKey) {
                ASSERT_T_TEMPLATE(T);
                std::stringstream ss;
                ss << this->mUniqueKeys[aTraceHeaderKey];
                T unique_trace_header_value;
                ss >> unique_trace_header_value;
                return unique_trace_header_value;
            }

            /**
             * @brief Unique Key value setter.
             *
             * @param[in] aTraceHeaderKey
             * @param[in] aTraceHeaderKeyValue.
             */
            template<typename T>
            inline void SetUniqueKeyValue(TraceHeaderKey aTraceHeaderKey, T aTraceHeaderKeyValue) {
                ASSERT_T_TEMPLATE(T);
                this->mUniqueKeys[aTraceHeaderKey] = std::to_string(aTraceHeaderKeyValue);
            }

            /**
             * @brief AddTrace function, adds a trace to gather traces.
             *
             * @param[in] apTrace.
             */
            inline void AddTrace(Trace *apTrace) {
                this->mTraces.push_back(apTrace);
            }

            /**
             * @brief AddTrace function, adds a trace to gather traces.
             *
             * @param[in] aTrace.
             */
            inline void AddTrace(const std::vector<Trace *> &aTraceVector) {
                for (auto const trace : aTraceVector) {
                    this->mTraces.push_back(trace);
                }
            }

            /**
             * @brief GetNumberTraces function, gets the number of traces in a gather.
             *
             * @return Number of traces in a gather.
             */
            inline size_t GetNumberTraces() {
                return this->mTraces.size();
            }

            /**
             * @brief GetTrace function, gets a trace at a given index.
             *
             * @param[in] aTraceIdx.
             *
             * @return Trace at the given index.
             */
            inline Trace *GetTrace(unsigned int aTraceIdx) {
                return this->mTraces.at(aTraceIdx);
            }

            /**
             * @brief RemoveTrace function, removes a trace at a given index.
             *
             * @param[in] aTraceIdx.
             */
            inline void RemoveTrace(unsigned int aTraceIdx) {
                delete this->mTraces[aTraceIdx];
                this->mTraces.erase(this->mTraces.begin() + aTraceIdx);
            }

            /**
             * @brief Sort Gather function, using custom compare function.
             */
            void SortGather(const std::vector<std::pair<TraceHeaderKey,
                    Gather::SortDirection>> &aSortingKeys);

            /**
             * @brief
             * Set the sampling rate of the gather.
             *
             * @param[in] aSamplingRate
             * A floating number indicating the sampling rate between each sample.
             * Microseconds (μs) for time data, Hertz (Hz) for frequency data,
             * meters (m) or feet (ft) for depth data
             */
            void SetSamplingRate(float aSamplingRate) {
                this->mSamplingRate = aSamplingRate;
            }

            /**
             * @brief
             * Get the sampling rate for the traces.
             *
             * @return
             * A floating number indicating the sampling rate between each sample.
             * Microseconds (μs) for time data, Hertz (Hz) for frequency data,
             * meters (m) or feet (ft) for depth data
             */
            inline float GetSamplingRate() const {
                return this->mSamplingRate;
            }

        private:
            /// Unique Key-Value map.
            std::unordered_map<TraceHeaderKey, std::string> mUniqueKeys;
            /// Traces container.
            std::vector<Trace *> mTraces;
            /// The sampling rate for the traces in the gather.
            float mSamplingRate;
        };
    }//namespace dataunits
} //namespace thoth

namespace thoth {
    namespace dataunits {
        /**
         * @brief Sorting compare trace struct. Used to send parameters to the compare function.
         */
        struct trace_compare_t {
            /**
             * @brief Constructor.
             *
             * @param[in] aSortingKeys
             * Sorting keys vector and Sorting Direction enum
             * either ASC or DESC
             */
            explicit trace_compare_t(
                    const std::vector<std::pair<TraceHeaderKey, Gather::SortDirection>> &aSortingKeys);

            bool operator()(Trace *aTrace_1, Trace *aTrace_2) const;

        private:
            std::vector<std::pair<TraceHeaderKey, Gather::SortDirection>> mSortingKeys;
            int mKeysSize;
        };
    }//namespace dataunits
} //namespace thoth

#endif //THOTH_DATA_UNITS_GATHER_HPP
