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
// Created by zeyad-osama on 11/03/2021.
//

#ifndef THOTH_INDEXERS_INDEX_MAP_HPP
#define THOTH_INDEXERS_INDEX_MAP_HPP

#include <thoth/data-units/data-types/TraceHeaderKey.hpp>

#include <vector>

namespace thoth {
    namespace indexers {

        class IndexMap {
        public:
            /**
             * @brief Constructor.
             */
            IndexMap();

            /**
             * @brief Destructor.
             */
            ~IndexMap();

            /**
             * @brief Resets current map for reusable purposes.
             * @return Status flag. 
             */
            int
            Reset();

            /**
             * @brief Adds a byte position to the vector corresponding to the desired trace header key.
             * @param[in] aTraceHeaderKey 
             * @param[in] aTraceHeaderValue
             * @param[in] aBytePosition
             * @return Status flag. 
             */
            int
            Add(const dataunits::TraceHeaderKey &aTraceHeaderKey,
                const size_t &aTraceHeaderValue,
                const size_t &aBytePosition);

            /**
             * @brief Adds a byte position vector corresponding to the desired trace header key.
             * @param[in] aTraceHeaderKey 
             * @param[in] aTraceHeaderValue
             * @param[in] aBytePositions
             * @return Status flag. 
             */
            int
            Add(const dataunits::TraceHeaderKey &aTraceHeaderKey,
                const size_t &aTraceHeaderValue,
                const std::vector<size_t> &aBytePositions);

            /**
             * @brief Gets vector of byte positions corresponding to the desired trace header key and value.
             * @param[in] aTraceHeaderKey
             * @param[in] aTraceHeaderValue
             * @return Byte positions vector.
             */
            inline std::vector<size_t> &
            Get(const dataunits::TraceHeaderKey &aTraceHeaderKey,
                const size_t &aTraceHeaderValue) { return this->mIndexMap[aTraceHeaderKey][aTraceHeaderValue]; }

            /**
             * @brief Gets map of byte positions corresponding to the desired trace header key and value.
             * @param[in] aTraceHeaderKey
             * @param[in] aTraceHeaderValue
             * @return Byte positions map.
             */
            std::map<size_t, std::vector<size_t>>
            Get(const dataunits::TraceHeaderKey &aTraceHeaderKey,
                std::vector<size_t> &aTraceHeaderValues);

            /**
             * @brief Gets map of trace header values as keys and byte positions as values,
             * corresponding to the desired trace header key.
             *
             * @param[in] aTraceHeaderKey
             * @return Byte positions vector.
             */
            inline std::map<size_t, std::vector<size_t>> &
            Get(const dataunits::TraceHeaderKey &aTraceHeaderKey) { return this->mIndexMap[aTraceHeaderKey]; }

            /**
             * @brief Gets complete map. (i.e. Each trace header key and it's corresponding byte positions vector.)
             * @return Complete map.
             */
            inline std::unordered_map<dataunits::TraceHeaderKey, std::map<size_t, std::vector<size_t>>> &
            Get() { return this->mIndexMap; }

        private:
            /// Index map, having trace header key as key and vector of byte positions
            /// in file corresponding to this trace header key.
            std::unordered_map<dataunits::TraceHeaderKey, std::map<size_t, std::vector<size_t>>> mIndexMap;
        };

    } //namespace indexers
} //namespace thoth

#endif //THOTH_INDEXERS_INDEX_MAP_HPP
