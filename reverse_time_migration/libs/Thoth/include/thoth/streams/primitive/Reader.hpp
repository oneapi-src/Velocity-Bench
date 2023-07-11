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
// Created by amr-nasr on 12/01/2021.
//

#ifndef THOTH_STREAMS_READER_HPP
#define THOTH_STREAMS_READER_HPP

#include <thoth/streams/interface/Stream.hpp>
#include <thoth/configurations/interface/Configurable.hpp>
#include <thoth/data-units/concrete/Trace.hpp>
#include <thoth/data-units/concrete/Gather.hpp>
#include <thoth/data-units/data-types/TraceHeaderKey.hpp>

#include <unordered_map>
#include <vector>

namespace thoth {
    namespace streams {
        /**
         * @brief
         * Reader interface for seismic data.
         * This should be configurable to allow each reader to extract its unique additional properties.
         * A simple flow should be like the following:
         * Reader.AcquireConfiguration(...);
         * Reader.Initialize(...);
         * Do all other operations afterwards.
         */
        class Reader : public Stream {
        public:
            /**
             * @brief
             * Virtual destructor to allow correct destruction of concrete classes.
             */
            virtual ~Reader() override = default;

            /**
             * @brief
             * Initializes the reader with the appropriate settings applied
             * to the reader, and any preparations needed are done.
             * Should be called once at the start.
             *
             * @param[in] aGatherKeys
             * A vector containing the strings of the headers to utilize as the gathers in reading.
             * If provided an empty vector, will not gather on any key and will return all data.
             *
             * @param[in] aSortingKeys
             * A vector of pairs with each pair containing a string representing the header to sort on, and the sorting direction.
             * The vector is treated with the first element being the most major decider, and the last the most minor one.
             * If empty, no sorting is applied.
             *
             * @param[in] aPaths
             * List of paths to be processed by the the reader. 
             */
            virtual int
            Initialize(std::vector<std::string> &aGatherKeys,
                       std::vector<std::pair<std::string, dataunits::Gather::SortDirection>> &aSortingKeys,
                       std::vector<std::string> &aPaths) = 0;

            /**
             * @brief
             * Initializes the reader with the appropriate settings applied
             * to the reader, and any preparations needed are done.
             * Should be called once at the start.
             *
             * @param[in] aGatherKeys
             * A vector containing the trace header keys of the headers to utilize as the gathers in reading.
             * If provided an empty vector, will not gather on any key and will return all data.
             *
             * @param[in] aSortingKeys
             * A vector of pairs with each pair containing a string representing the header to sort on, and the sorting direction.
             * The vector is treated with the first element being the most major decider, and the last the most minor one.
             * If empty, no sorting is applied.
             *
             * @param[in] aPaths
             * List of paths to be processed by the the reader.
             */
            virtual int
            Initialize(std::vector<dataunits::TraceHeaderKey> &aGatherKeys,
                       std::vector<std::pair<dataunits::TraceHeaderKey, dataunits::Gather::SortDirection>> &aSortingKeys,
                       std::vector<std::string> &aPaths) = 0;

            /**
             * @brief
             * Release all resources and close everything.
             * Should be initialized again afterwards to be able to reuse it again.
             */
            virtual int
            Finalize() = 0;

            /**
             * @brief
             * Sets the reader to only read and set the headers in the gather read functionalities.
             * This will reduce the time needed, however won't populate the data in the traces at all.
             * You should be able to enable and disable as you want during the runs.
             * By default, this mode is considered false unless explicitly called and set.
             * 
             * @param[in] aEnableHeaderOnly
             * If true, make all gather read functions read and set the headers only.
             * If false, will make all gather read functions read headers and data.
             */
            virtual void
            SetHeaderOnlyMode(bool aEnableHeaderOnly) = 0;

            /**
             * @brief
             * Return the total number of gathers available for reading.
             * 
             * @return
             * An unsigned int representing how many unique gathers are available for the reader.
             */
            virtual unsigned int
            GetNumberOfGathers() = 0;

            /**
             * @brief
             * Get the unique values of gathers available to be read.
             * 
             * @return
             * A list of sub-lists.
             * Each sublist represent one gather unique value combination.
             * The sublist length is equal to the number of keys that the reader is set to gather on.
             * The list length is equal to the value returned by GetNumberOfGathers().
             * The order of values in the sublist matches the same order given in the initialize.
             */
            virtual std::vector<std::vector<std::string>>
            GetIdentifiers() = 0;

            /**
             * @brief
             * Get all the gathers that can be read by the reader.
             *
             * @return
             * A vector of all possible gathers.
             * The gather pointers returned should be deleted by the user to avoid memory leaks.
             */
            virtual std::vector<dataunits::Gather *>
            ReadAll() = 0;

            /**
             * @brief
             * Get a group of gathers with the requested unique values.
             *
             * @param[in] aHeaderValues
             * Each sublist represent one gather unique value combination.
             * The sublist length is equal to the number of keys that the reader is set to gather on.
             * The order of values in the sublist matches the same order given in the initialize.
             *
             * @return
             * A vector of the gathers matching the given request.
             * The gather pointers returned should be deleted by the user to avoid memory leaks.
             */
            virtual std::vector<dataunits::Gather *>
            Read(std::vector<std::vector<std::string>> aHeaderValues) = 0;

            /**
             * @brief
             * Get a group of gathers with the requested unique values.
             *
             * @param[in] aHeaderValues
             * Each list represent one gather unique value combination.
             * The list length is equal to the number of keys that the reader is set to gather on.
             * The order of values in the list matches the same order given in the initialize.
             * 
             * @return
             * The gather matching the given request.
             * The gather pointer returned should be deleted by the user to avoid memory leaks.
             */
            virtual thoth::dataunits::Gather *
            Read(std::vector<std::string> aHeaderValues) = 0;

            /**
             * @brief
             * Get a gather by its index.
             * 
             * @param[in] aIndex
             * The index of the gather to be read. Must be between 0 and 
             * the number returned by GetNumberOfGathers().
             * It should return the gather with the unique value equivalent to
             * the entry of GetUniqueGatherValues() at index aIndex.
             *
             * @return
             * The requested gather if possible, nullptr otherwise.
             * The gather pointer returned should be deleted by the user to avoid memory leaks.
             */
            virtual thoth::dataunits::Gather *
            Read(unsigned int aIndex) = 0;
        };

    } //streams
} //thoth

#endif //THOTH_STREAMS_READER_HPP
