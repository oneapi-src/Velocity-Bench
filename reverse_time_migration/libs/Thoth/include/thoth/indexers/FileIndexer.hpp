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

#ifndef THOTH_INDEXERS_FILE_INDEXER_HPP
#define THOTH_INDEXERS_FILE_INDEXER_HPP

#include <thoth/streams/helpers/InStreamHelper.hpp>
#include <thoth/streams/helpers/OutStreamHelper.hpp>
#include <thoth/indexers/IndexMap.hpp>

#include <string>

namespace thoth {
    namespace indexers {

        class FileIndexer {
        public:
            /**
             * @brief Explicit constructor.
             * @param aFilePath
             */
            explicit FileIndexer(std::string &aFilePath);

            /**
             * @brief Destructor.
             */
            ~FileIndexer();

            /**
             * @brief
             * Initializes the indexer with the appropriate settings to be applied.
             *
             * @return File initial size (i.e. Zero value).
             */
            size_t
            Initialize();

            /**
             * @brief
             * Release all resources and close everything.
             * Should be initialized again afterwards to be able to reuse it again.
             *
             * @return Status flag.
             */
            int
            Finalize();

            /**
             * @brief Do the actual indexing upon the given file. Takes trace header keys vector to index upon it
             * @param[in] aTraceHeaderKeys
             * @return Index Map.
             */
            indexers::IndexMap
            Index(const std::vector<dataunits::TraceHeaderKey> &aTraceHeaderKeys);

            /**
             * @brief Flushes current index map to the destined index file corresponding to the given file.
             * @return Status flag.
             */
            int
            Flush();

            /**
             * @brief Index Map getter.
             * @return IndexMap object.
             */
            inline indexers::IndexMap
            GetIndexMap() { return this->mIndexMap; }

            /**
             * @brief Indexed file path getter.
             */
            inline std::string
            GetIndexedFilePath() { return this->mOutFilePath; }

        private:
            static std::string
            GenerateOutFilePathName(std::string &aInFileName);


        private:
            /// Input ile path.
            std::string mInFilePath;
            /// Output file path.
            std::string mOutFilePath;
            /// File input stream helper.
            streams::helpers::InStreamHelper *mInStreamHelper;
            /// File output stream helper.
            streams::helpers::OutStreamHelper *mOutStreamHelper;
            /// Index map, having trace header key as key and vector of byte positions
            /// in file corresponding to this trace header key.
            indexers::IndexMap mIndexMap;

            FileIndexer &operator=(FileIndexer const &RHS) = delete;
        };

    } //namespace indexers
} //namespace thoth

#endif //THOTH_INDEXERS_FILE_INDEXER_HPP
