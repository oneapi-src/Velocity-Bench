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
// Created by zeyad-osama on 19/07/2020.
//

#ifndef SEISMIC_TOOLBOX_PARSERS_PARSER_H
#define SEISMIC_TOOLBOX_PARSERS_PARSER_H

#include <operations/common/Singleton.tpp>

#include <libraries/nlohmann/json.hpp>

namespace stbx {
    namespace parsers {
        /**
         * @brief Parser class responsible for registering .json files
         * to be parsed or a whole folder to parse all .json files inside.
         * <br>
         * Variant parsing functions are implemented to provide extraction of
         * some components from parsed map and return specific object instance.
         */
        class Parser : public operations::common::Singleton<Parser> {
        public:
            friend class Singleton<Parser>;

        public:
            /**
             * @brief Builds cne big map containing all registered files
             * key values pairs.
             * @return Map : json -> json::Value
             */
            nlohmann::json BuildMap();

            /**
             * @brief Registers a given .json files to be parsed.
             * @param folder : string       File path to be parsed
             */
            void RegisterFile(const std::string &file);

            const std::vector<std::string> &GetFiles() const;

            /**
             * @brief Registers all .json files found inside a folder.
             * @param folder : string       Folder path to be parsed
             */
            void RegisterFolder(const std::string &folder);

            /**
             * @brief Map getter.
             * @return Map : json -> json::Value
             */
            nlohmann::json GetMap();

        private:
            /// Map that holds configurations key value pairs
            nlohmann::json mMap;

            /// Registered files to be parsed.
            std::vector<std::string> mFiles;
        };
    }//namespace parsers
}//namespace stbx

#endif //SEISMIC_TOOLBOX_PARSERS_PARSER_H
