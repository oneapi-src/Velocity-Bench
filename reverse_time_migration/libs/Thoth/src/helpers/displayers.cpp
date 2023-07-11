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
// Created by zeyad-osama on 09/03/2021.
//

#include <thoth/helpers/displayers.h>

#include <thoth/lookups/tables/TextHeaderLookup.hpp>

#include <iostream>


void thoth::helpers::displayers::print_text_header(unsigned char *apTextHeader) {
    if (apTextHeader == nullptr) {
        std::cerr << "Error: Null pointer received. Nothing to be printed." << std::endl;
        return;
    }

    for (size_t i = 0; i < IO_SIZE_TEXT_HEADER; i++) {
        if ((i % 80) == 0)
            std::cout << std::endl;
        std::cout << apTextHeader[i];
    }
    std::cout << std::endl;
}
