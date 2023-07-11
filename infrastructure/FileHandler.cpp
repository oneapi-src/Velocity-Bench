/*
 * 
 * Modifications Copyright (C) 2023 Intel Corporation
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * 
 * of this software and associated documentation files (the "Software"),
 * 
 * to deal in the Software without restriction, including without limitation
 * 
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * 
 * and/or sell copies of the Software, and to permit persons to whom
 * 
 * the Software is furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included
 * 
 * in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * 
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * 
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * 
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
 * 
 * OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * 
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
 * 
 * OR OTHER DEALINGS IN THE SOFTWARE.
 * 
 * SPDX-License-Identifier: MIT
 */


#include "FileHandler.h"
#include <iostream>
#include <filesystem>

#define CPP_MODULE "FILE"
#include "Logging.h"

Utility::FileHandler::~FileHandler()
{
    m_FileHandler.close();
    LOG("Closing binary file handler");
}

Utility::FileHandler::FileHandler(std::string const &sFileName, Mode_t const mode)
    : m_FileHandler()
    , m_FileMode   (mode)
{
    switch(mode) {
        case Mode_t::INPUT_BINARY:  m_FileHandler.open(sFileName, std::ios::in  | std::ios::binary); break;
        case Mode_t::OUTPUT_BINARY: m_FileHandler.open(sFileName, std::ios::out | std::ios::binary); break;
        default: LOG_ERROR("Unknown file mode"); // Should never reach here 
    };

    if (!m_FileHandler.good()) {
        LOG_ERROR("Unable to correctly open " << sFileName);
    }
    LOG("Reading binary contents from " << sFileName);
}

bool Utility::FileHandler::FileExists(std::string const &sFileName)
{
    return std::filesystem::exists(sFileName);
}

std::vector<std::string> Utility::FileHandler::GetFilesFromDirectory(std::string const &sPath)
{
    if (!FileExists(sPath))
        return std::vector<std::string>();

    std::vector<std::string> vFoundFiles;
    for (auto const &elem : std::filesystem::directory_iterator(sPath))
        vFoundFiles.emplace_back(elem.path());
    return vFoundFiles; 
}

size_t Utility::FileHandler::RemoveFiles(std::vector<std::string> const &vFiles)
{
    unsigned int uiNumberOfFilesRemoved(0);
    for (auto const &sFile : vFiles)  {
        if (!std::filesystem::remove(sFile))
            continue;
        ++uiNumberOfFilesRemoved;
    }
    return uiNumberOfFilesRemoved;
}
