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


#ifndef CLASS_FILE_HANDLER_H
#define CLASS_FILE_HANDLER_H

// Note: Please compile using -lstdc++fs flag to enable the file system STL library
// For cmake, add stdc++fs into target_link_libraries

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>

namespace Utility 
{
class FileHandler
{
public:
    enum class Mode_t { INPUT_BINARY, OUTPUT_BINARY };
private:
    std::fstream m_FileHandler;
    Mode_t const m_FileMode;
public:
    FileHandler           (FileHandler const &RHS) = delete;
    FileHandler &operator=(FileHandler const &RHS) = delete;

    explicit FileHandler(std::string const &sFileName, Mode_t const mode);
            ~FileHandler();

    static bool FileExists(std::string const &sFileName);
    static std::vector<std::string> GetFilesFromDirectory(std::string const &sPath); // Does not perform recursive search
    static size_t RemoveFiles(std::vector<std::string> const &vFiles);
    static std::string GetCurrentDirectory() { return std::filesystem::current_path(); } 


template<typename T>
bool ReadBytes(unsigned int const uiNumberOfBytesToRead, T *pDestination)
{
    if (m_FileMode == Mode_t::OUTPUT_BINARY) {
        std::cerr << "File handler is not configured for reading" << std::endl; 
        return false;
    }

    if (!m_FileHandler.read(reinterpret_cast<char*>(pDestination), uiNumberOfBytesToRead)) {
        std::cerr << "FILE ERROR: Unable to read " << uiNumberOfBytesToRead << ". Can only read " << m_FileHandler.gcount() << std::endl;
        return false; 
    }

    if (!m_FileHandler.good()) {
        std::cerr << "FILE ERROR: Internal file reading error..." << std::endl;
        return false;
    }
    return true;
}

template<typename T>
bool WriteBytes(unsigned int const uiNumberOfBytesToWrite, T *pSource)
{
    if (m_FileMode == Mode_t::INPUT_BINARY) {
        std::cerr << "File handler is not configured for writing" << std::endl;
        return false;
    }

    m_FileHandler.write(reinterpret_cast<char*>(pSource), uiNumberOfBytesToWrite);
    if (!m_FileHandler.good()) {
        std::cerr << "FILE ERROR: Internal file writing error..." << std::endl;
        return false;
    }
}

};

};

#endif 
