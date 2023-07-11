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


#ifndef IO_FRAMEWORK_MANAGER_H
#define IO_FRAMEWORK_MANAGER_H

#include "../datatypes.h"

#include <iostream>
#include <string>
#include <vector>

class IOManager {
public:
    /*!
     * De-constructors should be overridden to ensure correct memory management.

     */

    // auto *temp2 = new seisio::general_traces();

    //   virtual IOManager() = 0;

    virtual ~IOManager() {};

    virtual void ReadTracesDataFromFile(std::string const &file_name,
                                        std::string const &sort_type,
                                        SeIO *sio) = 0;

    virtual void ReadVelocityDataFromFile(std::string const &file_name,
                                          std::string const &sort_type,
                                          SeIO *sio) = 0;

    virtual void ReadDensityDataFromFile(std::string const &file_name,
                                         std::string const &sort_type,
                                         SeIO *sio) = 0;

    // how we would like to pass the conditions for selective read ??
    virtual void ReadSelectiveDataFromFile(std::string const &file_name,
                                           std::string const &sort_type,
                                           SeIO *sio, int cond) = 0;

    virtual void WriteTracesDataToFile(std::string const &file_name,
                                       std::string const &sort_type,
                                       SeIO *sio) = 0;

    virtual void WriteVelocityDataToFile(std::string const &file_name, 
                                         std::string const &sort_type,
                                         SeIO *sio) = 0;

    virtual void WriteDensityDataToFile(std::string const &file_name, 
                                        std::string const &sort_type,
                                        SeIO *sio) = 0;

    virtual std::vector<uint>
    GetUniqueOccurences(std::string const &file_name, std::string const &key_name, uint min_threshold, uint max_threshold) = 0;

    // virtual void ReadTraceMetaData() = 0;

    //   virtual void ReadTracesData() = 0 ;
};

#endif // IO_FRAMEWORK_MANAGER_H
