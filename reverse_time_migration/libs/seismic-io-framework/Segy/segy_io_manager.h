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


#ifndef SEGY_IO_MANAGER_H
#define SEGY_IO_MANAGER_H

#include "../IO/io_manager.h"
#include "../datatypes.h"
#include "math.h"
#include "susegy.h"

inline float dBtoscale(float x) {
    int i = x / abs(x);

    return pow(10, i * log10(abs(x)));
}

class SEGYIOManager : public IOManager {

public:
    SEGYIOManager();

    ~SEGYIOManager();

    void ReadTracesDataFromFile(std::string const &file_name, std::string const &sort_type, SeIO *sio) override;

    void ReadVelocityDataFromFile(std::string const &file_name, std::string const &sort_type, SeIO *sio) override;

    void ReadDensityDataFromFile(std::string const &file_name, std::string const &sort_type, SeIO *sio) override;

    //  passign the condtions , how would like to pass the parameters ???
    // currently will take hsoy id as cond , later will be more general
    void ReadSelectiveDataFromFile(std::string const &file_name, std::string const &sort_type, SeIO *sio, int cond) override;

    void WriteTracesDataToFile(std::string const &file_name, std::string const &sort_type, SeIO *sio) override;
//using namespace std;

    void WriteVelocityDataToFile(std::string const &file_name, std::string const &sort_type, SeIO *sio) override;

    void WriteDensityDataToFile(std::string const &file_name, std::string const &sort_type, SeIO *sio) override;

    std::vector<uint> GetUniqueOccurences(std::string const &file_name, std::string const &key_name, uint min_threshold, uint max_threshold) override;
    // do we need structure for binary header , traces header or we should cancel
    // these functions ??
};

#endif // SEGY_IO_MANAGER_H
