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
// Created by zeyad-osama on 05/01/2021.
//

#include <stbx/utils/cnv/csv_to_png.hpp>

#if (USE_OpenCV)

#include <visualization/visualization.h>

#include <iostream>
#include <string>
#include <vector>

using namespace std;

int stbx::utils::cnv::csv_to_png(const string &dir, float percentile) {
    vector<string> files = vector<string>();
    string extension = ".csv";
    getdir(dir, files);
    files = filter_files(files, extension);
    uint nz, nx;
    if (files.empty()) {
        return EXIT_FAILURE;
    }

    cout << "Files to be processed : " << endl;
    for (auto &file : files) {
        cout << "\t" << file << endl;
    }

    ReadCsvHeader(&nz, &nx, files[0]);
    uint nx_nz = nx * nz;
    uint nt = files.size();
    cout << "Grid dimensions : nx = " << nx << " nz = " << nz << endl;
    auto *matrix = new float[nt * nx_nz];
    for (unsigned int i = 0; i < files.size(); i++) {
        cout << "Reading csv : " << files[i] << endl;
        ReadCsv(matrix + i * nx_nz, nz, nx, files[i]);
    }
    float *matrix_percentiled = normalize_percentile(matrix, nx_nz, nt, percentile);
    float minV = *min_element(matrix_percentiled, matrix_percentiled + nt * nx_nz);
    float maxV = *max_element(matrix_percentiled, matrix_percentiled + nt * nx_nz);
    for (unsigned int i = 0; i < files.size(); i++) {
        string s = files[i];
        s.replace(s.end() - 4, s.end(), ".png");
        GridToPNG(matrix_percentiled + i * nx_nz, nz, nx, (char *) s.c_str(), maxV,
                  minV);
    }

    delete[] matrix;
    if (matrix_percentiled != nullptr)
        delete[] matrix_percentiled;

    return EXIT_SUCCESS;
}

#endif
