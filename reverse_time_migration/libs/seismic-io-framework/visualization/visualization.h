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


#ifndef VISULAIZATION_h
#define VISULAIZATION_h

#include "../datatypes.h"

#ifdef USE_OpenCV

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#endif

#include <sys/types.h>
#include <unistd.h>

#include <pthread.h>

float *normalize_percentile(const float *a, int nx, int nz, float percentile);

void showRegGrid_par(const float *ptr, const int nz, const int nx,
                     float percentile);

void showRegGrid(const float *ptr, const int nz, const int nx,
                 float percentile);

void GridToPNG(const float *ptr, const int nz, const int nx, float percentile,
               char *filename);

void GridToPNG(const float *ptr, const int nz, const int nx, char *filename,
               float maxV, float minV);

// should they be private or not ??

void ReadCsvHeader(uint *nz, uint *nx, std::string filename);

void ReadCsv(float *mat, uint nz, uint nx, std::string filename);

std::vector<std::string> filter_files(std::vector<std::string> &files,
                                      std::string &extension);

bool hasEnding(std::string const &full_string, std::string const &ending);

int getdir(std::string dir, std::vector<std::string> &files);

#endif // VISUALIZATION_h
