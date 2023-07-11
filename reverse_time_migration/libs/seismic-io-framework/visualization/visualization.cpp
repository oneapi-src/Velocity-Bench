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


#include "visualization.h"
#include <dirent.h>
#include <errno.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/types.h>
#include <vector>

#ifdef USE_OpenCV

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#endif
using namespace std;

// got these from the open cv tools in the segy frame work
float *normalize_percentile(const float *a, int nx, int nz, float percentile) {
#ifdef USE_OpenCV
    float max_val = -std::numeric_limits<float>::max();
    vector<float> values;
    float *mod_img = new float[nx * nz];
    values.push_back(0);
    for (int i = 0; i < nx * nz; i++) {
        if (a[i] != 0) {
            values.push_back(fabs(a[i]));
        }
    }
    std::sort(values.begin(), values.end());
    int index = ((percentile / 100.0) * (values.size() - 1));
    if (index < 0) {
        index = 0;
    }
    max_val = values[index];
    for (int i = 0; i < nx * nz; i++) {
        if (fabs(a[i]) > max_val) {
            if (a[i] > 0) {
                mod_img[i] = max_val;
            } else {
                mod_img[i] = -max_val;
            }
        } else {
            mod_img[i] = a[i];
        }
    }
    return mod_img;
#endif
}

void showRegGrid_par(const float *ptr, const int nz, const int nx,
                     float percentile) {
#ifdef USE_OpenCV
    pid_t pid;
    pid = fork();
    if (pid == 0)
        return;
    else
        showRegGrid(ptr, nz, nx, percentile);
    exit(EXIT_FAILURE);
#endif
}

void showRegGrid(const float *ptr, const int nz, const int nx,
                 float percentile) {
#ifdef USE_OpenCV
    string title = "velocity";
    float *ptr_new = normalize_percentile(ptr, nx, nz, percentile);
    const cv::Mat data(nz, nx, CV_32FC1, const_cast<float *>(ptr_new));
    cv::Mat data_displayed(nz, nx, CV_8UC1);
    float minV = *min_element(data.begin<float>(), data.end<float>());
    float maxV = *max_element(data.begin<float>(), data.end<float>());
    cv::Mat data_scaled = (data - minV) / (maxV - minV);
    data_scaled.convertTo(data_displayed, CV_8UC1, 255.0, 0);
    //    cv::applyColorMap(data_displayed, data_displayed, cv::COLORMAP_BONE);

    cv::namedWindow(title, cv::WINDOW_NORMAL); // Create a window for display.
    cv::imshow(title, data_displayed);
    cv::waitKey(0);
#endif
}

void GridToPNG(const float *ptr, const int nz, const int nx, float percentile,
               char *filename) {
#ifdef USE_OpenCV
    string title = "velocity";
    float *ptr_new = normalize_percentile(ptr, nx, nz, percentile);
    const cv::Mat data(nz, nx, CV_32FC1, const_cast<float *>(ptr_new));
    cv::Mat data_displayed(nz, nx, CV_8UC1);
    float minV = *min_element(data.begin<float>(), data.end<float>());
    float maxV = *max_element(data.begin<float>(), data.end<float>());
    cv::Mat data_scaled = (data - minV) / (maxV - minV);
    data_scaled.convertTo(data_displayed, CV_8UC1, 255.0, 0);
    //    cv::applyColorMap(data_displayed, data_displayed, cv::COLORMAP_BONE);
    cv::imwrite(filename, data_displayed);
    delete[] ptr_new;
#endif
}

void GridToPNG(const float *ptr, const int nz, const int nx, char *filename,
               float maxV, float minV) {
#ifdef USE_OpenCV
    string title = "velocity";
    const cv::Mat data(nz, nx, CV_32FC1, const_cast<float *>(ptr));
    cv::Mat data_displayed(nz, nx, CV_8UC1);
    cv::Mat data_scaled = (data - minV) / (maxV - minV);
    data_scaled.convertTo(data_displayed, CV_8UC1, 255.0, 0);
    //    cv::applyColorMap(data_displayed, data_displayed, cv::COLORMAP_BONE);
    cv::imwrite(filename, data_displayed);
#endif
}

int getdir(string dir, vector<string> &files) {
#ifdef USE_OpenCV
    DIR *dp;
    struct dirent *dirp;
    if ((dp = opendir(dir.c_str())) == NULL) {
        cout << "Error(" << errno << ") opening " << dir << endl;
        return errno;
    }

    while ((dirp = readdir(dp)) != NULL) {
        files.push_back(string(dirp->d_name));
    }
    closedir(dp);
    return 0;
#endif
}

bool hasEnding(std::string const &full_string, std::string const &ending) {
#ifdef USE_OpenCV
    if (full_string.length() >= ending.length()) {
        return (0 == full_string.compare(full_string.length() - ending.length(),
                                         ending.length(), ending));
    } else {
        return false;
    }
#endif
}

vector<string> filter_files(vector<string> &files, string &extension) {
#ifdef USE_OpenCV
    vector<string> filtered;
    for (unsigned int i = 0; i < files.size(); i++) {
        if (hasEnding(files[i], extension)) {
            filtered.push_back(files[i]);
        }
    }
    return filtered;
#endif
}

void ReadCsv(float *mat, uint nz, uint nx, string filename) {
#ifdef USE_OpenCV
    std::ifstream in(filename);
    string line, word;
    getline(in, line);
    for (uint row = 0; row < nz; row++) {
        getline(in, line, '\n');
        stringstream s(line);
        for (uint col = 0; col < nx; col++) {
            getline(s, word, ',');
            mat[row * nx + col] = stod(word);
        }
    }
#endif
}

void ReadCsvHeader(uint *nz, uint *nx, string filename) {
#ifdef USE_OpenCV
    std::ifstream in(filename);
    char v;
    int n;
    in >> n;
    *nx = n;
    in >> v;
    in >> n;
    *nz = n;
    in >> v;
#endif
}
