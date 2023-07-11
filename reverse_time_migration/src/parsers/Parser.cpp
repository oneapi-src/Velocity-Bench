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

#include <stbx/parsers/Parser.h>

#include <iostream>
#include <dirent.h>
#include <string>
#include <fstream>

#define JSON_EXTENSION ".json"

using namespace std;
using namespace stbx::parsers;
using json = nlohmann::json;


void Parser::RegisterFile(const std::string &file) {
    this->mFiles.push_back(file);
}

void Parser::RegisterFolder(const std::string &folder) {
    struct dirent *entry;
    DIR *dp = opendir(folder.c_str());
    if (dp != nullptr)
        while ((entry = readdir(dp))) {
            string file_name = entry->d_name;
            if (file_name.find(JSON_EXTENSION) != string::npos) {
                this->mFiles.push_back(folder + file_name);
            }
        }

    closedir(dp);
}

json Parser::BuildMap() {
    vector<int> erase;
    string extension = string(JSON_EXTENSION);
    int size = this->mFiles.size();
    cout << "The following configuration files were detected : " << endl;
    for (int i = 0; i < size; i++) {
        cout << "\t" << (i + 1) << ". " << this->mFiles[i] << endl;
        if (this->mFiles[i].substr(this->mFiles[i].size() - extension.size()) != extension) {
            cout << "\tRegistered file extension is not supported..." << endl << endl;
            erase.push_back(i);
        }
    }
    for (int &i:erase) {
        this->mFiles.erase(this->mFiles.begin() + i);
    }

    for (string &f : this->mFiles) {
        std::ifstream in(f.c_str());
        json temp;
        in >> temp;
        this->mMap.merge_patch(temp);
    }
    return this->mMap;
}

json Parser::GetMap() {
    return this->mMap;
}

const vector<std::string> &Parser::GetFiles() const {
    return this->mFiles;
}
