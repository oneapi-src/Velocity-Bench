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
// Created by pancee on 12/15/20.
//

#include <thoth/loaders/BufferLoader.hpp>

#include <iostream>
#include <cstdlib>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

using namespace thoth::streams;

BufferLoader::BufferLoader(int aGapSize) : mGapSize(aGapSize) {
    InitBuffer(this->mGapSize);
};

BufferLoader::BufferLoader(FILE *aFile, int aGapSize) : mGapSize(aGapSize) {

    // determine the size of the file then create
    // a buffer of size + GAP_SIZE
    struct stat buf;

    fstat(fileno(aFile), &buf);
    long file_size = buf.st_size;
    InitBuffer(file_size + GAP_SIZE);
    MoveGapToPoint();
    ExpandGap((int) file_size);
    unsigned int amount = fread(this->mpGapStart, 1, file_size, aFile);

    this->mpGapStart += amount;
}


