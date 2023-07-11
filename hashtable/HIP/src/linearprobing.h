/* Modifications Copyright (C) 2023 Intel Corporation
 *
 * This Program is subject to the terms of The Unlicense.​
 * If a copy of the license was not distributed with this file, ​
 * you can obtain one at https://spdx.org/licenses/Unlicense.html​
 *​
 *
 * SPDX-License-Identifier: Unlicense
 */

#pragma once

#include "hip/hip_runtime.h"
#include <iostream>
#include <sstream>

#ifndef CPP_MODULE
#define CPP_MODULE "UNKN"
#endif

struct KeyValue
{
    uint32_t key;
    uint32_t value;
};

const uint32_t kHashTableCapacity = 256 * 1024 * 1024;

const uint32_t kNumKeyValues = kHashTableCapacity / 2;

const uint32_t kEmpty = 0xFFFFFFFF;

const uint32_t NUM_LOOPS = 1;

KeyValue* create_hashtable();

void insert_hashtable(KeyValue* hashtable, const KeyValue* kvs, uint32_t num_kvs);
void lookup_hashtable(KeyValue* hashtable,       KeyValue* kvs, uint32_t num_kvs);
void delete_hashtable(KeyValue* hashtable, const KeyValue* kvs, uint32_t num_kvs);

std::vector<KeyValue> iterate_hashtable(KeyValue* hashtable);

void destroy_hashtable(KeyValue* hashtable);

#define checkCUDA(expression)                                   \
{                                                               \
    hipError_t const status(expression);                        \
    if (status != hipSuccess) {                                 \
        std::stringstream sErrorMessage;                        \
        sErrorMessage << "Error on line " << __LINE__ << ": "   \
                      << hipGetErrorString(status) << "\n";     \
        throw std::runtime_error(sErrorMessage.str());          \
        std::exit(EXIT_FAILURE);                                \
    }                                                           \
}

#define LOG_ERROR(msg)                                                              \
{                                                                                   \
    std::stringstream sErrorMessage;                                                \
    sErrorMessage << CPP_MODULE << " ERROR(" << __LINE__<< "): " <<  msg << "\n";   \
    std::cerr << sErrorMessage.str();                                               \
    throw std::runtime_error(sErrorMessage.str());                                  \
}
