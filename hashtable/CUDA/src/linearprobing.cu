/*​ Modifications Copyright (C) 2023 Intel Corporation
 *
 * This Program is subject to the terms of The Unlicense.​
 * If a copy of the license was not distributed with this file, ​
 * you can obtain one at https://spdx.org/licenses/Unlicense.html​
 *​
 *
 * SPDX-License-Identifier: Unlicense
 */

#include "stdio.h"
#include "stdint.h"
#include "vector"

#define CPP_MODULE "KERNEL"
#include "linearprobing.h"

// 32 bit Murmur3 hash
__device__
uint32_t hash(uint32_t k)
{
    k ^= k >> 16;
    k *= 0x85ebca6b;
    k ^= k >> 13;
    k *= 0xc2b2ae35;
    k ^= k >> 16;
    return k & (kHashTableCapacity - 1);
}

// Create a hash table. For linear probing, this is just an array of KeyValues
KeyValue* create_hashtable()
{
    KeyValue* hashtable;

    try {
        // Allocate memory
        checkCUDA(cudaMalloc(&hashtable, sizeof(KeyValue) * kHashTableCapacity));

        // Initialize hash table to empty
        static_assert(kEmpty == 0xFFFFFFFF, "memset expected kEmpty=0xFFFFFFFF");
        checkCUDA(cudaMemset(hashtable, 0xFF, sizeof(KeyValue) * kHashTableCapacity));
        checkCUDA(cudaDeviceSynchronize());
    } catch (std::exception const& e) {
        LOG_ERROR("Exception caught, \'" << e.what() << "\'");
    } catch (...) {
        LOG_ERROR("Unknown exception caught, bailing...");
    }

    return hashtable;
}

// Insert the key/values in kvs into the hashtable
__global__
void gpu_hashtable_insert(
    KeyValue* hashtable,
    const KeyValue* kvs,
    unsigned int numkvs)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numkvs) {
        uint32_t key   = kvs[tid].key;
        uint32_t value = kvs[tid].value;
        uint32_t slot  = hash(key);

        while (true) {
            uint32_t prev = atomicCAS(&hashtable[slot].key, kEmpty, key);
            if (prev == kEmpty || prev == key) {
                hashtable[slot].value = value;
                return;
            }

            slot = (slot + 1) & (kHashTableCapacity - 1);
        }
    }
}

void insert_hashtable(
    KeyValue* pHashTable, // hashtable
    const KeyValue* kvs,  // starting position for this batch of key-value pairs
    uint32_t num_kvs)     // number of key-value pairs in this batch
{
    try {
        // Copy this batch of key-value pairs to the device
        KeyValue* device_kvs;
        checkCUDA(cudaMalloc(&device_kvs, sizeof(KeyValue) * num_kvs));
        checkCUDA(cudaMemcpy(device_kvs, kvs, sizeof(KeyValue) * num_kvs, cudaMemcpyHostToDevice));

#ifdef CALC_BLOCK
        // Have CUDA calculate the thread block size
        int mingridsize;
        int threadblocksize = 256;
        checkCUDA(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, gpu_hashtable_insert, 0, 0));
        std::cout << "insert threadblocksize: " << threadblocksize << std::endl;
#else
        int threadblocksize = 1024; // perf does not seem to vary w/ thread block size (for all kernels in hashtable)
#endif

        int gridsize = ((uint32_t)num_kvs + threadblocksize - 1) / threadblocksize;

        gpu_hashtable_insert<<<gridsize, threadblocksize>>>(
            pHashTable,
            device_kvs,
            (uint32_t)num_kvs);
        CUDA_CHECK_LAST_ERROR();
        checkCUDA(cudaDeviceSynchronize());

        checkCUDA(cudaFree(device_kvs));
    } catch (std::exception const& e) {
        LOG_ERROR("Exception caught, \'" << e.what() << "\'");
    } catch (...) {
        LOG_ERROR("Unknown exception caught, bailing...");
    }
}

// Lookup keys in the hashtable, and return the values
__global__
void gpu_hashtable_lookup(
    KeyValue* hashtable,
    KeyValue* kvs,
    unsigned int numkvs)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numkvs) {
        uint32_t key  = kvs[tid].key;
        uint32_t slot = hash(key);

        while (true) {
            if (hashtable[slot].key == key) {
                kvs[tid].value = hashtable[slot].value;
                return;
            }
            if (hashtable[slot].key == kEmpty) {
                kvs[tid].value = kEmpty;
                return;
            }
            slot = (slot + 1) & (kHashTableCapacity - 1);
        }
    }
}

void lookup_hashtable(
    KeyValue* pHashTable,
    KeyValue* kvs,
    uint32_t num_kvs)
{
    try {
        // Copy this batch of key-value pairs to the device
        KeyValue* device_kvs;
        checkCUDA(cudaMalloc(&device_kvs, sizeof(KeyValue) * num_kvs));
        checkCUDA(cudaMemcpy(device_kvs, kvs, sizeof(KeyValue) * num_kvs, cudaMemcpyHostToDevice));

#ifdef CALC_BLOCK
        // Have CUDA calculate the thread block size
        int mingridsize;
        int threadblocksize = 256;
        checkCUDA(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, gpu_hashtable_insert, 0, 0));
        std::cout << "lookup threadblocksize: " << threadblocksize << std::endl;
#else
        int threadblocksize = 1024;
#endif

        int gridsize = ((uint32_t)num_kvs + threadblocksize - 1) / threadblocksize;

        gpu_hashtable_lookup<<<gridsize, threadblocksize>>>(
            pHashTable,
            device_kvs,
            (uint32_t)num_kvs);
        CUDA_CHECK_LAST_ERROR();
        checkCUDA(cudaDeviceSynchronize());

        checkCUDA(cudaFree(device_kvs));
    } catch (std::exception const& e) {
        LOG_ERROR("Exception caught, \'" << e.what() << "\'");
    } catch (...) {
        LOG_ERROR("Unknown exception caught, bailing...");
    }
}

// Delete each key in kvs from the hash table, if the key exists
// A deleted key is left in the hash table, but its value is set to kEmpty
// Deleted keys are not reused; once a key is assigned a slot, it never moves
__global__
void gpu_hashtable_delete(
    KeyValue* hashtable,
    const KeyValue* kvs,
    unsigned int numkvs)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numkvs) {
        uint32_t key  = kvs[tid].key;
        uint32_t slot = hash(key);

        while (true) {
            if (hashtable[slot].key == key) {
                hashtable[slot].value = kEmpty;
                return;
            }
            if (hashtable[slot].key == kEmpty) {
                return;
            }
            slot = (slot + 1) & (kHashTableCapacity - 1);
        }
    }
}

void delete_hashtable(
    KeyValue* pHashTable,
    const KeyValue* kvs,
    uint32_t num_kvs)
{
    try {
        // Copy the keyvalues to the GPU
        KeyValue* device_kvs;
        checkCUDA(cudaMalloc(&device_kvs, sizeof(KeyValue) * num_kvs));
        checkCUDA(cudaMemcpy(device_kvs, kvs, sizeof(KeyValue) * num_kvs, cudaMemcpyHostToDevice));

#ifdef CALC_BLOCK
        // Have CUDA calculate the thread block size
        int mingridsize;
        int threadblocksize = 256;
        checkCUDA(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, gpu_hashtable_insert, 0, 0));
        std::cout << "delete threadblocksize: " << threadblocksize << std::endl;
#else
        int threadblocksize = 1024;
#endif

        int gridsize = ((uint32_t)num_kvs + threadblocksize - 1) / threadblocksize;

        gpu_hashtable_delete<<<gridsize, threadblocksize>>>(
            pHashTable,
            device_kvs,
            (uint32_t)num_kvs);
        CUDA_CHECK_LAST_ERROR();
        checkCUDA(cudaDeviceSynchronize());

        checkCUDA(cudaFree(device_kvs));
    } catch (std::exception const& e) {
        LOG_ERROR("Exception caught, \'" << e.what() << "\'");
    } catch (...) {
        LOG_ERROR("Unknown exception caught, bailing...");
    }
}

// Iterate over every item in the hashtable; return non-empty key/values
__global__
void gpu_iterate_hashtable(
    KeyValue* pHashTable,
    KeyValue* kvs,
    uint32_t* kvs_size)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < kHashTableCapacity) {
        if (pHashTable[tid].key != kEmpty) {
            uint32_t value = pHashTable[tid].value;
            if (value != kEmpty) {
                uint32_t size = atomicAdd(kvs_size, 1);
                kvs[size] = pHashTable[tid];
            }
        }
    }
}

std::vector<KeyValue> iterate_hashtable(KeyValue* pHashTable)
{
    std::vector<KeyValue> kvs;

    try {
        uint32_t* device_num_kvs;
        KeyValue* device_kvs;
        checkCUDA(cudaMalloc(&device_num_kvs, sizeof(uint32_t)));
        checkCUDA(cudaMalloc(&device_kvs, sizeof(KeyValue) * kNumKeyValues));

        checkCUDA(cudaMemset(device_num_kvs, 0, sizeof(uint32_t)));

#ifdef CALC_BLOCK
        // Have CUDA calculate the thread block size
        int mingridsize;
        int threadblocksize = 256;
        checkCUDA(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, gpu_iterate_hashtable, 0, 0));
        std::cout << "iterate threadblocksize: " << threadblocksize << std::endl;
#else
        int threadblocksize = 1024;
#endif

        int gridsize = (kHashTableCapacity + threadblocksize - 1) / threadblocksize;

        gpu_iterate_hashtable<<<gridsize, threadblocksize>>>(
            pHashTable,
            device_kvs,
            device_num_kvs);
        CUDA_CHECK_LAST_ERROR();
        uint32_t num_kvs;
        checkCUDA(cudaMemcpy(&num_kvs, device_num_kvs, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        checkCUDA(cudaDeviceSynchronize());

        kvs.resize(num_kvs);

        checkCUDA(cudaMemcpy(kvs.data(), device_kvs, sizeof(KeyValue) * num_kvs, cudaMemcpyDeviceToHost));
        checkCUDA(cudaDeviceSynchronize());

        checkCUDA(cudaFree(device_kvs));
        checkCUDA(cudaFree(device_num_kvs));
    } catch (std::exception const& e) {
        LOG_ERROR("Exception caught, \'" << e.what() << "\'");
    } catch (...) {
        LOG_ERROR("Unknown exception caught, bailing...");
    }

    return kvs;
}

// Free the memory of the hashtable
void destroy_hashtable(KeyValue* pHashTable)
{
    checkCUDA(cudaFree(pHashTable));
}
