/* Modifications Copyright (C) 2023 Intel Corporation
 *
 * This Program is subject to the terms of The Unlicense.​
 * If a copy of the license was not distributed with this file, ​
 * you can obtain one at https://spdx.org/licenses/Unlicense.html​
 *​
 *
 * SPDX-License-Identifier: Unlicense
 */

#include "algorithm"
#include "random"
#include "stdint.h"
#include "stdio.h"
#include "unordered_map"
#include "unordered_set"
#include "vector"
#include "chrono"
#include <cstring>
#include "linearprobing.h"

// #define DEBUG_TIME
#define CPP_MODULE "MAIN"
#define ITERATIONS 10

#define TIMER_START() time_start = std::chrono::steady_clock::now();
#define TIMER_END()                                                                         \
    time_end = std::chrono::steady_clock::now();                                            \
    time_total  = std::chrono::duration<double, std::milli>(time_end - time_start).count();
#define TIMER_PRINT(name) std::cout << name <<": " << time_total / 1e3 << " s\n";

#ifdef DEBUG_TIME
#define START_TIMER() start_time = std::chrono::steady_clock::now();
#define STOP_TIMER()                                                                        \
    stop_time = std::chrono::steady_clock::now();                                           \
    duration  = std::chrono::duration<double, std::milli>(stop_time - start_time).count();  \
    tot_time += duration;
#define PRINT_TIMER(name) std::cout <<name <<"      : " << duration << " ms\n";
#endif

using Time = std::chrono::time_point<std::chrono::steady_clock>;

Time start_timer() 
{
    return std::chrono::steady_clock::now();
}

double get_elapsed_time(Time start) 
{
    Time end = std::chrono::steady_clock::now();

    std::chrono::duration<double> d = end - start;
    std::chrono::microseconds us = std::chrono::duration_cast<std::chrono::microseconds>(d);
    return us.count() / 1000.0f;
}

// Create random keys/values in the range [0, kEmpty)
// kEmpty is used to indicate an empty slot
std::vector<KeyValue> generate_random_keyvalues(
    std::mt19937& rnd,
    uint32_t numkvs)
{
    std::uniform_int_distribution<uint32_t> dis(0, kEmpty - 1);

    std::vector<KeyValue> kvs;
    kvs.reserve(numkvs);

    for (uint32_t i = 0; i < numkvs; i++)
    {
        uint32_t rand0 = dis(rnd);
        uint32_t rand1 = dis(rnd);
        kvs.push_back(KeyValue{rand0, rand1});
    }

    return kvs;
}

// return numshuffledkvs random items from kvs
std::vector<KeyValue> shuffle_keyvalues(
    std::mt19937& rnd,
    std::vector<KeyValue> kvs,
    uint32_t numshuffledkvs)
{
    std::shuffle(kvs.begin(), kvs.end(), rnd);

    std::vector<KeyValue> shuffled_kvs;
    shuffled_kvs.resize(numshuffledkvs);

    std::copy(kvs.begin(), kvs.begin() + numshuffledkvs, shuffled_kvs.begin());

    return shuffled_kvs;
}

void test_unordered_map(
    std::vector<KeyValue> insert_kvs,
    std::vector<KeyValue> delete_kvs) 
{
    Time timer = start_timer();

    printf("Timing std::unordered_map...\n");

    {
        std::unordered_map<uint32_t, uint32_t> kvs_map;
        for (auto& kv : insert_kvs) 
        {
            kvs_map[kv.key] = kv.value;
        }
        for (auto& kv : delete_kvs)
        {
            auto i = kvs_map.find(kv.key);
            if (i != kvs_map.end())
                kvs_map.erase(i);
        }
    }

    double milliseconds = get_elapsed_time(timer);
    double seconds = milliseconds / 1000.0f;
    printf("Total time for std::unordered_map: %f ms (%f Mkeys/second)\n",
        milliseconds, kNumKeyValues / seconds / 1000000.0f);
}

void process_cmdline_arguments(int argc, char* argv[], bool &verify, uint32_t &seed) {
    if (argc == 1) {
        return;
    } else if (argc == 2) {
        if (strcmp(argv[1], "--no-verify") == 0) {
            verify = false;
            return;
        }
    } else if (argc == 4) {
        if (strcmp(argv[1], "--no-verify") == 0 || strcmp(argv[2], "--seed") == 0) {
            verify = false;
            seed = std::stoi(argv[3]);
            return;
        } else if (strcmp(argv[1], "--seed") == 0 || strcmp(argv[3], "--no-verify") == 0) {
            verify = false;
            seed = std::stoi(argv[2]);
            return;
        }
    }
    throw std::runtime_error("Wrong number/type of arguments was provided\n");
}

void test_correctness(
    std::vector<KeyValue>,
    std::vector<KeyValue>,
    std::vector<KeyValue>);

int main(int argc, char* argv[])
{
    std::chrono::steady_clock::time_point time_start;
    std::chrono::steady_clock::time_point time_end;
    double time_total = 0.0;

    try {

    bool verify = true;
    // To recreate the same random numbers across runs of the program, set seed to a specific
    // number instead of a number from random_device
    std::random_device rd;
    uint32_t seed = rd();
    process_cmdline_arguments(argc, argv, verify, seed);
    std::mt19937 rnd(seed);  // mersenne_twister_engine

    // printf("Random number generator seed = %u\n", seed);

    // double seconds;
    // for (uint32_t n = 0; n < NUM_LOOPS; ++n) {
        // printf("Initializing keyvalue pairs with random numbers...\n");

#ifdef DEBUG_TIME
        std::chrono::steady_clock::time_point start_time;
        std::chrono::steady_clock::time_point stop_time;
        double duration = 0.0;
        double tot_time = 0.0;

        START_TIMER();
#endif
        std::vector<KeyValue> insert_kvs = generate_random_keyvalues(rnd, kNumKeyValues);
        std::vector<KeyValue> lookup_kvs = shuffle_keyvalues(rnd, insert_kvs, kNumKeyValues / 2);
        std::vector<KeyValue> delete_kvs = shuffle_keyvalues(rnd, insert_kvs, kNumKeyValues / 2);

#ifdef DEBUG_TIME
STOP_TIMER();
PRINT_TIMER("generate_hashtable ");
#endif

        TIMER_START()

#ifdef DEBUG_TIME
START_TIMER();
#endif
        checkCUDA(cudaSetDevice(0));

#ifdef DEBUG_TIME
STOP_TIMER();
PRINT_TIMER("init               ");
#endif

    std::vector<KeyValue> kvs;
    for (int iter = 0; iter < ITERATIONS; iter++){
#ifdef DEBUG_TIME
START_TIMER();
#endif
        // Allocates device memory for the hashtable and
        // fills every byte with 0xFF (so each key (and value) is set to 0xFFFFFFFF)
        KeyValue* pHashTable = create_hashtable();
#ifdef DEBUG_TIME
STOP_TIMER();
PRINT_TIMER("create_hashtable   ");

START_TIMER();
#endif
        // Insert items into the hash table in batches of num_inserts_per_batch
        const uint32_t num_insert_batches = 16;
        uint32_t num_inserts_per_batch = (uint32_t)insert_kvs.size() / num_insert_batches;
        for (uint32_t i = 0; i < num_insert_batches; i++) {

            insert_hashtable(
                pHashTable,
                insert_kvs.data() + i * num_inserts_per_batch,
                num_inserts_per_batch);
        }
#ifdef DEBUG_TIME
STOP_TIMER();
PRINT_TIMER("insert_hashtable   ");

START_TIMER();
#endif
        // Look up items from the hash table in batches of num_lookups_per_batch
        const uint32_t num_lookup_batches = 8;
        uint32_t num_lookups_per_batch = (uint32_t)lookup_kvs.size() / num_lookup_batches;
        for (uint32_t i = 0; i < num_lookup_batches; i++) {

            lookup_hashtable(
                pHashTable,
                lookup_kvs.data() + i * num_lookups_per_batch,
                num_lookups_per_batch);
        }
#ifdef DEBUG_TIME
STOP_TIMER();
PRINT_TIMER("lookup_hashtable   ");

START_TIMER();
#endif
        // Delete items from the hash table in batches of num_deletes_per_batch
        const uint32_t num_delete_batches = 8;
        uint32_t num_deletes_per_batch = (uint32_t)delete_kvs.size() / num_delete_batches;
        for (uint32_t i = 0; i < num_delete_batches; i++) {

            delete_hashtable(
                pHashTable,
                delete_kvs.data() + i * num_deletes_per_batch,
                num_deletes_per_batch);
        }
#ifdef DEBUG_TIME
STOP_TIMER();
PRINT_TIMER("delete_hashtable   ");

START_TIMER();
#endif
        // Get all the key-values from the hash table
        kvs = iterate_hashtable(pHashTable);
#ifdef DEBUG_TIME
STOP_TIMER();
PRINT_TIMER("iterate_hashtable  ");
#endif
// std::cout << "tot_time: " << tot_time << " ms" << std::endl;

        // Summarize results
        // double milliseconds = get_elapsed_time(timer);
        // seconds = milliseconds / 1000.0f;

        destroy_hashtable(pHashTable);

    }
    TIMER_END()
    TIMER_PRINT("hashtable - total time for whole calculation")
    printf("%f million keys/second\n", kNumKeyValues / (time_total / ITERATIONS / 1000.0f) / 1000000.0f);

        if (verify) {
            test_unordered_map(insert_kvs, delete_kvs);
            test_correctness(insert_kvs, delete_kvs, std::move(kvs));
            printf("Success\n");
        }
    // }
    // printf("Total time: %f s\n", seconds);
    // printf("%f million keys/second\n", kNumKeyValues / seconds / 1000000.0f);
    
    } catch (std::exception const& e) {
        std::cout << "Exception caught, \'" << e.what() << "\'";
        return 1;
    } catch (...) {
        std::cout << "Unknown exception caught, bailing...";
        return 2;
    }

    return 0;
}
