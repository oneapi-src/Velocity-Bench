/* 
 * Copyright (C) <2023> Intel Corporation
 * 
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License, as published
 * by the Free Software Foundation; either version 2 of the License,
 * or (at your option) any later version.
 *  
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *  
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, see <http://www.gnu.org/licenses/>.
 *  
 * 
 * SPDX-License-Identifier: GPL-2.0-or-later
 * 
 */ 

#include <sycl/sycl.hpp>
#include "SYCLMiner.h"
#include "ethash_sycl_miner_kernel.h"

#include "ethash_sycl_miner_kernel_globals.h"

#include "dpcpp_helper.h"

#include "fnv.dp.hpp"

#define copy(dst, src, count)          \
    for (int i = 0; i != count; ++i) { \
        (dst)[i] = (src)[i];           \
    }

#include "keccak.dp.hpp"

#include "dagger_shuffled.dp.hpp"

void ethash_search(Search_results *g_output, uint64_t start_nonce, sycl::nd_item<1> item_ct1, uint32_t d_dag_size, const sycl::uint2 *keccak_round_constants, hash128_t *d_dag, hash32_t d_header, uint64_t d_target, int *pdShuffleOffsets)
{
    uint32_t const gid = item_ct1.get_group(0) * item_ct1.get_local_range().get(0) + item_ct1.get_local_id(0);
    sycl::uint2    mix[4];
    if (compute_hash(start_nonce + gid, mix, item_ct1, d_dag_size, keccak_round_constants, d_dag, d_header, d_target, pdShuffleOffsets))
        return;
    uint32_t index = LocalDPCT::atomic_fetch_compare_inc((uint32_t *)&g_output->count, 0xffffffff);
    if (index >= MAX_SEARCH_RESULTS)
        return;
    g_output->result[index].gid    = gid;
    g_output->result[index].mix[0] = mix[0].x();
    g_output->result[index].mix[1] = mix[0].y();
    g_output->result[index].mix[2] = mix[1].x();
    g_output->result[index].mix[3] = mix[1].y();
    g_output->result[index].mix[4] = mix[2].x();
    g_output->result[index].mix[5] = mix[2].y();
    g_output->result[index].mix[6] = mix[3].x();
    g_output->result[index].mix[7] = mix[3].y();
}

void dev::eth::SYCLMiner::run_ethash_search(uint32_t gridSize, uint32_t blockSize, sycl::queue *stream, Search_results *g_output, uint64_t start_nonce)
{
    stream->submit([&](sycl::handler &cgh) {
        auto d_dag_size_ptr_ct1             = d_dag_size;
        auto keccak_round_constants_ptr_ct1 = keccak_round_constants;
        auto d_dag_ptr_ct1                  = d_dag;
        auto d_header_ptr_ct1               = d_header;
        auto d_target_ptr_ct1               = d_target;
        auto pdcShuffleOffsets              = pdShuffleOffsets;

        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(gridSize) * sycl::range<1>(blockSize), sycl::range<1>(blockSize)),
            [=](sycl::nd_item<1> item_ct1) {
                ethash_search(g_output, start_nonce, item_ct1, *d_dag_size_ptr_ct1, keccak_round_constants_ptr_ct1, *d_dag_ptr_ct1, *d_header_ptr_ct1, *d_target_ptr_ct1, pdcShuffleOffsets);
            });
    });
}

#define ETHASH_DATASET_PARENTS 256
#define NODE_WORDS             (64 / 4)

void ethash_calculate_dag_item(uint32_t start, sycl::nd_item<1> item_ct1, uint32_t d_dag_size, const sycl::uint2 *keccak_round_constants, hash128_t *d_dag, uint32_t d_light_size, hash64_t *d_light)
{
    uint32_t const node_index = start + item_ct1.get_group(0) * item_ct1.get_local_range().get(0) + item_ct1.get_local_id(0);
    if (((node_index >> 1) & (~1)) >= d_dag_size)
        return;
    union {
        hash128_t   dag_node;
        sycl::uint2 dag_node_mem[25];
    };
    copy(dag_node.uint4s, d_light[node_index % d_light_size].uint4s, 4);
    dag_node.words[0] ^= node_index;
    SHA3_512(dag_node_mem, keccak_round_constants);

    const int thread_id = item_ct1.get_local_id(0) & 3;
    auto g = item_ct1.get_sub_group();
    int const iSubGroupThreadId(g.get_local_id());

    for (uint32_t i = 0; i != ETHASH_DATASET_PARENTS; ++i) {
        uint32_t parent_index = fnv(node_index ^ i, dag_node.words[i % NODE_WORDS]) % d_light_size;
        for (uint32_t t = 0; t < 4; t++) {
            uint32_t shuffle_index = 0;
            if (item_ct1.get_sub_group().get_local_id() < 4)
                shuffle_index = sycl::select_from_group(g, parent_index, t); 
            else if (item_ct1.get_sub_group().get_local_id() < 8)
                shuffle_index = sycl::select_from_group(g, parent_index, t + 4);  
            else if (item_ct1.get_sub_group().get_local_id() < 12)
                shuffle_index = sycl::select_from_group(g, parent_index, t + 8);  
            else if (item_ct1.get_sub_group().get_local_id() < 16)
                shuffle_index = sycl::select_from_group(g, parent_index, t + 12); 
            else if (item_ct1.get_sub_group().get_local_id() < 20)
                shuffle_index = sycl::select_from_group(g, parent_index, t + 16); 
            else if (item_ct1.get_sub_group().get_local_id() < 24)
                shuffle_index = sycl::select_from_group(g, parent_index, t + 20); 
            else if (item_ct1.get_sub_group().get_local_id() < 28)
                shuffle_index = sycl::select_from_group(g, parent_index, t + 24); 
            else if (item_ct1.get_sub_group().get_local_id() < 32)
                shuffle_index = sycl::select_from_group(g, parent_index, t + 28); 

#ifdef USE_AMD_BACKEND // Shuffle on AMD GPUs is done over 64 threads
            else if (item_ct1.get_sub_group().get_local_id() < 36)
                shuffle_index = sycl::select_from_group(g, parent_index, t + 32);
            else if (item_ct1.get_sub_group().get_local_id() < 40)
                shuffle_index = sycl::select_from_group(g, parent_index, t + 36);
            else if (item_ct1.get_sub_group().get_local_id() < 44)
                shuffle_index = sycl::select_from_group(g, parent_index, t + 40);
            else if (item_ct1.get_sub_group().get_local_id() < 48)
                shuffle_index = sycl::select_from_group(g, parent_index, t + 44);
            else if (item_ct1.get_sub_group().get_local_id() < 52)
                shuffle_index = sycl::select_from_group(g, parent_index, t + 48);
            else if (item_ct1.get_sub_group().get_local_id() < 56)
                shuffle_index = sycl::select_from_group(g, parent_index, t + 52);
            else if (item_ct1.get_sub_group().get_local_id() < 60)
                shuffle_index = sycl::select_from_group(g, parent_index, t + 56);
            else if (item_ct1.get_sub_group().get_local_id() < 64)
                shuffle_index = sycl::select_from_group(g, parent_index, t + 60);
#endif

            sycl::uint4 p4 = d_light[shuffle_index].uint4s[thread_id & 3];

            for (int w = 0; w < 4; w++) {
                int w1 = w;

                if (iSubGroupThreadId < 4)
                    w1 = w;
                else if (iSubGroupThreadId < 8)
                    w1 = w + 4;
                else if (iSubGroupThreadId < 12)
                    w1 = w + 8;
                else if (iSubGroupThreadId < 16)
                    w1 = w + 12;
                else if (iSubGroupThreadId < 20)
                    w1 = w + 16;
                else if (iSubGroupThreadId < 24)
                    w1 = w + 20;
                else if (iSubGroupThreadId < 28)
                    w1 = w + 24;
                else if (iSubGroupThreadId < 32)
                    w1 = w + 28;
#ifdef USE_AMD_BACKEND
                else if (iSubGroupThreadId < 36)
                    w1 = w + 32;
                else if (iSubGroupThreadId < 40)
                    w1 = w + 36;
                else if (iSubGroupThreadId < 44)
                    w1 = w + 40;
                else if (iSubGroupThreadId < 48)
                    w1 = w + 44;
                else if (iSubGroupThreadId < 52)
                    w1 = w + 48;
                else if (iSubGroupThreadId < 56)
                    w1 = w + 52;
                else if (iSubGroupThreadId < 60)
                    w1 = w + 56;
                else if (iSubGroupThreadId < 64)
                    w1 = w + 60;
#endif

                sycl::uint4 s4 = sycl::uint4(sycl::select_from_group(item_ct1.get_sub_group(), p4.x(), w1), 
                                             sycl::select_from_group(item_ct1.get_sub_group(), p4.y(), w1), 
                                             sycl::select_from_group(item_ct1.get_sub_group(), p4.z(), w1), 
                                             sycl::select_from_group(item_ct1.get_sub_group(), p4.w(), w1));

                if (t == (thread_id & 3)) {
                    dag_node.uint4s[w] = fnv4(dag_node.uint4s[w], s4);
                }
            }
        }
    }
    SHA3_512(dag_node_mem, keccak_round_constants);
    hash64_t *dag_nodes = (hash64_t *)d_dag;
    copy(dag_nodes[node_index].uint4s, dag_node.uint4s, 4);
}

void dev::eth::SYCLMiner::ethash_generate_dag(uint64_t dag_size, uint32_t gridSize, uint32_t blockSize, sycl::queue *stream)

try {
    const uint32_t work = (uint32_t)(dag_size / sizeof(hash64_t));
    const uint32_t run  = gridSize * blockSize;

    std::cout << "\nCurrent device: " << stream->get_device().get_info<sycl::info::device::name>() << "\n"
              << " Max work group size " << stream->get_device().get_info<sycl::info::device::max_work_group_size>() << "\n";

    uint32_t base(0);
    for (base = 0; base <= work - run; base += run) {
        ////std::cout << "Launching at base: " << base << std::endl;

        stream->submit([&](sycl::handler &cgh) {
            auto d_dag_size_ptr_ct1             = d_dag_size;
            auto keccak_round_constants_ptr_ct1 = keccak_round_constants;
            auto d_dag_ptr_ct1                  = d_dag;
            auto d_light_size_ptr_ct1           = d_light_size;
            auto d_light_ptr_ct1                = d_light;

            cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(gridSize) * sycl::range<1>(blockSize), sycl::range<1>(blockSize)),
                [=](sycl::nd_item<1> item_ct1) {
                    ethash_calculate_dag_item(base, item_ct1, *d_dag_size_ptr_ct1, keccak_round_constants_ptr_ct1, *d_dag_ptr_ct1, *d_light_size_ptr_ct1, *d_light_ptr_ct1);
                });
        });
        stream->wait_and_throw();
    }
    if (base < work) {
        uint32_t lastGrid = work - base;
        lastGrid          = (lastGrid + blockSize - 1) / blockSize;

        stream->submit([&](sycl::handler &cgh) {
            auto d_dag_size_ptr_ct1             = d_dag_size;
            auto keccak_round_constants_ptr_ct1 = keccak_round_constants;
            auto d_dag_ptr_ct1                  = d_dag;
            auto d_light_size_ptr_ct1           = d_light_size;
            auto d_light_ptr_ct1                = d_light;

            cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(lastGrid) * sycl::range<1>(blockSize), sycl::range<1>(blockSize)),
                [=](sycl::nd_item<1> item_ct1) {
                    ethash_calculate_dag_item(base, item_ct1, *d_dag_size_ptr_ct1, keccak_round_constants_ptr_ct1, *d_dag_ptr_ct1, *d_light_size_ptr_ct1, *d_light_ptr_ct1);
                });
        });

        stream->wait_and_throw();
    }
} catch (sycl::exception const &exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__
              << std::endl;
    std::exit(1);
}

void dev::eth::SYCLMiner::set_constants(hash128_t *_dag, uint32_t _dag_size, hash64_t *_light, uint32_t _light_size) ///, sycl::queue &DefaultQueue )
{
    try {
        m_DefaultQueue.memcpy(keccak_round_constants, vKeccakConstants.data(), sizeof(sycl::uint2) * vKeccakConstants.size()).wait();

        m_DefaultQueue.memcpy(d_dag, &_dag, sizeof(hash128_t *)).wait();

        m_DefaultQueue.memcpy(d_dag_size, &_dag_size, sizeof(uint32_t)).wait();

        m_DefaultQueue.memcpy(d_light, &_light, sizeof(hash64_t *)).wait();

        m_DefaultQueue.memcpy(d_light_size, &_light_size, sizeof(uint32_t)).wait();

        m_DefaultQueue.memcpy(pdShuffleOffsets, vShuffleOffsets.data(), sizeof(int) * vShuffleOffsets.size()).wait();

    } catch (sycl::exception const &exc) {
        std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__
                  << std::endl;
        std::exit(1);
    }
}

void dev::eth::SYCLMiner::get_constants(hash128_t **_dag, uint32_t *_dag_size, hash64_t **_light, uint32_t *_light_size) ///, sycl::queue &DefaultQueue)
{
    try {
        /*
           Using the direct address of the targets did not work.
           So I've to read first into local variables when using cudaMemcpyFromSymbol()
        */
        if (_dag) {
            hash128_t *_d;
            m_DefaultQueue.memcpy(&_d, d_dag, sizeof(hash128_t *)).wait();
            *_dag = _d;
        }

        if (_dag_size) {
            uint32_t _ds;
            m_DefaultQueue.memcpy(&_ds, d_dag_size, sizeof(uint32_t)).wait();
            *_dag_size = _ds;
        }

        if (_light) {
            hash64_t *_l;
            m_DefaultQueue.memcpy(&_l, d_light, sizeof(hash64_t *)).wait();
            *_light = _l;
        }

        if (_light_size) {
            uint32_t _ls;
            m_DefaultQueue.memcpy(&_ls, d_light_size, sizeof(uint32_t)).wait();
            *_light_size = _ls;
        }
    } catch (sycl::exception const &exc) {
        std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__
                  << std::endl;
        std::exit(1);
    }
}

void dev::eth::SYCLMiner::set_header(hash32_t _header)
{
    try {
        m_DefaultQueue.memcpy(d_header, &_header, sizeof(hash32_t)).wait();
    } catch (sycl::exception const &exc) {
        std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__
                  << std::endl;
        std::exit(1);
    }
}

void dev::eth::SYCLMiner::set_target(uint64_t _target)
{
    try {
        m_DefaultQueue.memcpy(d_target, &_target, sizeof(uint64_t)).wait();
    } catch (sycl::exception const &exc) {
        std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__
                  << std::endl;
        std::exit(1);
    }
}
