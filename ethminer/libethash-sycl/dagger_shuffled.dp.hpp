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

#include <sycl.hpp>
#include "ethash_sycl_miner_kernel_globals.h"

#include "ethash_sycl_miner_kernel.h"

#include "dpcpp_helper.h"

#define _PARALLEL_HASH 4

#ifdef USE_LOOP_UNROLLING
#define mix_and_shuffle(t, a, p, b, thread_id)                                   \
    offset[p] = fnv(init0[p] ^ (a + b), ((uint32_t *)&mix[p])[b]) % d_dag_size;  \
    offset[p] = item_ct1.get_sub_group().shuffle(offset[p], t + iShuffleOffset); \
    mix[p]    = fnv4(mix[p], d_dag[offset[p]].uint4s[thread_id]);
#endif

DEV_INLINE bool compute_hash(uint64_t nonce, sycl::uint2 *mix_hash, sycl::nd_item<1> item_ct1, uint32_t d_dag_size, const sycl::uint2 *keccak_round_constants, hash128_t *d_dag, hash32_t d_header, uint64_t d_target, int *pdShuffleOffsets)
{
    // sha3_512(header .. nonce)
    sycl::uint2 state[12];

    state[4] = vectorize(nonce);

    keccak_f1600_init(state, keccak_round_constants, d_header);

    // Threads work together in this phase in groups of 8.
    const int thread_id = item_ct1.get_local_id(0) & (THREADS_PER_HASH - 1);
    const int mix_idx   = thread_id & 3;

    int const iSubGroupThreadId(item_ct1.get_sub_group().get_local_id());
    int const iShuffleOffset(pdShuffleOffsets[iSubGroupThreadId]);

#ifndef USE_LOOP_UNROLLING

    for (int i = 0; i < THREADS_PER_HASH; i += _PARALLEL_HASH) {
        sycl::uint4 mix[_PARALLEL_HASH];
        uint32_t    offset[_PARALLEL_HASH];
        uint32_t    init0[_PARALLEL_HASH];

        // share init among threads
        for (int p = 0; p < _PARALLEL_HASH; p++) {
            sycl::uint2 shuffle[8];
            shuffle[0].x() = item_ct1.get_sub_group().shuffle(state[0].x(), i + p + iShuffleOffset);
            shuffle[0].y() = item_ct1.get_sub_group().shuffle(state[0].y(), i + p + iShuffleOffset);

            shuffle[1].x() = item_ct1.get_sub_group().shuffle(state[1].x(), i + p + iShuffleOffset);
            shuffle[1].y() = item_ct1.get_sub_group().shuffle(state[1].y(), i + p + iShuffleOffset);

            shuffle[2].x() = item_ct1.get_sub_group().shuffle(state[2].x(), i + p + iShuffleOffset);
            shuffle[2].y() = item_ct1.get_sub_group().shuffle(state[2].y(), i + p + iShuffleOffset);

            shuffle[3].x() = item_ct1.get_sub_group().shuffle(state[3].x(), i + p + iShuffleOffset);
            shuffle[3].y() = item_ct1.get_sub_group().shuffle(state[3].y(), i + p + iShuffleOffset);

            shuffle[4].x() = item_ct1.get_sub_group().shuffle(state[4].x(), i + p + iShuffleOffset);
            shuffle[4].y() = item_ct1.get_sub_group().shuffle(state[4].y(), i + p + iShuffleOffset);

            shuffle[5].x() = item_ct1.get_sub_group().shuffle(state[5].x(), i + p + iShuffleOffset);
            shuffle[5].y() = item_ct1.get_sub_group().shuffle(state[5].y(), i + p + iShuffleOffset);

            shuffle[6].x() = item_ct1.get_sub_group().shuffle(state[6].x(), i + p + iShuffleOffset);
            shuffle[6].y() = item_ct1.get_sub_group().shuffle(state[6].y(), i + p + iShuffleOffset);

            shuffle[7].x() = item_ct1.get_sub_group().shuffle(state[7].x(), i + p + iShuffleOffset);
            shuffle[7].y() = item_ct1.get_sub_group().shuffle(state[7].y(), i + p + iShuffleOffset);

            assert(mix_idx <= 3);

            switch (mix_idx) {
            case 0:
                mix[p] = vectorize2(shuffle[0], shuffle[1]);
                break;
            case 1:
                mix[p] = vectorize2(shuffle[2], shuffle[3]);
                break;
            case 2:
                mix[p] = vectorize2(shuffle[4], shuffle[5]);
                break;
            default:
              mix[p] = vectorize2(shuffle[6], shuffle[7]);
              break;
            }

            init0[p] = item_ct1.get_sub_group().shuffle(shuffle[0].x(), iShuffleOffset);
        }

        for (uint32_t a = 0; a < ACCESSES; a += 4)
        {
            int t = bfe(a, 2u, 3u);
#pragma unroll
            for (uint32_t b = 0; b < 4; b++)
            {
#pragma unroll
                for (int p = 0; p < _PARALLEL_HASH; p++)
                {
                    offset[p] = fnv(init0[p] ^ (a + b), ((uint32_t*)&mix[p])[b]) % d_dag_size;

                    offset[p] = item_ct1.get_sub_group().shuffle(offset[p], t + iShuffleOffset);
                    mix[p] = fnv4(mix[p], d_dag[offset[p]].uint4s[thread_id]);
                }
            }
        }

        for (int p = 0; p < _PARALLEL_HASH; p++) {
            sycl::uint2 shuffle[4];
            uint32_t    thread_mix = fnv_reduce(mix[p]);

            // update mix across threads

            shuffle[0].x() = item_ct1.get_sub_group().shuffle(thread_mix, 0 + iShuffleOffset);
            shuffle[0].y() = item_ct1.get_sub_group().shuffle(thread_mix, 1 + iShuffleOffset);
            shuffle[1].x() = item_ct1.get_sub_group().shuffle(thread_mix, 2 + iShuffleOffset);
            shuffle[1].y() = item_ct1.get_sub_group().shuffle(thread_mix, 3 + iShuffleOffset);
            shuffle[2].x() = item_ct1.get_sub_group().shuffle(thread_mix, 4 + iShuffleOffset);
            shuffle[2].y() = item_ct1.get_sub_group().shuffle(thread_mix, 5 + iShuffleOffset);
            shuffle[3].x() = item_ct1.get_sub_group().shuffle(thread_mix, 6 + iShuffleOffset);
            shuffle[3].y() = item_ct1.get_sub_group().shuffle(thread_mix, 7 + iShuffleOffset);

            if ((i + p) == thread_id) {
                // move mix into state:
                state[8]  = shuffle[0];
                state[9]  = shuffle[1];
                state[10] = shuffle[2];
                state[11] = shuffle[3];
            }
        }
    }
#else
    for (int i = 0; i < THREADS_PER_HASH; i += _PARALLEL_HASH) {
        sycl::uint4 mix[_PARALLEL_HASH];
        uint32_t    offset[_PARALLEL_HASH];
        uint32_t    init0[_PARALLEL_HASH];

        // share init among threads
        for (int p = 0; p < _PARALLEL_HASH; p++) {
            sycl::uint2 shuffle_0, shuffle_1, shuffle_2, shuffle_3, shuffle_4, shuffle_5, shuffle_6, shuffle_7;
            //////sycl::uint2 shuffle[8];
            //////for (int j = 0; j < 8; j++)
            //////{
            shuffle_0.x() = item_ct1.get_sub_group().shuffle(state[0].x(), i + p + iShuffleOffset);
            shuffle_0.y() = item_ct1.get_sub_group().shuffle(state[0].y(), i + p + iShuffleOffset);

            shuffle_1.x() = item_ct1.get_sub_group().shuffle(state[1].x(), i + p + iShuffleOffset);
            shuffle_1.y() = item_ct1.get_sub_group().shuffle(state[1].y(), i + p + iShuffleOffset);

            shuffle_2.x() = item_ct1.get_sub_group().shuffle(state[2].x(), i + p + iShuffleOffset);
            shuffle_2.y() = item_ct1.get_sub_group().shuffle(state[2].y(), i + p + iShuffleOffset);

            shuffle_3.x() = item_ct1.get_sub_group().shuffle(state[3].x(), i + p + iShuffleOffset);
            shuffle_3.y() = item_ct1.get_sub_group().shuffle(state[3].y(), i + p + iShuffleOffset);

            shuffle_4.x() = item_ct1.get_sub_group().shuffle(state[4].x(), i + p + iShuffleOffset);
            shuffle_4.y() = item_ct1.get_sub_group().shuffle(state[4].y(), i + p + iShuffleOffset);

            shuffle_5.x() = item_ct1.get_sub_group().shuffle(state[5].x(), i + p + iShuffleOffset);
            shuffle_5.y() = item_ct1.get_sub_group().shuffle(state[5].y(), i + p + iShuffleOffset);

            shuffle_6.x() = item_ct1.get_sub_group().shuffle(state[6].x(), i + p + iShuffleOffset);
            shuffle_6.y() = item_ct1.get_sub_group().shuffle(state[6].y(), i + p + iShuffleOffset);

            shuffle_7.x() = item_ct1.get_sub_group().shuffle(state[7].x(), i + p + iShuffleOffset);
            shuffle_7.y() = item_ct1.get_sub_group().shuffle(state[7].y(), i + p + iShuffleOffset);

            /////}
            assert(mix_idx <= 3);

            switch (mix_idx) {
            case 0:
                mix[p] = vectorize2(shuffle_0, shuffle_1);
                break;
            case 1:
                mix[p] = vectorize2(shuffle_2, shuffle_3);
                break;
            case 2:
                mix[p] = vectorize2(shuffle_4, shuffle_5);
                break;
            default:
              mix[p] = vectorize2(shuffle_6, shuffle_7);
              break;
            }

            init0[p] = item_ct1.get_sub_group().shuffle(shuffle_0.x(), iShuffleOffset);
        }

        //////for (uint32_t a = 0; a < ACCESSES; a += 4)
        //////{
        int t(0); ///= bfe(0, 2u, 3u);
                  /////for (uint32_t b = 0; b < 4; b++)
        ///////{
        t = bfe(0, 2u, 3u);
        mix_and_shuffle(t, 0 /*a*/, 0 /*p*/, 0 /*b*/, thread_id);
        mix_and_shuffle(t, 0 /*a*/, 1 /*p*/, 0 /*b*/, thread_id);
        mix_and_shuffle(t, 0 /*a*/, 2 /*p*/, 0 /*b*/, thread_id);
        mix_and_shuffle(t, 0 /*a*/, 3 /*p*/, 0 /*b*/, thread_id);

        mix_and_shuffle(t, 0 /*a*/, 0 /*p*/, 1 /*b*/, thread_id);
        mix_and_shuffle(t, 0 /*a*/, 1 /*p*/, 1 /*b*/, thread_id);
        mix_and_shuffle(t, 0 /*a*/, 2 /*p*/, 1 /*b*/, thread_id);
        mix_and_shuffle(t, 0 /*a*/, 3 /*p*/, 1 /*b*/, thread_id);

        mix_and_shuffle(t, 0 /*a*/, 0 /*p*/, 2 /*b*/, thread_id);
        mix_and_shuffle(t, 0 /*a*/, 1 /*p*/, 2 /*b*/, thread_id);
        mix_and_shuffle(t, 0 /*a*/, 2 /*p*/, 2 /*b*/, thread_id);
        mix_and_shuffle(t, 0 /*a*/, 3 /*p*/, 2 /*b*/, thread_id);

        mix_and_shuffle(t, 0 /*a*/, 0 /*p*/, 3 /*b*/, thread_id);
        mix_and_shuffle(t, 0 /*a*/, 1 /*p*/, 3 /*b*/, thread_id);
        mix_and_shuffle(t, 0 /*a*/, 2 /*p*/, 3 /*b*/, thread_id);
        mix_and_shuffle(t, 0 /*a*/, 3 /*p*/, 3 /*b*/, thread_id);

        t = bfe(4, 2u, 3u);
        mix_and_shuffle(t, 4 /*a*/, 0 /*p*/, 0 /*b*/, thread_id);
        mix_and_shuffle(t, 4 /*a*/, 1 /*p*/, 0 /*b*/, thread_id);
        mix_and_shuffle(t, 4 /*a*/, 2 /*p*/, 0 /*b*/, thread_id);
        mix_and_shuffle(t, 4 /*a*/, 3 /*p*/, 0 /*b*/, thread_id);

        mix_and_shuffle(t, 4 /*a*/, 0 /*p*/, 1 /*b*/, thread_id);
        mix_and_shuffle(t, 4 /*a*/, 1 /*p*/, 1 /*b*/, thread_id);
        mix_and_shuffle(t, 4 /*a*/, 2 /*p*/, 1 /*b*/, thread_id);
        mix_and_shuffle(t, 4 /*a*/, 3 /*p*/, 1 /*b*/, thread_id);

        mix_and_shuffle(t, 4 /*a*/, 0 /*p*/, 2 /*b*/, thread_id);
        mix_and_shuffle(t, 4 /*a*/, 1 /*p*/, 2 /*b*/, thread_id);
        mix_and_shuffle(t, 4 /*a*/, 2 /*p*/, 2 /*b*/, thread_id);
        mix_and_shuffle(t, 4 /*a*/, 3 /*p*/, 2 /*b*/, thread_id);

        mix_and_shuffle(t, 4 /*a*/, 0 /*p*/, 3 /*b*/, thread_id);
        mix_and_shuffle(t, 4 /*a*/, 1 /*p*/, 3 /*b*/, thread_id);
        mix_and_shuffle(t, 4 /*a*/, 2 /*p*/, 3 /*b*/, thread_id);
        mix_and_shuffle(t, 4 /*a*/, 3 /*p*/, 3 /*b*/, thread_id);

        t = bfe(8, 2u, 3u);
        mix_and_shuffle(t, 8 /*a*/, 0 /*p*/, 0 /*b*/, thread_id);
        mix_and_shuffle(t, 8 /*a*/, 1 /*p*/, 0 /*b*/, thread_id);
        mix_and_shuffle(t, 8 /*a*/, 2 /*p*/, 0 /*b*/, thread_id);
        mix_and_shuffle(t, 8 /*a*/, 3 /*p*/, 0 /*b*/, thread_id);

        mix_and_shuffle(t, 8 /*a*/, 0 /*p*/, 1 /*b*/, thread_id);
        mix_and_shuffle(t, 8 /*a*/, 1 /*p*/, 1 /*b*/, thread_id);
        mix_and_shuffle(t, 8 /*a*/, 2 /*p*/, 1 /*b*/, thread_id);
        mix_and_shuffle(t, 8 /*a*/, 3 /*p*/, 1 /*b*/, thread_id);

        mix_and_shuffle(t, 8 /*a*/, 0 /*p*/, 2 /*b*/, thread_id);
        mix_and_shuffle(t, 8 /*a*/, 1 /*p*/, 2 /*b*/, thread_id);
        mix_and_shuffle(t, 8 /*a*/, 2 /*p*/, 2 /*b*/, thread_id);
        mix_and_shuffle(t, 8 /*a*/, 3 /*p*/, 2 /*b*/, thread_id);

        mix_and_shuffle(t, 8 /*a*/, 0 /*p*/, 3 /*b*/, thread_id);
        mix_and_shuffle(t, 8 /*a*/, 1 /*p*/, 3 /*b*/, thread_id);
        mix_and_shuffle(t, 8 /*a*/, 2 /*p*/, 3 /*b*/, thread_id);
        mix_and_shuffle(t, 8 /*a*/, 3 /*p*/, 3 /*b*/, thread_id);

        t = bfe(12, 2u, 3u);
        mix_and_shuffle(t, 12 /*a*/, 0 /*p*/, 0 /*b*/, thread_id);
        mix_and_shuffle(t, 12 /*a*/, 1 /*p*/, 0 /*b*/, thread_id);
        mix_and_shuffle(t, 12 /*a*/, 2 /*p*/, 0 /*b*/, thread_id);
        mix_and_shuffle(t, 12 /*a*/, 3 /*p*/, 0 /*b*/, thread_id);

        mix_and_shuffle(t, 12 /*a*/, 0 /*p*/, 1 /*b*/, thread_id);
        mix_and_shuffle(t, 12 /*a*/, 1 /*p*/, 1 /*b*/, thread_id);
        mix_and_shuffle(t, 12 /*a*/, 2 /*p*/, 1 /*b*/, thread_id);
        mix_and_shuffle(t, 12 /*a*/, 3 /*p*/, 1 /*b*/, thread_id);

        mix_and_shuffle(t, 12 /*a*/, 0 /*p*/, 2 /*b*/, thread_id);
        mix_and_shuffle(t, 12 /*a*/, 1 /*p*/, 2 /*b*/, thread_id);
        mix_and_shuffle(t, 12 /*a*/, 2 /*p*/, 2 /*b*/, thread_id);
        mix_and_shuffle(t, 12 /*a*/, 3 /*p*/, 2 /*b*/, thread_id);

        mix_and_shuffle(t, 12 /*a*/, 0 /*p*/, 3 /*b*/, thread_id);
        mix_and_shuffle(t, 12 /*a*/, 1 /*p*/, 3 /*b*/, thread_id);
        mix_and_shuffle(t, 12 /*a*/, 2 /*p*/, 3 /*b*/, thread_id);
        mix_and_shuffle(t, 12 /*a*/, 3 /*p*/, 3 /*b*/, thread_id);

        t = bfe(16, 2u, 3u);
        mix_and_shuffle(t, 16 /*a*/, 0 /*p*/, 0 /*b*/, thread_id);
        mix_and_shuffle(t, 16 /*a*/, 1 /*p*/, 0 /*b*/, thread_id);
        mix_and_shuffle(t, 16 /*a*/, 2 /*p*/, 0 /*b*/, thread_id);
        mix_and_shuffle(t, 16 /*a*/, 3 /*p*/, 0 /*b*/, thread_id);

        mix_and_shuffle(t, 16 /*a*/, 0 /*p*/, 1 /*b*/, thread_id);
        mix_and_shuffle(t, 16 /*a*/, 1 /*p*/, 1 /*b*/, thread_id);
        mix_and_shuffle(t, 16 /*a*/, 2 /*p*/, 1 /*b*/, thread_id);
        mix_and_shuffle(t, 16 /*a*/, 3 /*p*/, 1 /*b*/, thread_id);

        mix_and_shuffle(t, 16 /*a*/, 0 /*p*/, 2 /*b*/, thread_id);
        mix_and_shuffle(t, 16 /*a*/, 1 /*p*/, 2 /*b*/, thread_id);
        mix_and_shuffle(t, 16 /*a*/, 2 /*p*/, 2 /*b*/, thread_id);
        mix_and_shuffle(t, 16 /*a*/, 3 /*p*/, 2 /*b*/, thread_id);

        mix_and_shuffle(t, 16 /*a*/, 0 /*p*/, 3 /*b*/, thread_id);
        mix_and_shuffle(t, 16 /*a*/, 1 /*p*/, 3 /*b*/, thread_id);
        mix_and_shuffle(t, 16 /*a*/, 2 /*p*/, 3 /*b*/, thread_id);
        mix_and_shuffle(t, 16 /*a*/, 3 /*p*/, 3 /*b*/, thread_id);

        t = bfe(20, 2u, 3u);
        mix_and_shuffle(t, 20 /*a*/, 0 /*p*/, 0 /*b*/, thread_id);
        mix_and_shuffle(t, 20 /*a*/, 1 /*p*/, 0 /*b*/, thread_id);
        mix_and_shuffle(t, 20 /*a*/, 2 /*p*/, 0 /*b*/, thread_id);
        mix_and_shuffle(t, 20 /*a*/, 3 /*p*/, 0 /*b*/, thread_id);

        mix_and_shuffle(t, 20 /*a*/, 0 /*p*/, 1 /*b*/, thread_id);
        mix_and_shuffle(t, 20 /*a*/, 1 /*p*/, 1 /*b*/, thread_id);
        mix_and_shuffle(t, 20 /*a*/, 2 /*p*/, 1 /*b*/, thread_id);
        mix_and_shuffle(t, 20 /*a*/, 3 /*p*/, 1 /*b*/, thread_id);

        mix_and_shuffle(t, 20 /*a*/, 0 /*p*/, 2 /*b*/, thread_id);
        mix_and_shuffle(t, 20 /*a*/, 1 /*p*/, 2 /*b*/, thread_id);
        mix_and_shuffle(t, 20 /*a*/, 2 /*p*/, 2 /*b*/, thread_id);
        mix_and_shuffle(t, 20 /*a*/, 3 /*p*/, 2 /*b*/, thread_id);

        mix_and_shuffle(t, 20 /*a*/, 0 /*p*/, 3 /*b*/, thread_id);
        mix_and_shuffle(t, 20 /*a*/, 1 /*p*/, 3 /*b*/, thread_id);
        mix_and_shuffle(t, 20 /*a*/, 2 /*p*/, 3 /*b*/, thread_id);
        mix_and_shuffle(t, 20 /*a*/, 3 /*p*/, 3 /*b*/, thread_id);

        t = bfe(24, 2u, 3u);
        mix_and_shuffle(t, 24 /*a*/, 0 /*p*/, 0 /*b*/, thread_id);
        mix_and_shuffle(t, 24 /*a*/, 1 /*p*/, 0 /*b*/, thread_id);
        mix_and_shuffle(t, 24 /*a*/, 2 /*p*/, 0 /*b*/, thread_id);
        mix_and_shuffle(t, 24 /*a*/, 3 /*p*/, 0 /*b*/, thread_id);

        mix_and_shuffle(t, 24 /*a*/, 0 /*p*/, 1 /*b*/, thread_id);
        mix_and_shuffle(t, 24 /*a*/, 1 /*p*/, 1 /*b*/, thread_id);
        mix_and_shuffle(t, 24 /*a*/, 2 /*p*/, 1 /*b*/, thread_id);
        mix_and_shuffle(t, 24 /*a*/, 3 /*p*/, 1 /*b*/, thread_id);

        mix_and_shuffle(t, 24 /*a*/, 0 /*p*/, 2 /*b*/, thread_id);
        mix_and_shuffle(t, 24 /*a*/, 1 /*p*/, 2 /*b*/, thread_id);
        mix_and_shuffle(t, 24 /*a*/, 2 /*p*/, 2 /*b*/, thread_id);
        mix_and_shuffle(t, 24 /*a*/, 3 /*p*/, 2 /*b*/, thread_id);

        mix_and_shuffle(t, 24 /*a*/, 0 /*p*/, 3 /*b*/, thread_id);
        mix_and_shuffle(t, 24 /*a*/, 1 /*p*/, 3 /*b*/, thread_id);
        mix_and_shuffle(t, 24 /*a*/, 2 /*p*/, 3 /*b*/, thread_id);
        mix_and_shuffle(t, 24 /*a*/, 3 /*p*/, 3 /*b*/, thread_id);

        t = bfe(28, 2u, 3u);
        mix_and_shuffle(t, 28 /*a*/, 0 /*p*/, 0 /*b*/, thread_id);
        mix_and_shuffle(t, 28 /*a*/, 1 /*p*/, 0 /*b*/, thread_id);
        mix_and_shuffle(t, 28 /*a*/, 2 /*p*/, 0 /*b*/, thread_id);
        mix_and_shuffle(t, 28 /*a*/, 3 /*p*/, 0 /*b*/, thread_id);

        mix_and_shuffle(t, 28 /*a*/, 0 /*p*/, 1 /*b*/, thread_id);
        mix_and_shuffle(t, 28 /*a*/, 1 /*p*/, 1 /*b*/, thread_id);
        mix_and_shuffle(t, 28 /*a*/, 2 /*p*/, 1 /*b*/, thread_id);
        mix_and_shuffle(t, 28 /*a*/, 3 /*p*/, 1 /*b*/, thread_id);

        mix_and_shuffle(t, 28 /*a*/, 0 /*p*/, 2 /*b*/, thread_id);
        mix_and_shuffle(t, 28 /*a*/, 1 /*p*/, 2 /*b*/, thread_id);
        mix_and_shuffle(t, 28 /*a*/, 2 /*p*/, 2 /*b*/, thread_id);
        mix_and_shuffle(t, 28 /*a*/, 3 /*p*/, 2 /*b*/, thread_id);

        mix_and_shuffle(t, 28 /*a*/, 0 /*p*/, 3 /*b*/, thread_id);
        mix_and_shuffle(t, 28 /*a*/, 1 /*p*/, 3 /*b*/, thread_id);
        mix_and_shuffle(t, 28 /*a*/, 2 /*p*/, 3 /*b*/, thread_id);
        mix_and_shuffle(t, 28 /*a*/, 3 /*p*/, 3 /*b*/, thread_id);

        t = bfe(32, 2u, 3u);
        mix_and_shuffle(t, 32 /*a*/, 0 /*p*/, 0 /*b*/, thread_id);
        mix_and_shuffle(t, 32 /*a*/, 1 /*p*/, 0 /*b*/, thread_id);
        mix_and_shuffle(t, 32 /*a*/, 2 /*p*/, 0 /*b*/, thread_id);
        mix_and_shuffle(t, 32 /*a*/, 3 /*p*/, 0 /*b*/, thread_id);

        mix_and_shuffle(t, 32 /*a*/, 0 /*p*/, 1 /*b*/, thread_id);
        mix_and_shuffle(t, 32 /*a*/, 1 /*p*/, 1 /*b*/, thread_id);
        mix_and_shuffle(t, 32 /*a*/, 2 /*p*/, 1 /*b*/, thread_id);
        mix_and_shuffle(t, 32 /*a*/, 3 /*p*/, 1 /*b*/, thread_id);

        mix_and_shuffle(t, 32 /*a*/, 0 /*p*/, 2 /*b*/, thread_id);
        mix_and_shuffle(t, 32 /*a*/, 1 /*p*/, 2 /*b*/, thread_id);
        mix_and_shuffle(t, 32 /*a*/, 2 /*p*/, 2 /*b*/, thread_id);
        mix_and_shuffle(t, 32 /*a*/, 3 /*p*/, 2 /*b*/, thread_id);

        mix_and_shuffle(t, 32 /*a*/, 0 /*p*/, 3 /*b*/, thread_id);
        mix_and_shuffle(t, 32 /*a*/, 1 /*p*/, 3 /*b*/, thread_id);
        mix_and_shuffle(t, 32 /*a*/, 2 /*p*/, 3 /*b*/, thread_id);
        mix_and_shuffle(t, 32 /*a*/, 3 /*p*/, 3 /*b*/, thread_id);

        t = bfe(36, 2u, 3u);
        mix_and_shuffle(t, 36 /*a*/, 0 /*p*/, 0 /*b*/, thread_id);
        mix_and_shuffle(t, 36 /*a*/, 1 /*p*/, 0 /*b*/, thread_id);
        mix_and_shuffle(t, 36 /*a*/, 2 /*p*/, 0 /*b*/, thread_id);
        mix_and_shuffle(t, 36 /*a*/, 3 /*p*/, 0 /*b*/, thread_id);

        mix_and_shuffle(t, 36 /*a*/, 0 /*p*/, 1 /*b*/, thread_id);
        mix_and_shuffle(t, 36 /*a*/, 1 /*p*/, 1 /*b*/, thread_id);
        mix_and_shuffle(t, 36 /*a*/, 2 /*p*/, 1 /*b*/, thread_id);
        mix_and_shuffle(t, 36 /*a*/, 3 /*p*/, 1 /*b*/, thread_id);

        mix_and_shuffle(t, 36 /*a*/, 0 /*p*/, 2 /*b*/, thread_id);
        mix_and_shuffle(t, 36 /*a*/, 1 /*p*/, 2 /*b*/, thread_id);
        mix_and_shuffle(t, 36 /*a*/, 2 /*p*/, 2 /*b*/, thread_id);
        mix_and_shuffle(t, 36 /*a*/, 3 /*p*/, 2 /*b*/, thread_id);

        mix_and_shuffle(t, 36 /*a*/, 0 /*p*/, 3 /*b*/, thread_id);
        mix_and_shuffle(t, 36 /*a*/, 1 /*p*/, 3 /*b*/, thread_id);
        mix_and_shuffle(t, 36 /*a*/, 2 /*p*/, 3 /*b*/, thread_id);
        mix_and_shuffle(t, 36 /*a*/, 3 /*p*/, 3 /*b*/, thread_id);

        t = bfe(40, 2u, 3u);
        mix_and_shuffle(t, 40 /*a*/, 0 /*p*/, 0 /*b*/, thread_id);
        mix_and_shuffle(t, 40 /*a*/, 1 /*p*/, 0 /*b*/, thread_id);
        mix_and_shuffle(t, 40 /*a*/, 2 /*p*/, 0 /*b*/, thread_id);
        mix_and_shuffle(t, 40 /*a*/, 3 /*p*/, 0 /*b*/, thread_id);

        mix_and_shuffle(t, 40 /*a*/, 0 /*p*/, 1 /*b*/, thread_id);
        mix_and_shuffle(t, 40 /*a*/, 1 /*p*/, 1 /*b*/, thread_id);
        mix_and_shuffle(t, 40 /*a*/, 2 /*p*/, 1 /*b*/, thread_id);
        mix_and_shuffle(t, 40 /*a*/, 3 /*p*/, 1 /*b*/, thread_id);

        mix_and_shuffle(t, 40 /*a*/, 0 /*p*/, 2 /*b*/, thread_id);
        mix_and_shuffle(t, 40 /*a*/, 1 /*p*/, 2 /*b*/, thread_id);
        mix_and_shuffle(t, 40 /*a*/, 2 /*p*/, 2 /*b*/, thread_id);
        mix_and_shuffle(t, 40 /*a*/, 3 /*p*/, 2 /*b*/, thread_id);

        mix_and_shuffle(t, 40 /*a*/, 0 /*p*/, 3 /*b*/, thread_id);
        mix_and_shuffle(t, 40 /*a*/, 1 /*p*/, 3 /*b*/, thread_id);
        mix_and_shuffle(t, 40 /*a*/, 2 /*p*/, 3 /*b*/, thread_id);
        mix_and_shuffle(t, 40 /*a*/, 3 /*p*/, 3 /*b*/, thread_id);

        t = bfe(44, 2u, 3u);
        mix_and_shuffle(t, 44 /*a*/, 0 /*p*/, 0 /*b*/, thread_id);
        mix_and_shuffle(t, 44 /*a*/, 1 /*p*/, 0 /*b*/, thread_id);
        mix_and_shuffle(t, 44 /*a*/, 2 /*p*/, 0 /*b*/, thread_id);
        mix_and_shuffle(t, 44 /*a*/, 3 /*p*/, 0 /*b*/, thread_id);

        mix_and_shuffle(t, 44 /*a*/, 0 /*p*/, 1 /*b*/, thread_id);
        mix_and_shuffle(t, 44 /*a*/, 1 /*p*/, 1 /*b*/, thread_id);
        mix_and_shuffle(t, 44 /*a*/, 2 /*p*/, 1 /*b*/, thread_id);
        mix_and_shuffle(t, 44 /*a*/, 3 /*p*/, 1 /*b*/, thread_id);

        mix_and_shuffle(t, 44 /*a*/, 0 /*p*/, 2 /*b*/, thread_id);
        mix_and_shuffle(t, 44 /*a*/, 1 /*p*/, 2 /*b*/, thread_id);
        mix_and_shuffle(t, 44 /*a*/, 2 /*p*/, 2 /*b*/, thread_id);
        mix_and_shuffle(t, 44 /*a*/, 3 /*p*/, 2 /*b*/, thread_id);

        mix_and_shuffle(t, 44 /*a*/, 0 /*p*/, 3 /*b*/, thread_id);
        mix_and_shuffle(t, 44 /*a*/, 1 /*p*/, 3 /*b*/, thread_id);
        mix_and_shuffle(t, 44 /*a*/, 2 /*p*/, 3 /*b*/, thread_id);
        mix_and_shuffle(t, 44 /*a*/, 3 /*p*/, 3 /*b*/, thread_id);

        t = bfe(48, 2u, 3u);
        mix_and_shuffle(t, 48 /*a*/, 0 /*p*/, 0 /*b*/, thread_id);
        mix_and_shuffle(t, 48 /*a*/, 1 /*p*/, 0 /*b*/, thread_id);
        mix_and_shuffle(t, 48 /*a*/, 2 /*p*/, 0 /*b*/, thread_id);
        mix_and_shuffle(t, 48 /*a*/, 3 /*p*/, 0 /*b*/, thread_id);

        mix_and_shuffle(t, 48 /*a*/, 0 /*p*/, 1 /*b*/, thread_id);
        mix_and_shuffle(t, 48 /*a*/, 1 /*p*/, 1 /*b*/, thread_id);
        mix_and_shuffle(t, 48 /*a*/, 2 /*p*/, 1 /*b*/, thread_id);
        mix_and_shuffle(t, 48 /*a*/, 3 /*p*/, 1 /*b*/, thread_id);

        mix_and_shuffle(t, 48 /*a*/, 0 /*p*/, 2 /*b*/, thread_id);
        mix_and_shuffle(t, 48 /*a*/, 1 /*p*/, 2 /*b*/, thread_id);
        mix_and_shuffle(t, 48 /*a*/, 2 /*p*/, 2 /*b*/, thread_id);
        mix_and_shuffle(t, 48 /*a*/, 3 /*p*/, 2 /*b*/, thread_id);

        mix_and_shuffle(t, 48 /*a*/, 0 /*p*/, 3 /*b*/, thread_id);
        mix_and_shuffle(t, 48 /*a*/, 1 /*p*/, 3 /*b*/, thread_id);
        mix_and_shuffle(t, 48 /*a*/, 2 /*p*/, 3 /*b*/, thread_id);
        mix_and_shuffle(t, 48 /*a*/, 3 /*p*/, 3 /*b*/, thread_id);

        t = bfe(52, 2u, 3u);
        mix_and_shuffle(t, 52 /*a*/, 0 /*p*/, 0 /*b*/, thread_id);
        mix_and_shuffle(t, 52 /*a*/, 1 /*p*/, 0 /*b*/, thread_id);
        mix_and_shuffle(t, 52 /*a*/, 2 /*p*/, 0 /*b*/, thread_id);
        mix_and_shuffle(t, 52 /*a*/, 3 /*p*/, 0 /*b*/, thread_id);

        mix_and_shuffle(t, 52 /*a*/, 0 /*p*/, 1 /*b*/, thread_id);
        mix_and_shuffle(t, 52 /*a*/, 1 /*p*/, 1 /*b*/, thread_id);
        mix_and_shuffle(t, 52 /*a*/, 2 /*p*/, 1 /*b*/, thread_id);
        mix_and_shuffle(t, 52 /*a*/, 3 /*p*/, 1 /*b*/, thread_id);

        mix_and_shuffle(t, 52 /*a*/, 0 /*p*/, 2 /*b*/, thread_id);
        mix_and_shuffle(t, 52 /*a*/, 1 /*p*/, 2 /*b*/, thread_id);
        mix_and_shuffle(t, 52 /*a*/, 2 /*p*/, 2 /*b*/, thread_id);
        mix_and_shuffle(t, 52 /*a*/, 3 /*p*/, 2 /*b*/, thread_id);

        mix_and_shuffle(t, 52 /*a*/, 0 /*p*/, 3 /*b*/, thread_id);
        mix_and_shuffle(t, 52 /*a*/, 1 /*p*/, 3 /*b*/, thread_id);
        mix_and_shuffle(t, 52 /*a*/, 2 /*p*/, 3 /*b*/, thread_id);
        mix_and_shuffle(t, 52 /*a*/, 3 /*p*/, 3 /*b*/, thread_id);

        t = bfe(56, 2u, 3u);
        mix_and_shuffle(t, 56 /*a*/, 0 /*p*/, 0 /*b*/, thread_id);
        mix_and_shuffle(t, 56 /*a*/, 1 /*p*/, 0 /*b*/, thread_id);
        mix_and_shuffle(t, 56 /*a*/, 2 /*p*/, 0 /*b*/, thread_id);
        mix_and_shuffle(t, 56 /*a*/, 3 /*p*/, 0 /*b*/, thread_id);

        mix_and_shuffle(t, 56 /*a*/, 0 /*p*/, 1 /*b*/, thread_id);
        mix_and_shuffle(t, 56 /*a*/, 1 /*p*/, 1 /*b*/, thread_id);
        mix_and_shuffle(t, 56 /*a*/, 2 /*p*/, 1 /*b*/, thread_id);
        mix_and_shuffle(t, 56 /*a*/, 3 /*p*/, 1 /*b*/, thread_id);

        mix_and_shuffle(t, 56 /*a*/, 0 /*p*/, 2 /*b*/, thread_id);
        mix_and_shuffle(t, 56 /*a*/, 1 /*p*/, 2 /*b*/, thread_id);
        mix_and_shuffle(t, 56 /*a*/, 2 /*p*/, 2 /*b*/, thread_id);
        mix_and_shuffle(t, 56 /*a*/, 3 /*p*/, 2 /*b*/, thread_id);

        mix_and_shuffle(t, 56 /*a*/, 0 /*p*/, 3 /*b*/, thread_id);
        mix_and_shuffle(t, 56 /*a*/, 1 /*p*/, 3 /*b*/, thread_id);
        mix_and_shuffle(t, 56 /*a*/, 2 /*p*/, 3 /*b*/, thread_id);
        mix_and_shuffle(t, 56 /*a*/, 3 /*p*/, 3 /*b*/, thread_id);

        t = bfe(60, 2u, 3u);
        mix_and_shuffle(t, 60 /*a*/, 0 /*p*/, 0 /*b*/, thread_id);
        mix_and_shuffle(t, 60 /*a*/, 1 /*p*/, 0 /*b*/, thread_id);
        mix_and_shuffle(t, 60 /*a*/, 2 /*p*/, 0 /*b*/, thread_id);
        mix_and_shuffle(t, 60 /*a*/, 3 /*p*/, 0 /*b*/, thread_id);

        mix_and_shuffle(t, 60 /*a*/, 0 /*p*/, 1 /*b*/, thread_id);
        mix_and_shuffle(t, 60 /*a*/, 1 /*p*/, 1 /*b*/, thread_id);
        mix_and_shuffle(t, 60 /*a*/, 2 /*p*/, 1 /*b*/, thread_id);
        mix_and_shuffle(t, 60 /*a*/, 3 /*p*/, 1 /*b*/, thread_id);

        mix_and_shuffle(t, 60 /*a*/, 0 /*p*/, 2 /*b*/, thread_id);
        mix_and_shuffle(t, 60 /*a*/, 1 /*p*/, 2 /*b*/, thread_id);
        mix_and_shuffle(t, 60 /*a*/, 2 /*p*/, 2 /*b*/, thread_id);
        mix_and_shuffle(t, 60 /*a*/, 3 /*p*/, 2 /*b*/, thread_id);

        mix_and_shuffle(t, 60 /*a*/, 0 /*p*/, 3 /*b*/, thread_id);
        mix_and_shuffle(t, 60 /*a*/, 1 /*p*/, 3 /*b*/, thread_id);
        mix_and_shuffle(t, 60 /*a*/, 2 /*p*/, 3 /*b*/, thread_id);
        mix_and_shuffle(t, 60 /*a*/, 3 /*p*/, 3 /*b*/, thread_id);

        /////            }
        /////        }

        for (int p = 0; p < _PARALLEL_HASH; p++) {
            ////sycl::uint2 shuffle[4];
            sycl::uint2 shuffle_0, shuffle_1, shuffle_2, shuffle_3;
            uint32_t    thread_mix = fnv_reduce(mix[p]);

            // update mix across threads

            shuffle_0.x() = item_ct1.get_sub_group().shuffle(thread_mix, 0 + iShuffleOffset);
            shuffle_0.y() = item_ct1.get_sub_group().shuffle(thread_mix, 1 + iShuffleOffset);
            shuffle_1.x() = item_ct1.get_sub_group().shuffle(thread_mix, 2 + iShuffleOffset);
            shuffle_1.y() = item_ct1.get_sub_group().shuffle(thread_mix, 3 + iShuffleOffset);
            shuffle_2.x() = item_ct1.get_sub_group().shuffle(thread_mix, 4 + iShuffleOffset);
            shuffle_2.y() = item_ct1.get_sub_group().shuffle(thread_mix, 5 + iShuffleOffset);
            shuffle_3.x() = item_ct1.get_sub_group().shuffle(thread_mix, 6 + iShuffleOffset);
            shuffle_3.y() = item_ct1.get_sub_group().shuffle(thread_mix, 7 + iShuffleOffset);

            if ((i + p) == thread_id) {
                // move mix into state:
                state[8]  = shuffle_0;
                state[9]  = shuffle_1;
                state[10] = shuffle_2;
                state[11] = shuffle_3;
            }
        }
    }


#endif

    // keccak_256(keccak_512(header..nonce) .. mix);
    uint64_t x = keccak_f1600_final(state, keccak_round_constants);
    if (cuda_swab64(x) > d_target)
        return true;

    mix_hash[0] = state[8];
    mix_hash[1] = state[9];
    mix_hash[2] = state[10];
    mix_hash[3] = state[11];

    return false;
}
