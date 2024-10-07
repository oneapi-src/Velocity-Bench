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

#pragma once

#include <sycl/sycl.hpp>
#include <stdint.h>
#include <sstream>
#include <stdexcept>
#include <string>

// It is virtually impossible to get more than
// one solution per stream hash calculation
// Leave room for up to 4 results. A power
// of 2 here will yield better SYCL optimization
#define MAX_SEARCH_RESULTS 4U

struct Search_Result {
    // One word for gid and 8 for mix hash
    uint32_t gid;
    uint32_t mix[8];
    uint32_t pad[7]; // pad to size power of 2
};

struct Search_results {
    Search_Result result[MAX_SEARCH_RESULTS];
    uint32_t      count = 0;
};

#define ACCESSES         64
#define THREADS_PER_HASH (128 / 16)

typedef struct dpct_type_7de91c {
    sycl::uint4 uint4s[32 / sizeof(sycl::uint4)];
} hash32_t;

typedef union dpct_type_114c50 {
    uint32_t    words[128 / sizeof(uint32_t)];
    sycl::uint2 uint2s[128 / sizeof(sycl::uint2)];
    sycl::uint4 uint4s[128 / sizeof(sycl::uint4)];
} hash128_t;

typedef union dpct_type_9fad47 {
    uint32_t    words[64 / sizeof(uint32_t)];
    sycl::uint2 uint2s[64 / sizeof(sycl::uint2)];
    sycl::uint4 uint4s[64 / sizeof(sycl::uint4)];
} hash64_t;

struct cuda_runtime_error : public virtual std::runtime_error {
    cuda_runtime_error(const std::string &msg) : std::runtime_error(msg) {}
};

void print_dev(uint8_t *x_gpu, size_t n, char *s);

void dev_print(uint8_t *p, int size, char *name);
