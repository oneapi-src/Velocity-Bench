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

#include <hip/hip_runtime.h>

#include <hip/hip_runtime.h>

#define DEV_INLINE __device__ __forceinline__

#ifdef __INTELLISENSE__
/* reduce vstudio warnings (__byteperm, blockIdx...) */
////#include <hip/device_functions.h>
////#include <device_launch_parameters.h>
////#define __launch_bounds__(max_tpb, min_blocks)
////#define asm("a" : "=l"(result) : "l"(a))
////#define __CUDA_ARCH__ 520  // highlight shuffle code by default.

////uint32_t __byte_perm(uint32_t x, uint32_t y, uint32_t z);
////uint32_t __shfl(uint32_t x, uint32_t y, uint32_t z);
////uint32_t atomicExch(uint32_t* x, uint32_t y);
////uint32_t atomicAdd(uint32_t* x, uint32_t y);
////void __syncthreads(void);
////void __threadfence(void);
////void __threadfence_block(void);
#endif

#include <stdint.h>

#ifndef MAX_GPUS
#define MAX_GPUS 32
#endif

extern "C" int     device_map[MAX_GPUS];
extern "C" long    device_sm[MAX_GPUS];
extern hipStream_t gpustream[MAX_GPUS];

// common functions
extern void     cuda_check_cpu_init(int thr_id, uint32_t threads);
extern void     cuda_check_cpu_setTarget(const void *ptarget);
extern void     cuda_check_cpu_setTarget_mod(const void *ptarget, const void *ptarget2);
extern uint32_t cuda_check_hash(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_inputHash);
extern uint32_t cuda_check_hash_suppl(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_inputHash, uint32_t foundnonce);
extern void     cudaReportHardwareFailure(int thr_id, hipError_t error, const char *func);

/////#ifndef __CUDA_ARCH__
/////// define blockDim and threadIdx for host
/////extern const dim3 blockDim;
/////extern const uint3 threadIdx;
/////#endif

#ifndef SPH_C32
#define SPH_C32(x) ((x##U))
// #define SPH_C32(x) ((uint32_t)(x ## U))
#endif

#ifndef SPH_C64
#define SPH_C64(x) ((x##ULL))
// #define SPH_C64(x) ((uint64_t)(x ## ULL))
#endif

#ifndef SPH_T32
#define SPH_T32(x) (x)
// #define SPH_T32(x) ((x) & SPH_C32(0xFFFFFFFF))
#endif
#ifndef SPH_T64
#define SPH_T64(x) (x)
// #define SPH_T64(x) ((x) & SPH_C64(0xFFFFFFFFFFFFFFFF))
#endif

#define ROTL32c(x, n) ((x) << (n)) | ((x) >> (32 - (n)))

#define ROTL32(x, n) ((x) << (n)) | ((x) >> (32 - (n)))

#define ROTR32(x, n) (((x) >> (n)) | ((x) << (32 - (n))))

#define cuda_swab32(x)                                                                       \
    ((((x) << 24) & 0xff000000u) | (((x) << 8) & 0x00ff0000u) | (((x) >> 8) & 0x0000ff00u) | \
     (((x) >> 24) & 0x000000ffu))

#define cuda_swab64(x) \
    ((uint64_t)((((uint64_t)(x)&0xff00000000000000ULL) >> 56) | (((uint64_t)(x)&0x00ff000000000000ULL) >> 40) | (((uint64_t)(x)&0x0000ff0000000000ULL) >> 24) | (((uint64_t)(x)&0x000000ff00000000ULL) >> 8) | (((uint64_t)(x)&0x00000000ff000000ULL) << 8) | (((uint64_t)(x)&0x0000000000ff0000ULL) << 24) | (((uint64_t)(x)&0x000000000000ff00ULL) << 40) | (((uint64_t)(x)&0x00000000000000ffULL) << 56)))

DEV_INLINE uint64_t devectorize(uint2 x)
{
    uint64_t result;
    result = ((uint64_t)x.x) | (((uint64_t)x.y) << 32);
    return result;
}

DEV_INLINE uint2 vectorize(const uint64_t x)
{
    uint2 result;
    result.x = x & 0xffffffff;
    result.y = x >> 32;
    return result;
}

DEV_INLINE void devectorize2(uint4 inn, uint2 &x, uint2 &y)
{
    x.x = inn.x;
    x.y = inn.y;
    y.x = inn.z;
    y.y = inn.w;
}

DEV_INLINE uint4 vectorize2(uint2 x, uint2 y)
{
    uint4 result;
    result.x = x.x;
    result.y = x.y;
    result.z = y.x;
    result.w = y.y;

    return result;
}

DEV_INLINE uint4 vectorize2(uint2 x)
{
    uint4 result;
    result.x = x.x;
    result.y = x.y;
    result.z = x.x;
    result.w = x.y;
    return result;
}

static DEV_INLINE uint2 vectorizelow(uint32_t v)
{
    uint2 result;
    result.x = v;
    result.y = 0;
    return result;
}
static DEV_INLINE uint2 vectorizehigh(uint32_t v)
{
    uint2 result;
    result.x = 0;
    result.y = v;
    return result;
}

static DEV_INLINE uint2 operator^(uint2 a, uint32_t b)
{
    return make_uint2(a.x ^ b, a.y);
}
static DEV_INLINE uint2 operator^(uint2 a, uint2 b)
{
    return make_uint2(a.x ^ b.x, a.y ^ b.y);
}
static DEV_INLINE uint2 operator&(uint2 a, uint2 b)
{
    return make_uint2(a.x & b.x, a.y & b.y);
}
static DEV_INLINE uint2 operator|(uint2 a, uint2 b)
{
    return make_uint2(a.x | b.x, a.y | b.y);
}
static DEV_INLINE uint2 operator~(uint2 a)
{
    return make_uint2(~a.x, ~a.y);
}
static DEV_INLINE void operator^=(uint2 &a, uint2 b)
{
    a = a ^ b;
}

static DEV_INLINE uint4 operator^(uint4 a, uint4 b)
{
    return make_uint4(a.x ^ b.x, a.y ^ b.y, a.z ^ b.z, a.w ^ b.w);
}
static DEV_INLINE uint4 operator&(uint4 a, uint4 b)
{
    return make_uint4(a.x & b.x, a.y & b.y, a.z & b.z, a.w & b.w);
}
static DEV_INLINE uint4 operator|(uint4 a, uint4 b)
{
    return make_uint4(a.x | b.x, a.y | b.y, a.z | b.z, a.w | b.w);
}
static DEV_INLINE uint4 operator~(uint4 a)
{
    return make_uint4(~a.x, ~a.y, ~a.z, ~a.w);
}
static DEV_INLINE void operator^=(uint4 &a, uint4 b)
{
    a = a ^ b;
}
static DEV_INLINE uint4 operator^(uint4 a, uint2 b)
{
    return make_uint4(a.x ^ b.x, a.y ^ b.y, a.z ^ b.x, a.w ^ b.y);
}

DEV_INLINE uint2 ROR2(const uint2 v, const int n)
{
    uint2 result;
    if (n <= 32) {
        result.y = ((v.y >> (n)) | (v.x << (32 - n)));
        result.x = ((v.x >> (n)) | (v.y << (32 - n)));
    } else {
        result.y = ((v.x >> (n - 32)) | (v.y << (64 - n)));
        result.x = ((v.y >> (n - 32)) | (v.x << (64 - n)));
    }
    return result;
}

DEV_INLINE uint32_t ROL8(const uint32_t x)
{
    return (x >> 24) + (x << 8);
}
DEV_INLINE uint32_t ROL16(const uint32_t x)
{
    return (x >> 16) + (x << 16);
}
DEV_INLINE uint32_t ROL24(const uint32_t x)
{
    return (x >> 8) + (x << 24);
}

DEV_INLINE uint2 ROR8(const uint2 a)
{
    uint2 result;
    result.x = (a.y << 24) + (a.x >> 8);
    result.y = (a.x << 24) + (a.y >> 8);

    return result;
}

DEV_INLINE uint2 ROR16(const uint2 a)
{
    uint2 result;

    result.x = (a.y << 16) + (a.x >> 16);
    result.y = (a.x << 16) + (a.y >> 16);

    return result;
}

DEV_INLINE uint2 ROR24(const uint2 a)
{
    uint2 result;
    result.x = (a.y << 8) + (a.x >> 24);
    result.y = (a.x << 8) + (a.y >> 24);

    return result;
}

DEV_INLINE uint2 ROL8(const uint2 a)
{
    uint2 result;
    result.x = (a.y >> 24) + (a.x << 8);
    result.y = (a.x >> 24) + (a.y << 8);

    return result;
}

DEV_INLINE uint2 ROL16(const uint2 a)
{
    uint2 result;
    result.x = (a.y >> 16) + (a.x << 16);
    result.y = (a.x >> 16) + (a.y << 16);

    return result;
}

DEV_INLINE uint2 ROL24(const uint2 a)
{
    uint2 result;
    result.x = (a.y >> 8) + (a.x << 24);
    result.y = (a.x >> 8) + (a.y << 24);

    return result;
}

__inline__ __device__ uint2 ROL2(const uint2 v, const int n)
{
    uint2 result;
    if (n <= 32) {
        result.y = ((v.y << (n)) | (v.x >> (32 - n)));
        result.x = ((v.x << (n)) | (v.y >> (32 - n)));
    } else {
        result.y = ((v.x << (n - 32)) | (v.y >> (64 - n)));
        result.x = ((v.y << (n - 32)) | (v.x >> (64 - n)));
    }
    return result;
}

static __forceinline__ __device__ uint2 SHL2(uint2 a, int offset)
{
    if (offset <= 32) {
        a.y = (a.y << offset) | (a.x >> (32 - offset));
        a.x = (a.x << offset);
    } else {
        a.y = (a.x << (offset - 32));
        a.x = 0;
    }
    return a;
}
static __forceinline__ __device__ uint2 SHR2(uint2 a, int offset)
{
    if (offset <= 32) {
        a.x = (a.x >> offset) | (a.y << (32 - offset));
        a.y = (a.y >> offset);
    } else {
        a.x = (a.y >> (offset - 32));
        a.y = 0;
    }
    return a;
}

DEV_INLINE uint32_t bfe(uint32_t x, uint32_t bit, uint32_t numBits)
{
    uint32_t ret;
    int      msb  = 31;
    int      pos  = bit & 0xff;     // pos restricted to 0..255 range
    int      len  = numBits & 0xff; // len restricted to 0..255 range
    int      sbit = 0;

    ret = 0;
    for (int i = 0; i <= msb; i++) {
        ret |= (((i < len && pos + i <= msb) ? (x & (1 << (pos + i))) >> (pos + i) : sbit) << i);
    }

    return ret;
}
