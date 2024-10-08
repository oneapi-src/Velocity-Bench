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

#define __dpct_inline__ __inline__ __attribute__((always_inline))
#define DEV_INLINE      __dpct_inline__

#ifdef __INTELLISENSE__
/* reduce vstudio warnings (__byteperm, blockIdx...) */
#include <device_functions.h>
#include <device_launch_parameters.h>
#define __launch_bounds__(max_tpb, min_blocks)
#define asm("a"            \
            : "=l"(result) \
            : "l"(a))
#define __SYCL_ARCH__ 520 // highlight shuffle code by default.

uint32_t __byte_perm_intel(uint32_t x, uint32_t y, uint32_t z);
uint32_t atomicExch(uint32_t *x, uint32_t y);
uint32_t atomicAdd(uint32_t *x, uint32_t y);
void     __syncthreads(void);
void     __threadfence(void);
void     __threadfence_block(void);
#endif

#include <stdint.h>

#ifndef MAX_GPUS
#define MAX_GPUS 32
#endif

extern "C" int      device_map[MAX_GPUS];
extern "C" long     device_sm[MAX_GPUS];
extern sycl::queue *gpustream[MAX_GPUS];

// common functions
extern void     cuda_check_cpu_init(int thr_id, uint32_t threads);
extern void     cuda_check_cpu_setTarget(const void *ptarget);
extern void     cuda_check_cpu_setTarget_mod(const void *ptarget, const void *ptarget2);
extern uint32_t cuda_check_hash(
    int       thr_id,
    uint32_t  threads,
    uint32_t  startNounce,
    uint32_t *d_inputHash);
extern uint32_t cuda_check_hash_suppl(
    int       thr_id,
    uint32_t  threads,
    uint32_t  startNounce,
    uint32_t *d_inputHash,
    uint32_t  foundnonce);
extern void cudaReportHardwareFailure(int thr_id, int error, const char *func);

namespace LocalDPCT
{
template <sycl::access::address_space addressSpace = sycl::access::address_space::global_space, 
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed, 
          sycl::memory_scope memoryScope = sycl::memory_scope::device>
inline unsigned int atomic_fetch_compare_inc(unsigned int *addr, unsigned int operand) {
    auto atm = sycl::atomic_ref<unsigned int, memoryOrder, memoryScope, addressSpace>(addr[0]);
    unsigned int old;
    while (true) {
        old = atm.load();
        if (old >= operand) {
            if (atm.compare_exchange_strong(old, 0))
                break;
        } else if (atm.compare_exchange_strong(old, old + 1))
            break;
    }
    return old;
}
} // namespace LocalDPCT

const int debug = 1;

bool is_print(uint32_t node_index)
{
    // return (node_index==0) || (node_index==1);
    return (node_index == 444);
}

#define ROTL32c(x, n) ((x) << (n)) | ((x) >> (32 - (n)))

#define ROTL32(x, n) ((x) << (n)) | ((x) >> (32 - (n)))

#define ROTR32(x, n) (((x) >> (n)) | ((x) << (32 - (n))))

#define cuda_swab32(x)                                                                       \
    ((((x) << 24) & 0xff000000u) | (((x) << 8) & 0x00ff0000u) | (((x) >> 8) & 0x0000ff00u) | \
     (((x) >> 24) & 0x000000ffu))

#define cuda_swab64(x) \
    ((uint64_t)((((uint64_t)(x)&0xff00000000000000ULL) >> 56) | (((uint64_t)(x)&0x00ff000000000000ULL) >> 40) | (((uint64_t)(x)&0x0000ff0000000000ULL) >> 24) | (((uint64_t)(x)&0x000000ff00000000ULL) >> 8) | (((uint64_t)(x)&0x00000000ff000000ULL) << 8) | (((uint64_t)(x)&0x0000000000ff0000ULL) << 24) | (((uint64_t)(x)&0x000000000000ff00ULL) << 40) | (((uint64_t)(x)&0x00000000000000ffULL) << 56)))

DEV_INLINE uint64_t devectorize(sycl::uint2 x)
{
    uint64_t result;
    result = ((uint64_t)x.x()) | (((uint64_t)x.y()) << 32);
    return result;
}

DEV_INLINE sycl::uint2 vectorize(const uint64_t x)
{
    sycl::uint2 result;
    result.x() = x & 0xffffffff;
    result.y() = x >> 32;
    return result;
}
DEV_INLINE void devectorize2(sycl::uint4 inn, sycl::uint2 &x, sycl::uint2 &y)
{
    x.x() = inn.x();
    x.y() = inn.y();
    y.x() = inn.z();
    y.y() = inn.w();
}

DEV_INLINE sycl::uint4 vectorize2(sycl::uint2 x, sycl::uint2 y)
{
    sycl::uint4 result;
    result.x() = x.x();
    result.y() = x.y();
    result.z() = y.x();
    result.w() = y.y();

    return result;
}

DEV_INLINE sycl::uint4 vectorize2(sycl::uint2 x)
{
    sycl::uint4 result;
    result.x() = x.x();
    result.y() = x.y();
    result.z() = x.x();
    result.w() = x.y();
    return result;
}

static DEV_INLINE sycl::uint2 vectorizelow(uint32_t v)
{
    sycl::uint2 result;
    result.x() = v;
    result.y() = 0;
    return result;
}
static DEV_INLINE sycl::uint2 vectorizehigh(uint32_t v)
{
    sycl::uint2 result;
    result.x() = 0;
    result.y() = v;
    return result;
}

/*
DPCT1011:0: The tool detected overloaded operators for built-in vector types, which may conflict
with the SYCL 1.2.1 standard operators (see 4.10.2.1 Vec interface). The tool inserted a namespace
to avoid the conflict. Use SYCL 1.2.1 standard operators instead.
*/
namespace dpct_operator_overloading
{

static DEV_INLINE sycl::uint2 operator^(sycl::uint2 a, uint32_t b)
{
    return sycl::uint2(a.x() ^ b, a.y());
}

} // namespace dpct_operator_overloading

/*
DPCT1011:1: The tool detected overloaded operators for built-in vector types, which may conflict
with the SYCL 1.2.1 standard operators (see 4.10.2.1 Vec interface). The tool inserted a namespace
to avoid the conflict. Use SYCL 1.2.1 standard operators instead.
*/
namespace dpct_operator_overloading
{

static DEV_INLINE sycl::uint2 operator^(sycl::uint2 a, sycl::uint2 b)
{
    return sycl::uint2(a.x() ^ b.x(), a.y() ^ b.y());
}

} // namespace dpct_operator_overloading

/*
DPCT1011:2: The tool detected overloaded operators for built-in vector types, which may conflict
with the SYCL 1.2.1 standard operators (see 4.10.2.1 Vec interface). The tool inserted a namespace
to avoid the conflict. Use SYCL 1.2.1 standard operators instead.
*/
namespace dpct_operator_overloading
{

static DEV_INLINE sycl::uint2 operator&(sycl::uint2 a, sycl::uint2 b)
{
    return sycl::uint2(a.x() & b.x(), a.y() & b.y());
}

} // namespace dpct_operator_overloading

/*
DPCT1011:3: The tool detected overloaded operators for built-in vector types, which may conflict
with the SYCL 1.2.1 standard operators (see 4.10.2.1 Vec interface). The tool inserted a namespace
to avoid the conflict. Use SYCL 1.2.1 standard operators instead.
*/
namespace dpct_operator_overloading
{

static DEV_INLINE sycl::uint2 operator|(sycl::uint2 a, sycl::uint2 b)
{
    return sycl::uint2(a.x() | b.x(), a.y() | b.y());
}

} // namespace dpct_operator_overloading

/*
DPCT1011:4: The tool detected overloaded operators for built-in vector types, which may conflict
with the SYCL 1.2.1 standard operators (see 4.10.2.1 Vec interface). The tool inserted a namespace
to avoid the conflict. Use SYCL 1.2.1 standard operators instead.
*/
namespace dpct_operator_overloading
{

static DEV_INLINE sycl::uint2 operator~(sycl::uint2 a)
{
    return sycl::uint2(~a.x(), ~a.y());
}

} // namespace dpct_operator_overloading

/*
DPCT1011:5: The tool detected overloaded operators for built-in vector types, which may conflict
with the SYCL 1.2.1 standard operators (see 4.10.2.1 Vec interface). The tool inserted a namespace
to avoid the conflict. Use SYCL 1.2.1 standard operators instead.
*/
namespace dpct_operator_overloading
{

static DEV_INLINE void operator^=(sycl::uint2 &a, sycl::uint2 b)
{
    a = dpct_operator_overloading::operator^(a, b);
}

} // namespace dpct_operator_overloading

/*
DPCT1011:6: The tool detected overloaded operators for built-in vector types, which may conflict
with the SYCL 1.2.1 standard operators (see 4.10.2.1 Vec interface). The tool inserted a namespace
to avoid the conflict. Use SYCL 1.2.1 standard operators instead.
*/
namespace dpct_operator_overloading
{

static DEV_INLINE sycl::uint4 operator^(sycl::uint4 a, sycl::uint4 b)
{
    return sycl::uint4(a.x() ^ b.x(), a.y() ^ b.y(), a.z() ^ b.z(), a.w() ^ b.w());
}

} // namespace dpct_operator_overloading

/*
DPCT1011:7: The tool detected overloaded operators for built-in vector types, which may conflict
with the SYCL 1.2.1 standard operators (see 4.10.2.1 Vec interface). The tool inserted a namespace
to avoid the conflict. Use SYCL 1.2.1 standard operators instead.
*/
namespace dpct_operator_overloading
{

static DEV_INLINE sycl::uint4 operator&(sycl::uint4 a, sycl::uint4 b)
{
    return sycl::uint4(a.x() & b.x(), a.y() & b.y(), a.z() & b.z(), a.w() & b.w());
}

} // namespace dpct_operator_overloading

/*
DPCT1011:8: The tool detected overloaded operators for built-in vector types, which may conflict
with the SYCL 1.2.1 standard operators (see 4.10.2.1 Vec interface). The tool inserted a namespace
to avoid the conflict. Use SYCL 1.2.1 standard operators instead.
*/
namespace dpct_operator_overloading
{

static DEV_INLINE sycl::uint4 operator|(sycl::uint4 a, sycl::uint4 b)
{
    return sycl::uint4(a.x() | b.x(), a.y() | b.y(), a.z() | b.z(), a.w() | b.w());
}

} // namespace dpct_operator_overloading

/*
DPCT1011:9: The tool detected overloaded operators for built-in vector types, which may conflict
with the SYCL 1.2.1 standard operators (see 4.10.2.1 Vec interface). The tool inserted a namespace
to avoid the conflict. Use SYCL 1.2.1 standard operators instead.
*/
namespace dpct_operator_overloading
{

static DEV_INLINE sycl::uint4 operator~(sycl::uint4 a)
{
    return sycl::uint4(~a.x(), ~a.y(), ~a.z(), ~a.w());
}

} // namespace dpct_operator_overloading

/*
DPCT1011:10: The tool detected overloaded operators for built-in vector types, which may conflict
with the SYCL 1.2.1 standard operators (see 4.10.2.1 Vec interface). The tool inserted a namespace
to avoid the conflict. Use SYCL 1.2.1 standard operators instead.
*/
namespace dpct_operator_overloading
{

static DEV_INLINE void operator^=(sycl::uint4 &a, sycl::uint4 b)
{
    a = dpct_operator_overloading::operator^(a, b);
}

} // namespace dpct_operator_overloading

/*
DPCT1011:11: The tool detected overloaded operators for built-in vector types, which may conflict
with the SYCL 1.2.1 standard operators (see 4.10.2.1 Vec interface). The tool inserted a namespace
to avoid the conflict. Use SYCL 1.2.1 standard operators instead.
*/
namespace dpct_operator_overloading
{

static DEV_INLINE sycl::uint4 operator^(sycl::uint4 a, sycl::uint2 b)
{
    return sycl::uint4(a.x() ^ b.x(), a.y() ^ b.y(), a.z() ^ b.x(), a.w() ^ b.y());
}

} // namespace dpct_operator_overloading

DEV_INLINE sycl::uint2 ROR2(const sycl::uint2 v, const int n)
{
    sycl::uint2 result;
    if (n <= 32) {
        result.y() = ((v.y() >> (n)) | (v.x() << (32 - n)));
        result.x() = ((v.x() >> (n)) | (v.y() << (32 - n)));
    } else {
        result.y() = ((v.x() >> (n - 32)) | (v.y() << (64 - n)));
        result.x() = ((v.y() >> (n - 32)) | (v.x() << (64 - n)));
    }
    return result;
}

DEV_INLINE uint32_t __byte_perm_intel(uint32_t x, uint32_t y, uint32_t s)
{
    uint32_t ret  = 0;
    uint64_t bigy = y;
    uint64_t bigx = x;
    uint64_t big  = bigy << 32 | bigx;
    uint8_t *sp   = (uint8_t *)&big;
    uint8_t *dp   = (uint8_t *)&ret;

    int a1 = (s & 0x7);
    int a2 = (s & 0x70) >> 4;
    int a3 = (s & 0x700) >> 8;
    int a4 = (s & 0x7000) >> 12;

    dp[0] = sp[a1];
    dp[1] = sp[a2];
    dp[2] = sp[a3];
    dp[3] = sp[a4];

    return ret;
}

DEV_INLINE uint32_t ROL8(const uint32_t x)
{
    return (x >> 24) + (x << 8);
}

DEV_INLINE uint32_t ROL8_new(const uint32_t x)
{
    return (x >> 24) + (x << 8);
}

DEV_INLINE uint32_t ROL16(const uint32_t x)
{
    return __byte_perm_intel(x, x, 0x1032);
}

DEV_INLINE uint32_t ROL16_new(const uint32_t x)
{
    return (x >> 16) + (x << 16);
}

DEV_INLINE uint32_t ROL24(const uint32_t x)
{
    return __byte_perm_intel(x, x, 0x0321);
}

DEV_INLINE uint32_t ROL24_new(const uint32_t x)
{
    return (x >> 8) + (x << 24);
}

DEV_INLINE sycl::uint2 ROR8(const sycl::uint2 a)
{
    sycl::uint2 result;
    result.x() = (a.y() << 24) + (a.x() >> 8);
    result.y() = (a.x() << 24) + (a.y() >> 8);

    return result;
}

DEV_INLINE sycl::uint2 ROR8_new(const sycl::uint2 a)
{
    sycl::uint2 result;
    result.x() = (a.y() << 24) + (a.x() >> 8);
    result.y() = (a.x() << 24) + (a.y() >> 8);

    return result;
}

DEV_INLINE sycl::uint2 ROR16(const sycl::uint2 a)
{
    sycl::uint2 result;
    result.x() = __byte_perm_intel(a.y(), a.x(), 0x1076);
    result.y() = __byte_perm_intel(a.y(), a.x(), 0x5432);

    return result;
}

DEV_INLINE sycl::uint2 ROR16_new(const sycl::uint2 a)
{
    sycl::uint2 result;
    result.x() = (a.y() << 16) + (a.x() >> 16);
    result.y() = (a.x() << 16) + (a.y() >> 16);

    return result;
}

DEV_INLINE sycl::uint2 ROR24(const sycl::uint2 a)
{
    sycl::uint2 result;
    result.x() = __byte_perm_intel(a.y(), a.x(), 0x2107);
    result.y() = __byte_perm_intel(a.y(), a.x(), 0x6543);

    return result;
}

DEV_INLINE sycl::uint2 ROR24_new(const sycl::uint2 a)
{
    sycl::uint2 result;
    result.x() = (a.y() << 8) + (a.x() >> 24);
    result.y() = (a.x() << 8) + (a.y() >> 24);

    return result;
}

DEV_INLINE sycl::uint2 ROL8(const sycl::uint2 a)
{
    sycl::uint2 result;
    result.x() = (a.y() >> 24) + (a.x() << 8);
    result.y() = (a.x() >> 24) + (a.y() << 8);

    return result;
}

DEV_INLINE sycl::uint2 ROL8_new(const sycl::uint2 a)
{
    sycl::uint2 result;
    result.x() = (a.y() >> 24) + (a.x() << 8);
    result.y() = (a.x() >> 24) + (a.y() << 8);

    return result;
}

DEV_INLINE sycl::uint2 ROL16(const sycl::uint2 a)
{
    sycl::uint2 result;
    result.x() = __byte_perm_intel(a.y(), a.x(), 0x5432);
    result.y() = __byte_perm_intel(a.y(), a.x(), 0x1076);

    return result;
}

DEV_INLINE sycl::uint2 ROL16_new(const sycl::uint2 a)
{
    sycl::uint2 result;
    result.x() = (a.y() >> 16) + (a.x() << 16);
    result.y() = (a.x() >> 16) + (a.y() << 16);

    return result;
}

DEV_INLINE sycl::uint2 ROL24(const sycl::uint2 a)
{
    sycl::uint2 result;
    result.x() = __byte_perm_intel(a.y(), a.x(), 0x4321);
    result.y() = __byte_perm_intel(a.y(), a.x(), 0x0765);

    return result;
}

DEV_INLINE sycl::uint2 ROL24_new(const sycl::uint2 a)
{
    sycl::uint2 result;
    result.x() = (a.y() >> 8) + (a.x() << 24);
    result.y() = (a.x() >> 8) + (a.y() << 24);

    return result;
}

__inline__ sycl::uint2 ROL2(const sycl::uint2 v, const int n)
{
    sycl::uint2 result;
    if (n <= 32) {
        result.y() = ((v.y() << (n)) | (v.x() >> (32 - n)));
        result.x() = ((v.x() << (n)) | (v.y() >> (32 - n)));
    } else {
        result.y() = ((v.x() << (n - 32)) | (v.y() >> (64 - n)));
        result.x() = ((v.y() << (n - 32)) | (v.x() >> (64 - n)));
    }
    return result;
}

DEV_INLINE uint64_t ROTR16(uint64_t x)
{

    return (((x) >> (16)) | ((x) << (64 - (16))));
}
DEV_INLINE uint64_t ROTL16(uint64_t x)
{

    return (((x) << (16)) | ((x) >> (64 - (16))));
}

static __dpct_inline__ sycl::uint2 SHL2(sycl::uint2 a, int offset)
{

    if (offset <= 32) {
        a.y() = (a.y() << offset) | (a.x() >> (32 - offset));
        a.x() = (a.x() << offset);
    } else {
        a.y() = (a.x() << (offset - 32));
        a.x() = 0;
    }
    return a;
}
static __dpct_inline__ sycl::uint2 SHR2(sycl::uint2 a, int offset)
{

    if (offset <= 32) {
        a.x() = (a.x() >> offset) | (a.y() << (32 - offset));
        a.y() = (a.y() >> offset);
    } else {
        a.x() = (a.y() >> (offset - 32));
        a.y() = 0;
    }
    return a;
}

DEV_INLINE uint32_t bfe(uint32_t x, uint32_t bit, uint32_t numBits)
{
    uint32_t ret;
    // asm("bfe.u32 %0, %1, %2, %3;" : "=r"(ret) : "r"(x), "r"(bit), "r"(numBits));
    // d, a, b, x
    int msb  = 31;
    int pos  = bit & 0xff;     // pos restricted to 0..255 range
    int len  = numBits & 0xff; // len restricted to 0..255 range
    int sbit = 0;

    ret = 0;
    for (int i = 0; i <= msb; i++) {
        // res[i] = (i<len && pos+i<=msb) ? x[pos+i] : sbit;
        ret |= (((i < len && pos + i <= msb) ? (x & (1 << (pos + i))) >> (pos + i) : sbit) << i);
    }
    return ret;
}
