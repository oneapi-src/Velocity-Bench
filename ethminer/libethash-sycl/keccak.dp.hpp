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
#include "dpcpp_helper.h"

DEV_INLINE sycl::uint2 xor5(const sycl::uint2 a, const sycl::uint2 b, const sycl::uint2 c, const sycl::uint2 d, const sycl::uint2 e)
{
    return a ^ b ^ c ^ d ^ e;
}

DEV_INLINE sycl::uint2 xor3(const sycl::uint2 a, const sycl::uint2 b, const sycl::uint2 c)
{
    return a ^ b ^ c;
}

DEV_INLINE sycl::uint2 chi(const sycl::uint2 a, const sycl::uint2 b, const sycl::uint2 c)
{
    return a ^ (~b) & c;
}

DEV_INLINE void keccak_f1600_init(
    sycl::uint2       *state,
    const sycl::uint2 *keccak_round_constants,
    hash32_t           d_header)
{
    sycl::uint2       s[25];
    sycl::uint2       t[5], u, v;
    const sycl::uint2 u2zero = sycl::uint2(0, 0);

    devectorize2(d_header.uint4s[0], s[0], s[1]);
    devectorize2(d_header.uint4s[1], s[2], s[3]);
    s[4] = state[4];
    s[5] = sycl::uint2(1, 0);
    s[6] = u2zero;
    s[7] = u2zero;
    s[8] = sycl::uint2(0, 0x80000000);
    for (uint32_t i = 9; i < 25; i++)
        s[i] = u2zero;

    /* theta: c = a[0,i] ^ a[1,i] ^ .. a[4,i] */
    t[0].x() = s[0].x() ^ s[5].x();
    t[0].y() = s[0].y();
    t[1]     = s[1];
    t[2]     = s[2];
    t[3].x() = s[3].x();
    t[3].y() = s[3].y() ^ s[8].y();
    t[4]     = s[4];

    /* theta: d[i] = c[i+4] ^ rotl(c[i+1],1) */
    /* theta: a[0,i], a[1,i], .. a[4,i] ^= d[i] */

    u = dpct_operator_overloading::operator^(t[4], ROL2(t[1], 1));
    s[0] ^= u;
    s[5] ^= u;
    s[10] ^= u;
    s[15] ^= u;
    s[20] ^= u;

    u = dpct_operator_overloading::operator^(t[0], ROL2(t[2], 1));
    s[1] ^= u;
    s[6] ^= u;
    s[11] ^= u;
    s[16] ^= u;
    s[21] ^= u;

    u = dpct_operator_overloading::operator^(t[1], ROL2(t[3], 1));
    s[2] ^= u;
    s[7] ^= u;
    s[12] ^= u;
    s[17] ^= u;
    s[22] ^= u;

    u = dpct_operator_overloading::operator^(t[2], ROL2(t[4], 1));
    s[3] ^= u;
    s[8] ^= u;
    s[13] ^= u;
    s[18] ^= u;
    s[23] ^= u;

    u = dpct_operator_overloading::operator^(t[3], ROL2(t[0], 1));
    s[4] ^= u;
    s[9] ^= u;
    s[14] ^= u;
    s[19] ^= u;
    s[24] ^= u;

    /* rho pi: b[..] = rotl(a[..], ..) */
    u = s[1];

    s[1]  = ROL2(s[6], 44);
    s[6]  = ROL2(s[9], 20);
    s[9]  = ROL2(s[22], 61);
    s[22] = ROL2(s[14], 39);
    s[14] = ROL2(s[20], 18);
    s[20] = ROL2(s[2], 62);
    s[2]  = ROL2(s[12], 43);
    s[12] = ROL2(s[13], 25);
    s[13] = ROL8(s[19]);
    s[19] = ROR8(s[23]);
    s[23] = ROL2(s[15], 41);
    s[15] = ROL2(s[4], 27);
    s[4]  = ROL2(s[24], 14);
    s[24] = ROL2(s[21], 2);
    s[21] = ROL2(s[8], 55);
    s[8]  = ROL2(s[16], 45);
    s[16] = ROL2(s[5], 36);
    s[5]  = ROL2(s[3], 28);
    s[3]  = ROL2(s[18], 21);
    s[18] = ROL2(s[17], 15);
    s[17] = ROL2(s[11], 10);
    s[11] = ROL2(s[7], 6);
    s[7]  = ROL2(s[10], 3);
    s[10] = ROL2(u, 1);

    /* chi: a[i,j] ^= ~b[i,j+1] & b[i,j+2] */

    u    = s[0];
    v    = s[1];
    s[0] = chi(s[0], s[1], s[2]);
    s[1] = chi(s[1], s[2], s[3]);
    s[2] = chi(s[2], s[3], s[4]);
    s[3] = chi(s[3], s[4], u);
    s[4] = chi(s[4], u, v);

    u    = s[5];
    v    = s[6];
    s[5] = chi(s[5], s[6], s[7]);
    s[6] = chi(s[6], s[7], s[8]);
    s[7] = chi(s[7], s[8], s[9]);
    s[8] = chi(s[8], s[9], u);
    s[9] = chi(s[9], u, v);

    u     = s[10];
    v     = s[11];
    s[10] = chi(s[10], s[11], s[12]);
    s[11] = chi(s[11], s[12], s[13]);
    s[12] = chi(s[12], s[13], s[14]);
    s[13] = chi(s[13], s[14], u);
    s[14] = chi(s[14], u, v);

    u     = s[15];
    v     = s[16];
    s[15] = chi(s[15], s[16], s[17]);
    s[16] = chi(s[16], s[17], s[18]);
    s[17] = chi(s[17], s[18], s[19]);
    s[18] = chi(s[18], s[19], u);
    s[19] = chi(s[19], u, v);

    u     = s[20];
    v     = s[21];
    s[20] = chi(s[20], s[21], s[22]);
    s[21] = chi(s[21], s[22], s[23]);
    s[22] = chi(s[22], s[23], s[24]);
    s[23] = chi(s[23], s[24], u);
    s[24] = chi(s[24], u, v);

    /* iota: a[0,0] ^= round constant */
    s[0] ^= keccak_round_constants[0];

    for (int i = 1; i < 23; i++) {
        /* theta: c = a[0,i] ^ a[1,i] ^ .. a[4,i] */
        t[0] = xor5(s[0], s[5], s[10], s[15], s[20]);
        t[1] = xor5(s[1], s[6], s[11], s[16], s[21]);
        t[2] = xor5(s[2], s[7], s[12], s[17], s[22]);
        t[3] = xor5(s[3], s[8], s[13], s[18], s[23]);
        t[4] = xor5(s[4], s[9], s[14], s[19], s[24]);

        /* theta: d[i] = c[i+4] ^ rotl(c[i+1],1) */
        /* theta: a[0,i], a[1,i], .. a[4,i] ^= d[i] */

        u = dpct_operator_overloading::operator^(t[4], ROL2(t[1], 1));
        s[0] ^= u;
        s[5] ^= u;
        s[10] ^= u;
        s[15] ^= u;
        s[20] ^= u;

        u = dpct_operator_overloading::operator^(t[0], ROL2(t[2], 1));
        s[1] ^= u;
        s[6] ^= u;
        s[11] ^= u;
        s[16] ^= u;
        s[21] ^= u;

        u = dpct_operator_overloading::operator^(t[1], ROL2(t[3], 1));
        s[2] ^= u;
        s[7] ^= u;
        s[12] ^= u;
        s[17] ^= u;
        s[22] ^= u;

        u = dpct_operator_overloading::operator^(t[2], ROL2(t[4], 1));
        s[3] ^= u;
        s[8] ^= u;
        s[13] ^= u;
        s[18] ^= u;
        s[23] ^= u;

        u = dpct_operator_overloading::operator^(t[3], ROL2(t[0], 1));
        s[4] ^= u;
        s[9] ^= u;
        s[14] ^= u;
        s[19] ^= u;
        s[24] ^= u;

        /* rho pi: b[..] = rotl(a[..], ..) */
        u = s[1];

        s[1]  = ROL2(s[6], 44);
        s[6]  = ROL2(s[9], 20);
        s[9]  = ROL2(s[22], 61);
        s[22] = ROL2(s[14], 39);
        s[14] = ROL2(s[20], 18);
        s[20] = ROL2(s[2], 62);
        s[2]  = ROL2(s[12], 43);
        s[12] = ROL2(s[13], 25);
        s[13] = ROL8(s[19]);
        s[19] = ROR8(s[23]);
        s[23] = ROL2(s[15], 41);
        s[15] = ROL2(s[4], 27);
        s[4]  = ROL2(s[24], 14);
        s[24] = ROL2(s[21], 2);
        s[21] = ROL2(s[8], 55);
        s[8]  = ROL2(s[16], 45);
        s[16] = ROL2(s[5], 36);
        s[5]  = ROL2(s[3], 28);
        s[3]  = ROL2(s[18], 21);
        s[18] = ROL2(s[17], 15);
        s[17] = ROL2(s[11], 10);
        s[11] = ROL2(s[7], 6);
        s[7]  = ROL2(s[10], 3);
        s[10] = ROL2(u, 1);

        /* chi: a[i,j] ^= ~b[i,j+1] & b[i,j+2] */

        u    = s[0];
        v    = s[1];
        s[0] = chi(s[0], s[1], s[2]);
        s[1] = chi(s[1], s[2], s[3]);
        s[2] = chi(s[2], s[3], s[4]);
        s[3] = chi(s[3], s[4], u);
        s[4] = chi(s[4], u, v);

        u    = s[5];
        v    = s[6];
        s[5] = chi(s[5], s[6], s[7]);
        s[6] = chi(s[6], s[7], s[8]);
        s[7] = chi(s[7], s[8], s[9]);
        s[8] = chi(s[8], s[9], u);
        s[9] = chi(s[9], u, v);

        u     = s[10];
        v     = s[11];
        s[10] = chi(s[10], s[11], s[12]);
        s[11] = chi(s[11], s[12], s[13]);
        s[12] = chi(s[12], s[13], s[14]);
        s[13] = chi(s[13], s[14], u);
        s[14] = chi(s[14], u, v);

        u     = s[15];
        v     = s[16];
        s[15] = chi(s[15], s[16], s[17]);
        s[16] = chi(s[16], s[17], s[18]);
        s[17] = chi(s[17], s[18], s[19]);
        s[18] = chi(s[18], s[19], u);
        s[19] = chi(s[19], u, v);

        u     = s[20];
        v     = s[21];
        s[20] = chi(s[20], s[21], s[22]);
        s[21] = chi(s[21], s[22], s[23]);
        s[22] = chi(s[22], s[23], s[24]);
        s[23] = chi(s[23], s[24], u);
        s[24] = chi(s[24], u, v);

        /* iota: a[0,0] ^= round constant */
        s[0] ^= keccak_round_constants[i];
    }

    /* theta: c = a[0,i] ^ a[1,i] ^ .. a[4,i] */
    t[0] = xor5(s[0], s[5], s[10], s[15], s[20]);
    t[1] = xor5(s[1], s[6], s[11], s[16], s[21]);
    t[2] = xor5(s[2], s[7], s[12], s[17], s[22]);
    t[3] = xor5(s[3], s[8], s[13], s[18], s[23]);
    t[4] = xor5(s[4], s[9], s[14], s[19], s[24]);

    /* theta: d[i] = c[i+4] ^ rotl(c[i+1],1) */
    /* theta: a[0,i], a[1,i], .. a[4,i] ^= d[i] */

    u = dpct_operator_overloading::operator^(t[4], ROL2(t[1], 1));
    s[0] ^= u;
    s[10] ^= u;

    u = dpct_operator_overloading::operator^(t[0], ROL2(t[2], 1));
    s[6] ^= u;
    s[16] ^= u;

    u = dpct_operator_overloading::operator^(t[1], ROL2(t[3], 1));
    s[12] ^= u;
    s[22] ^= u;

    u = dpct_operator_overloading::operator^(t[2], ROL2(t[4], 1));
    s[3] ^= u;
    s[18] ^= u;

    u = dpct_operator_overloading::operator^(t[3], ROL2(t[0], 1));
    s[9] ^= u;
    s[24] ^= u;

    /* rho pi: b[..] = rotl(a[..], ..) */
    u = s[1];

    s[1] = ROL2(s[6], 44);
    s[6] = ROL2(s[9], 20);
    s[9] = ROL2(s[22], 61);
    s[2] = ROL2(s[12], 43);
    s[4] = ROL2(s[24], 14);
    s[8] = ROL2(s[16], 45);
    s[5] = ROL2(s[3], 28);
    s[3] = ROL2(s[18], 21);
    s[7] = ROL2(s[10], 3);

    /* chi: a[i,j] ^= ~b[i,j+1] & b[i,j+2] */

    u    = s[0];
    v    = s[1];
    s[0] = chi(s[0], s[1], s[2]);
    s[1] = chi(s[1], s[2], s[3]);
    s[2] = chi(s[2], s[3], s[4]);
    s[3] = chi(s[3], s[4], u);
    s[4] = chi(s[4], u, v);
    s[5] = chi(s[5], s[6], s[7]);
    s[6] = chi(s[6], s[7], s[8]);
    s[7] = chi(s[7], s[8], s[9]);

    /* iota: a[0,0] ^= round constant */
    s[0] ^= keccak_round_constants[23];

    for (int i = 0; i < 12; ++i)
        state[i] = s[i];
}

DEV_INLINE uint64_t keccak_f1600_final(
    sycl::uint2       *state,
    const sycl::uint2 *keccak_round_constants)
{
    sycl::uint2       s[25];
    sycl::uint2       t[5], u, v;
    const sycl::uint2 u2zero = sycl::uint2(0, 0);

    for (int i = 0; i < 12; ++i)
        s[i] = state[i];

    s[12] = sycl::uint2(1, 0);
    s[13] = u2zero;
    s[14] = u2zero;
    s[15] = u2zero;
    s[16] = sycl::uint2(0, 0x80000000);
    for (uint32_t i = 17; i < 25; i++)
        s[i] = u2zero;

    /* theta: c = a[0,i] ^ a[1,i] ^ .. a[4,i] */
    t[0] = xor3(s[0], s[5], s[10]);
    t[1] = dpct_operator_overloading::operator^(xor3(s[1], s[6], s[11]), s[16]);
    t[2] = xor3(s[2], s[7], s[12]);
    t[3] = dpct_operator_overloading::operator^(s[3], s[8]);
    t[4] = dpct_operator_overloading::operator^(s[4], s[9]);

    /* theta: d[i] = c[i+4] ^ rotl(c[i+1],1) */
    /* theta: a[0,i], a[1,i], .. a[4,i] ^= d[i] */

    u = dpct_operator_overloading::operator^(t[4], ROL2(t[1], 1));
    s[0] ^= u;
    s[5] ^= u;
    s[10] ^= u;
    s[15] ^= u;
    s[20] ^= u;

    u = dpct_operator_overloading::operator^(t[0], ROL2(t[2], 1));
    s[1] ^= u;
    s[6] ^= u;
    s[11] ^= u;
    s[16] ^= u;
    s[21] ^= u;

    u = dpct_operator_overloading::operator^(t[1], ROL2(t[3], 1));
    s[2] ^= u;
    s[7] ^= u;
    s[12] ^= u;
    s[17] ^= u;
    s[22] ^= u;

    u = dpct_operator_overloading::operator^(t[2], ROL2(t[4], 1));
    s[3] ^= u;
    s[8] ^= u;
    s[13] ^= u;
    s[18] ^= u;
    s[23] ^= u;

    u = dpct_operator_overloading::operator^(t[3], ROL2(t[0], 1));
    s[4] ^= u;
    s[9] ^= u;
    s[14] ^= u;
    s[19] ^= u;
    s[24] ^= u;

    /* rho pi: b[..] = rotl(a[..], ..) */
    u = s[1];

    s[1]  = ROL2(s[6], 44);
    s[6]  = ROL2(s[9], 20);
    s[9]  = ROL2(s[22], 61);
    s[22] = ROL2(s[14], 39);
    s[14] = ROL2(s[20], 18);
    s[20] = ROL2(s[2], 62);
    s[2]  = ROL2(s[12], 43);
    s[12] = ROL2(s[13], 25);
    s[13] = ROL8(s[19]);
    s[19] = ROR8(s[23]);
    s[23] = ROL2(s[15], 41);
    s[15] = ROL2(s[4], 27);
    s[4]  = ROL2(s[24], 14);
    s[24] = ROL2(s[21], 2);
    s[21] = ROL2(s[8], 55);
    s[8]  = ROL2(s[16], 45);
    s[16] = ROL2(s[5], 36);
    s[5]  = ROL2(s[3], 28);
    s[3]  = ROL2(s[18], 21);
    s[18] = ROL2(s[17], 15);
    s[17] = ROL2(s[11], 10);
    s[11] = ROL2(s[7], 6);
    s[7]  = ROL2(s[10], 3);
    s[10] = ROL2(u, 1);

    /* chi: a[i,j] ^= ~b[i,j+1] & b[i,j+2] */
    u    = s[0];
    v    = s[1];
    s[0] = chi(s[0], s[1], s[2]);
    s[1] = chi(s[1], s[2], s[3]);
    s[2] = chi(s[2], s[3], s[4]);
    s[3] = chi(s[3], s[4], u);
    s[4] = chi(s[4], u, v);

    u    = s[5];
    v    = s[6];
    s[5] = chi(s[5], s[6], s[7]);
    s[6] = chi(s[6], s[7], s[8]);
    s[7] = chi(s[7], s[8], s[9]);
    s[8] = chi(s[8], s[9], u);
    s[9] = chi(s[9], u, v);

    u     = s[10];
    v     = s[11];
    s[10] = chi(s[10], s[11], s[12]);
    s[11] = chi(s[11], s[12], s[13]);
    s[12] = chi(s[12], s[13], s[14]);
    s[13] = chi(s[13], s[14], u);
    s[14] = chi(s[14], u, v);

    u     = s[15];
    v     = s[16];
    s[15] = chi(s[15], s[16], s[17]);
    s[16] = chi(s[16], s[17], s[18]);
    s[17] = chi(s[17], s[18], s[19]);
    s[18] = chi(s[18], s[19], u);
    s[19] = chi(s[19], u, v);

    u     = s[20];
    v     = s[21];
    s[20] = chi(s[20], s[21], s[22]);
    s[21] = chi(s[21], s[22], s[23]);
    s[22] = chi(s[22], s[23], s[24]);
    s[23] = chi(s[23], s[24], u);
    s[24] = chi(s[24], u, v);

    /* iota: a[0,0] ^= round constant */
    s[0] ^= keccak_round_constants[0];

    for (int i = 1; i < 23; i++) {
        /* theta: c = a[0,i] ^ a[1,i] ^ .. a[4,i] */
        t[0] = xor5(s[0], s[5], s[10], s[15], s[20]);
        t[1] = xor5(s[1], s[6], s[11], s[16], s[21]);
        t[2] = xor5(s[2], s[7], s[12], s[17], s[22]);
        t[3] = xor5(s[3], s[8], s[13], s[18], s[23]);
        t[4] = xor5(s[4], s[9], s[14], s[19], s[24]);

        /* theta: d[i] = c[i+4] ^ rotl(c[i+1],1) */
        /* theta: a[0,i], a[1,i], .. a[4,i] ^= d[i] */

        u = dpct_operator_overloading::operator^(t[4], ROL2(t[1], 1));
        s[0] ^= u;
        s[5] ^= u;
        s[10] ^= u;
        s[15] ^= u;
        s[20] ^= u;

        u = dpct_operator_overloading::operator^(t[0], ROL2(t[2], 1));
        s[1] ^= u;
        s[6] ^= u;
        s[11] ^= u;
        s[16] ^= u;
        s[21] ^= u;

        u = dpct_operator_overloading::operator^(t[1], ROL2(t[3], 1));
        s[2] ^= u;
        s[7] ^= u;
        s[12] ^= u;
        s[17] ^= u;
        s[22] ^= u;

        u = dpct_operator_overloading::operator^(t[2], ROL2(t[4], 1));
        s[3] ^= u;
        s[8] ^= u;
        s[13] ^= u;
        s[18] ^= u;
        s[23] ^= u;

        u = dpct_operator_overloading::operator^(t[3], ROL2(t[0], 1));
        s[4] ^= u;
        s[9] ^= u;
        s[14] ^= u;
        s[19] ^= u;
        s[24] ^= u;

        /* rho pi: b[..] = rotl(a[..], ..) */
        u = s[1];

        s[1]  = ROL2(s[6], 44);
        s[6]  = ROL2(s[9], 20);
        s[9]  = ROL2(s[22], 61);
        s[22] = ROL2(s[14], 39);
        s[14] = ROL2(s[20], 18);
        s[20] = ROL2(s[2], 62);
        s[2]  = ROL2(s[12], 43);
        s[12] = ROL2(s[13], 25);
        s[13] = ROL8(s[19]);
        s[19] = ROR8(s[23]);
        s[23] = ROL2(s[15], 41);
        s[15] = ROL2(s[4], 27);
        s[4]  = ROL2(s[24], 14);
        s[24] = ROL2(s[21], 2);
        s[21] = ROL2(s[8], 55);
        s[8]  = ROL2(s[16], 45);
        s[16] = ROL2(s[5], 36);
        s[5]  = ROL2(s[3], 28);
        s[3]  = ROL2(s[18], 21);
        s[18] = ROL2(s[17], 15);
        s[17] = ROL2(s[11], 10);
        s[11] = ROL2(s[7], 6);
        s[7]  = ROL2(s[10], 3);
        s[10] = ROL2(u, 1);

        /* chi: a[i,j] ^= ~b[i,j+1] & b[i,j+2] */
        u    = s[0];
        v    = s[1];
        s[0] = chi(s[0], s[1], s[2]);
        s[1] = chi(s[1], s[2], s[3]);
        s[2] = chi(s[2], s[3], s[4]);
        s[3] = chi(s[3], s[4], u);
        s[4] = chi(s[4], u, v);

        u    = s[5];
        v    = s[6];
        s[5] = chi(s[5], s[6], s[7]);
        s[6] = chi(s[6], s[7], s[8]);
        s[7] = chi(s[7], s[8], s[9]);
        s[8] = chi(s[8], s[9], u);
        s[9] = chi(s[9], u, v);

        u     = s[10];
        v     = s[11];
        s[10] = chi(s[10], s[11], s[12]);
        s[11] = chi(s[11], s[12], s[13]);
        s[12] = chi(s[12], s[13], s[14]);
        s[13] = chi(s[13], s[14], u);
        s[14] = chi(s[14], u, v);

        u     = s[15];
        v     = s[16];
        s[15] = chi(s[15], s[16], s[17]);
        s[16] = chi(s[16], s[17], s[18]);
        s[17] = chi(s[17], s[18], s[19]);
        s[18] = chi(s[18], s[19], u);
        s[19] = chi(s[19], u, v);

        u     = s[20];
        v     = s[21];
        s[20] = chi(s[20], s[21], s[22]);
        s[21] = chi(s[21], s[22], s[23]);
        s[22] = chi(s[22], s[23], s[24]);
        s[23] = chi(s[23], s[24], u);
        s[24] = chi(s[24], u, v);

        /* iota: a[0,0] ^= round constant */
        s[0] ^= keccak_round_constants[i];
    }

    t[0] = xor5(s[0], s[5], s[10], s[15], s[20]);
    t[1] = xor5(s[1], s[6], s[11], s[16], s[21]);
    t[2] = xor5(s[2], s[7], s[12], s[17], s[22]);
    t[3] = xor5(s[3], s[8], s[13], s[18], s[23]);
    t[4] = xor5(s[4], s[9], s[14], s[19], s[24]);

    s[0]  = xor3(s[0], t[4], ROL2(t[1], 1));
    s[6]  = xor3(s[6], t[0], ROL2(t[2], 1));
    s[12] = xor3(s[12], t[1], ROL2(t[3], 1));

    s[1] = ROL2(s[6], 44);
    s[2] = ROL2(s[12], 43);

    s[0] = chi(s[0], s[1], s[2]);

    /* iota: a[0,0] ^= round constant */
    // s[0] ^= vectorize(keccak_round_constants[23]);
    return devectorize(dpct_operator_overloading::operator^(s[0], keccak_round_constants[23]));
}

DEV_INLINE void SHA3_512(sycl::uint2 *s, const sycl::uint2 *keccak_round_constants)
{
    sycl::uint2 t[5], u, v;

    for (uint32_t i = 8; i < 25; i++) {
        s[i] = sycl::uint2(0, 0);
    }
    s[8].x() = 1;
    s[8].y() = 0x80000000;

    for (int i = 0; i < 23; i++) {
        /* theta: c = a[0,i] ^ a[1,i] ^ .. a[4,i] */
        t[0] = xor5(s[0], s[5], s[10], s[15], s[20]);
        t[1] = xor5(s[1], s[6], s[11], s[16], s[21]);
        t[2] = xor5(s[2], s[7], s[12], s[17], s[22]);
        t[3] = xor5(s[3], s[8], s[13], s[18], s[23]);
        t[4] = xor5(s[4], s[9], s[14], s[19], s[24]);

        /* theta: d[i] = c[i+4] ^ rotl(c[i+1],1) */
        /* theta: a[0,i], a[1,i], .. a[4,i] ^= d[i] */

        u = dpct_operator_overloading::operator^(t[4], ROL2(t[1], 1));
        s[0] ^= u;
        s[5] ^= u;
        s[10] ^= u;
        s[15] ^= u;
        s[20] ^= u;

        u = dpct_operator_overloading::operator^(t[0], ROL2(t[2], 1));
        s[1] ^= u;
        s[6] ^= u;
        s[11] ^= u;
        s[16] ^= u;
        s[21] ^= u;

        u = dpct_operator_overloading::operator^(t[1], ROL2(t[3], 1));
        s[2] ^= u;
        s[7] ^= u;
        s[12] ^= u;
        s[17] ^= u;
        s[22] ^= u;

        u = dpct_operator_overloading::operator^(t[2], ROL2(t[4], 1));
        s[3] ^= u;
        s[8] ^= u;
        s[13] ^= u;
        s[18] ^= u;
        s[23] ^= u;

        u = dpct_operator_overloading::operator^(t[3], ROL2(t[0], 1));
        s[4] ^= u;
        s[9] ^= u;
        s[14] ^= u;
        s[19] ^= u;
        s[24] ^= u;

        /* rho pi: b[..] = rotl(a[..], ..) */
        u = s[1];

        s[1]  = ROL2(s[6], 44);
        s[6]  = ROL2(s[9], 20);
        s[9]  = ROL2(s[22], 61);
        s[22] = ROL2(s[14], 39);
        s[14] = ROL2(s[20], 18);
        s[20] = ROL2(s[2], 62);
        s[2]  = ROL2(s[12], 43);
        s[12] = ROL2(s[13], 25);
        s[13] = ROL2(s[19], 8);
        s[19] = ROL2(s[23], 56);
        s[23] = ROL2(s[15], 41);
        s[15] = ROL2(s[4], 27);
        s[4]  = ROL2(s[24], 14);
        s[24] = ROL2(s[21], 2);
        s[21] = ROL2(s[8], 55);
        s[8]  = ROL2(s[16], 45);
        s[16] = ROL2(s[5], 36);
        s[5]  = ROL2(s[3], 28);
        s[3]  = ROL2(s[18], 21);
        s[18] = ROL2(s[17], 15);
        s[17] = ROL2(s[11], 10);
        s[11] = ROL2(s[7], 6);
        s[7]  = ROL2(s[10], 3);
        s[10] = ROL2(u, 1);

        /* chi: a[i,j] ^= ~b[i,j+1] & b[i,j+2] */
        u    = s[0];
        v    = s[1];
        s[0] = chi(s[0], s[1], s[2]);
        s[1] = chi(s[1], s[2], s[3]);
        s[2] = chi(s[2], s[3], s[4]);
        s[3] = chi(s[3], s[4], u);
        s[4] = chi(s[4], u, v);

        u    = s[5];
        v    = s[6];
        s[5] = chi(s[5], s[6], s[7]);
        s[6] = chi(s[6], s[7], s[8]);
        s[7] = chi(s[7], s[8], s[9]);
        s[8] = chi(s[8], s[9], u);
        s[9] = chi(s[9], u, v);

        u     = s[10];
        v     = s[11];
        s[10] = chi(s[10], s[11], s[12]);
        s[11] = chi(s[11], s[12], s[13]);
        s[12] = chi(s[12], s[13], s[14]);
        s[13] = chi(s[13], s[14], u);
        s[14] = chi(s[14], u, v);

        u     = s[15];
        v     = s[16];
        s[15] = chi(s[15], s[16], s[17]);
        s[16] = chi(s[16], s[17], s[18]);
        s[17] = chi(s[17], s[18], s[19]);
        s[18] = chi(s[18], s[19], u);
        s[19] = chi(s[19], u, v);

        u     = s[20];
        v     = s[21];
        s[20] = chi(s[20], s[21], s[22]);
        s[21] = chi(s[21], s[22], s[23]);
        s[22] = chi(s[22], s[23], s[24]);
        s[23] = chi(s[23], s[24], u);
        s[24] = chi(s[24], u, v);

        /* iota: a[0,0] ^= round constant */
        s[0] ^= LDG(keccak_round_constants[i]);
    }

    /* theta: c = a[0,i] ^ a[1,i] ^ .. a[4,i] */
    t[0] = xor5(s[0], s[5], s[10], s[15], s[20]);
    t[1] = xor5(s[1], s[6], s[11], s[16], s[21]);
    t[2] = xor5(s[2], s[7], s[12], s[17], s[22]);
    t[3] = xor5(s[3], s[8], s[13], s[18], s[23]);
    t[4] = xor5(s[4], s[9], s[14], s[19], s[24]);

    /* theta: d[i] = c[i+4] ^ rotl(c[i+1],1) */
    /* theta: a[0,i], a[1,i], .. a[4,i] ^= d[i] */

    u = dpct_operator_overloading::operator^(t[4], ROL2(t[1], 1));
    s[0] ^= u;
    s[10] ^= u;

    u = dpct_operator_overloading::operator^(t[0], ROL2(t[2], 1));
    s[6] ^= u;
    s[16] ^= u;

    u = dpct_operator_overloading::operator^(t[1], ROL2(t[3], 1));
    s[12] ^= u;
    s[22] ^= u;

    u = dpct_operator_overloading::operator^(t[2], ROL2(t[4], 1));
    s[3] ^= u;
    s[18] ^= u;

    u = dpct_operator_overloading::operator^(t[3], ROL2(t[0], 1));
    s[9] ^= u;
    s[24] ^= u;

    /* rho pi: b[..] = rotl(a[..], ..) */
    u = s[1];

    s[1] = ROL2(s[6], 44);
    s[6] = ROL2(s[9], 20);
    s[9] = ROL2(s[22], 61);
    s[2] = ROL2(s[12], 43);
    s[4] = ROL2(s[24], 14);
    s[8] = ROL2(s[16], 45);
    s[5] = ROL2(s[3], 28);
    s[3] = ROL2(s[18], 21);
    s[7] = ROL2(s[10], 3);

    /* chi: a[i,j] ^= ~b[i,j+1] & b[i,j+2] */

    u    = s[0];
    v    = s[1];
    s[0] = chi(s[0], s[1], s[2]);
    s[1] = chi(s[1], s[2], s[3]);
    s[2] = chi(s[2], s[3], s[4]);
    s[3] = chi(s[3], s[4], u);
    s[4] = chi(s[4], u, v);
    s[5] = chi(s[5], s[6], s[7]);
    s[6] = chi(s[6], s[7], s[8]);
    s[7] = chi(s[7], s[8], s[9]);

    /* iota: a[0,0] ^= round constant */
    s[0] ^= LDG(keccak_round_constants[23]);
}

//---------------------------------------------------------------------

static sycl::uint2 const Keccak_f1600_RC[24] = {
    {0x00000001, 0x00000000},
    {0x00008082, 0x00000000},
    {0x0000808a, 0x80000000},
    {0x80008000, 0x80000000},
    {0x0000808b, 0x00000000},
    {0x80000001, 0x00000000},
    {0x80008081, 0x80000000},
    {0x00008009, 0x80000000},
    {0x0000008a, 0x00000000},
    {0x00000088, 0x00000000},
    {0x80008009, 0x00000000},
    {0x8000000a, 0x00000000},
    {0x8000808b, 0x00000000},
    {0x0000008b, 0x80000000},
    {0x00008089, 0x80000000},
    {0x00008003, 0x80000000},
    {0x00008002, 0x80000000},
    {0x00000080, 0x80000000},
    {0x0000800a, 0x00000000},
    {0x8000000a, 0x80000000},
    {0x80008081, 0x80000000},
    {0x00008080, 0x80000000},
    {0x80000001, 0x00000000},
    {0x80008008, 0x80000000},
};
DEV_INLINE uint64_t as_ulong(sycl::uint2 x)
{
    using res_vec_type = sycl::vec<sycl::opencl::cl_ulong, 1>;
    res_vec_type y     = x.as<res_vec_type>();
    return y[0];
}
#define ROTL64_1(x, y) (sycl::rotate(as_ulong(x), (uint64_t)(y)))
#define ROTL64_2(x, y) ROTL64_1(x, (y) + 32)

#define KECCAKF_1600_RND(a, i, outsz)                                                                                  \
    do {                                                                                                               \
        const sycl::uint2 m0 = a[0] ^ a[5] ^ a[10] ^ a[15] ^ a[20] ^ ROTL64_1(a[2] ^ a[7] ^ a[12] ^ a[17] ^ a[22], 1); \
        const sycl::uint2 m1 = a[1] ^ a[6] ^ a[11] ^ a[16] ^ a[21] ^ ROTL64_1(a[3] ^ a[8] ^ a[13] ^ a[18] ^ a[23], 1); \
        const sycl::uint2 m2 = a[2] ^ a[7] ^ a[12] ^ a[17] ^ a[22] ^ ROTL64_1(a[4] ^ a[9] ^ a[14] ^ a[19] ^ a[24], 1); \
        const sycl::uint2 m3 = a[3] ^ a[8] ^ a[13] ^ a[18] ^ a[23] ^ ROTL64_1(a[0] ^ a[5] ^ a[10] ^ a[15] ^ a[20], 1); \
        const sycl::uint2 m4 = a[4] ^ a[9] ^ a[14] ^ a[19] ^ a[24] ^ ROTL64_1(a[1] ^ a[6] ^ a[11] ^ a[16] ^ a[21], 1); \
                                                                                                                       \
        const sycl::uint2 tmp = a[1] ^ m0;                                                                             \
                                                                                                                       \
        a[0] ^= m4;                                                                                                    \
        a[5] ^= m4;                                                                                                    \
        a[10] ^= m4;                                                                                                   \
        a[15] ^= m4;                                                                                                   \
        a[20] ^= m4;                                                                                                   \
                                                                                                                       \
        a[6] ^= m0;                                                                                                    \
        a[11] ^= m0;                                                                                                   \
        a[16] ^= m0;                                                                                                   \
        a[21] ^= m0;                                                                                                   \
                                                                                                                       \
        a[2] ^= m1;                                                                                                    \
        a[7] ^= m1;                                                                                                    \
        a[12] ^= m1;                                                                                                   \
        a[17] ^= m1;                                                                                                   \
        a[22] ^= m1;                                                                                                   \
                                                                                                                       \
        a[3] ^= m2;                                                                                                    \
        a[8] ^= m2;                                                                                                    \
        a[13] ^= m2;                                                                                                   \
        a[18] ^= m2;                                                                                                   \
        a[23] ^= m2;                                                                                                   \
                                                                                                                       \
        a[4] ^= m3;                                                                                                    \
        a[9] ^= m3;                                                                                                    \
        a[14] ^= m3;                                                                                                   \
        a[19] ^= m3;                                                                                                   \
        a[24] ^= m3;                                                                                                   \
                                                                                                                       \
        a[1]  = ROTL64_2(a[6], 12);                                                                                    \
        a[6]  = ROTL64_1(a[9], 20);                                                                                    \
        a[9]  = ROTL64_2(a[22], 29);                                                                                   \
        a[22] = ROTL64_2(a[14], 7);                                                                                    \
        a[14] = ROTL64_1(a[20], 18);                                                                                   \
        a[20] = ROTL64_2(a[2], 30);                                                                                    \
        a[2]  = ROTL64_2(a[12], 11);                                                                                   \
        a[12] = ROTL64_1(a[13], 25);                                                                                   \
        a[13] = ROTL64_1(a[19], 8);                                                                                    \
        a[19] = ROTL64_2(a[23], 24);                                                                                   \
        a[23] = ROTL64_2(a[15], 9);                                                                                    \
        a[15] = ROTL64_1(a[4], 27);                                                                                    \
        a[4]  = ROTL64_1(a[24], 14);                                                                                   \
        a[24] = ROTL64_1(a[21], 2);                                                                                    \
        a[21] = ROTL64_2(a[8], 23);                                                                                    \
        a[8]  = ROTL64_2(a[16], 13);                                                                                   \
        a[16] = ROTL64_2(a[5], 4);                                                                                     \
        a[5]  = ROTL64_1(a[3], 28);                                                                                    \
        a[3]  = ROTL64_1(a[18], 21);                                                                                   \
        a[18] = ROTL64_1(a[17], 15);                                                                                   \
        a[17] = ROTL64_1(a[11], 10);                                                                                   \
        a[11] = ROTL64_1(a[7], 6);                                                                                     \
        a[7]  = ROTL64_1(a[10], 3);                                                                                    \
        a[10] = ROTL64_1(tmp, 1);                                                                                      \
                                                                                                                       \
        sycl::uint2 m5 = a[0];                                                                                         \
        sycl::uint2 m6 = a[1];                                                                                         \
        a[0]           = bitselect(a[0] ^ a[2], a[0], a[1]);                                                           \
        a[0] ^= sycl::uint2(keccak_round_constants[i]);                                                                \
        if (outsz > 1) {                                                                                               \
            a[1] = bitselect(a[1] ^ a[3], a[1], a[2]);                                                                 \
            a[2] = bitselect(a[2] ^ a[4], a[2], a[3]);                                                                 \
            a[3] = bitselect(a[3] ^ m5, a[3], a[4]);                                                                   \
            a[4] = bitselect(a[4] ^ m6, a[4], m5);                                                                     \
            if (outsz > 4) {                                                                                           \
                m5   = a[5];                                                                                           \
                m6   = a[6];                                                                                           \
                a[5] = bitselect(a[5] ^ a[7], a[5], a[6]);                                                             \
                a[6] = bitselect(a[6] ^ a[8], a[6], a[7]);                                                             \
                a[7] = bitselect(a[7] ^ a[9], a[7], a[8]);                                                             \
                a[8] = bitselect(a[8] ^ m5, a[8], a[9]);                                                               \
                a[9] = bitselect(a[9] ^ m6, a[9], m5);                                                                 \
                if (outsz > 8) {                                                                                       \
                    m5    = a[10];                                                                                     \
                    m6    = a[11];                                                                                     \
                    a[10] = bitselect(a[10] ^ a[12], a[10], a[11]);                                                    \
                    a[11] = bitselect(a[11] ^ a[13], a[11], a[12]);                                                    \
                    a[12] = bitselect(a[12] ^ a[14], a[12], a[13]);                                                    \
                    a[13] = bitselect(a[13] ^ m5, a[13], a[14]);                                                       \
                    a[14] = bitselect(a[14] ^ m6, a[14], m5);                                                          \
                    m5    = a[15];                                                                                     \
                    m6    = a[16];                                                                                     \
                    a[15] = bitselect(a[15] ^ a[17], a[15], a[16]);                                                    \
                    a[16] = bitselect(a[16] ^ a[18], a[16], a[17]);                                                    \
                    a[17] = bitselect(a[17] ^ a[19], a[17], a[18]);                                                    \
                    a[18] = bitselect(a[18] ^ m5, a[18], a[19]);                                                       \
                    a[19] = bitselect(a[19] ^ m6, a[19], m5);                                                          \
                    m5    = a[20];                                                                                     \
                    m6    = a[21];                                                                                     \
                    a[20] = bitselect(a[20] ^ a[22], a[20], a[21]);                                                    \
                    a[21] = bitselect(a[21] ^ a[23], a[21], a[22]);                                                    \
                    a[22] = bitselect(a[22] ^ a[24], a[22], a[23]);                                                    \
                    a[23] = bitselect(a[23] ^ m5, a[23], a[24]);                                                       \
                    a[24] = bitselect(a[24] ^ m6, a[24], m5);                                                          \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
    } while (0);

#define KECCAK_PROCESS(st, in_size, out_size)    \
    do {                                         \
        for (int r = 0; r < 24; ++r) {           \
            int os = (r < 23 ? 25 : (out_size)); \
            KECCAKF_1600_RND(st, r, os);         \
        }                                        \
    } while (0);

DEV_INLINE void SHA3_512_2(sycl::uint2 *s, const sycl::uint2 *keccak_round_constants)
{
    sycl::uint2 st[25];

    for (uint i = 0; i < 8; ++i)
        st[i] = s[i];

    st[8] = {0x00000001, 0x80000000};

    for (uint i = 9; i != 25; ++i)
        st[i] = (sycl::uint2)(0);

    KECCAK_PROCESS(st, 8, 8);

    for (uint i = 0; i < 8; ++i)
        s[i] = st[i];
}
