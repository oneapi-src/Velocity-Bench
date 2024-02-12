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
#include <vector>
#pragma once

std::vector<sycl::uint2> const vKeccakConstants({{0x00000001, 0x00000000}, {0x00008082, 0x00000000}, {0x0000808a, 0x80000000}, {0x80008000, 0x80000000}, {0x0000808b, 0x00000000}, {0x80000001, 0x00000000}, {0x80008081, 0x80000000}, {0x00008009, 0x80000000}, {0x0000008a, 0x00000000}, {0x00000088, 0x00000000}, {0x80008009, 0x00000000}, {0x8000000a, 0x00000000}, {0x8000808b, 0x00000000}, {0x0000008b, 0x80000000}, {0x00008089, 0x80000000}, {0x00008003, 0x80000000}, {0x00008002, 0x80000000}, {0x00000080, 0x80000000}, {0x0000800a, 0x00000000}, {0x8000000a, 0x80000000}, {0x80008081, 0x80000000}, {0x00008080, 0x80000000}, {0x80000001, 0x00000000}, {0x80008008, 0x80000000}}); // 24 Elements

std::vector<int> const vShuffleOffsets({
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    8,
    16,
    16,
    16,
    16,
    16,
    16,
    16,
    16,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
#ifdef USE_AMD_BACKEND
    32,
    32,
    32,
    32,
    32,
    32,
    32,
    32,
    40,
    40,
    40,
    40,
    40,
    40,
    40,
    40,
    48,
    48,
    48,
    48,
    48,
    48,
    48,
    48,
    56,
    56,
    56,
    56,
    56,
    56,
    56,
    56,
#endif
});

#if (DPCT_COMPATIBILITY_TEMP >= 320)
/*
DPCT1026:15: The call to __ldg was removed, because there is no correspoinding API in DPC++.
*/
#define LDG(x) (x)
#else
#define LDG(x) (x)
#endif
