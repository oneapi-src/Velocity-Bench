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
#define FNV_PRIME 0x01000193

#define fnv(x, y) ((x)*FNV_PRIME ^ (y))

DEV_INLINE sycl::uint4 fnv4_p(bool allow, const sycl::stream &out, sycl::uint4 a, sycl::uint4 b)
{
    sycl::uint4 c;
    if (allow) {
        out << "fnv4 " << a << " " << b << "\n";
    }

    c.x() = a.x() * FNV_PRIME ^ b.x();
    c.y() = a.y() * FNV_PRIME ^ b.y();
    c.z() = a.z() * FNV_PRIME ^ b.z();
    c.w() = a.w() * FNV_PRIME ^ b.w();
    return c;
}

DEV_INLINE sycl::uint4 fnv4(sycl::uint4 a, sycl::uint4 b)
{
    sycl::uint4 c;
    c = a * FNV_PRIME ^ b;
    /*
        c.x() = a.x() * FNV_PRIME ^ b.x();
        c.y() = a.y() * FNV_PRIME ^ b.y();
        c.z() = a.z() * FNV_PRIME ^ b.z();
        c.w() = a.w() * FNV_PRIME ^ b.w();
    */
    return c;
}

DEV_INLINE uint32_t fnv_reduce(sycl::uint4 v)
{
    return fnv(fnv(fnv(v.x(), v.y()), v.z()), v.w());
}
