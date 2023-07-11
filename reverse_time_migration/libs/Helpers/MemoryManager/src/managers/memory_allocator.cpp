/*
 * Modifications Copyright (C) 2023 Intel Corporation
 *
 * This Program is subject to the terms of the GNU Lesser General Public License v3.0 or later
 * 
 * If a copy of the license was not distributed with this file, you can obtain one at 
 * https://www.gnu.org/licenses/lgpl-3.0-standalone.html
 * 
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */


//
// Created by amr-nasr on 20/10/2019.
//

#include <memory-manager/managers/memory_allocator.h>

#include <memory-manager/managers/memory_tracker.h>

#include <cstdlib>
#include <unordered_map>

#define MASK_ALLOC_OFFSET(x) (x)
#define CACHELINE_BYTES 64

using namespace std;


/**
 * @brief Define unordered_map called base_pointer it is key wil be pointer to void
 * and its values will also be pointer to void and is called base_pointers
 */
static unordered_map<void *, void *> base_pointers;

void *mem_allocate(const unsigned long long size_of_type,
                   const unsigned long long number_of_elements, const string &name) {
    return mem_allocate(size_of_type, number_of_elements, name, 0);
}

void *mem_allocate(const unsigned long long size_of_type,
                   const unsigned long long number_of_elements, const string &name,
                   uint half_length_padding) {
    return mem_allocate(size_of_type, number_of_elements, name, 0, 0);
}

void *mem_allocate(const unsigned long long size_of_type,
                   const unsigned long long number_of_elements, const string &name,
                   uint half_length_padding, uint masking_allocation_factor) {
#ifndef __INTEL_COMPILER
    /*!if the intel compiler is not defined
     * this function is used to ensure the alignment of float variables
     * assume vector length =4 then 16 bytes then 16 is for alignment
     * MASK_ALLOC_OFFSET:for each array to be in different cache line
     * so now ptr_base is aligned and start alignment at the half_length_padding
     * example:
     * assume size_of_type is size_of_float = 4 bytes *16 = 64 (cache_line size)
     * and MASK_ALLOC_OFFSET(0)=0 and number of elements =6 so now we have for
     * each array number of floats reserved equals (6+16) =22 floats which equals
     * 1 cache line of size 64(16float) and extra 6 floats
     */
    void *ptr_base =
        malloc(size_of_type * (number_of_elements + 16 +
                               MASK_ALLOC_OFFSET(masking_allocation_factor)));
#else
    /*!if the intel compiler is  defined
     * this function is used to ensure the alignment of float variables
     * assume vector length =4 then 16 bytes then 16 is for alignment
     * MASK_ALLOC_OFFSET:for each array to be in different cache line
     * so now ptr_base is aligned and start alignment at the half_length_padding
     * note:for _mm_malloc it needs the cache_line number of bytes to be able to
     * do the  alignment
     */
    void *ptr_base = _mm_malloc(
            (number_of_elements + 16 + MASK_ALLOC_OFFSET(masking_allocation_factor)) *
            size_of_type,
            CACHELINE_BYTES);
#endif
    if (ptr_base == nullptr) {
        return nullptr;
    }
    /*!this function is for memory tracking
     * if the memory tracker is enabled it will work and add overhead
     * if not,it will be converted to empty function so the compiler will optimize
     * it it will keep track of the pointer ptr_base with the the variable called
     * name (which is passed as attribute for the function) name.c_str():change
     * string to array of characters and returns a  pointer to character  to an
     * array that contains a null-terminated sequence of characters representing
     * the current value of the basic_string object
     */
    name_ptr(ptr_base, (char *) name.c_str());

    /*!for the inner domain to be aligned without half_length we subtract the
     * half_length_padding so ptr has an offset to make the alignment match the
     * computational domain start so now ptr is aligned and start alignment at the
     * inner domain example: assume half_length_padding=2 size_of_type is
     * size_of_float = 4 bytes and MASK_ALLOC_OFFSET(0)=0 so &(((char
     * *)ptr_base):points to the same address that ptr_base points to but as
     * pointer to character (point to 1 byte) so with half_length_padding=2 it
     * points to [14*size_of_float] so the ptr points to the 14 element of the
     * reserved locations of pointer ptr_base and now element 14 and 15 points to
     * the 2 half_length_padding and they are th end of the first cache line and
     * the start of the inner domain will be at the start of the next cache line
     * so now ptr is aligned and start alignment at the inner domain
     */
    void *ptr =
            &(((char *) ptr_base)[(16 - half_length_padding +
                                   MASK_ALLOC_OFFSET(masking_allocation_factor)) *
                                  size_of_type]);

    /*!for the unordered map (base_pointers) the key is ptr which is aligned and
     * starts alignment at the inner domain and the value is ptr_base which is
     * aligned and start alignment at the half_length_padding
     */
    base_pointers[ptr] = ptr_base;

    // return the ptr: aligned pointer that start alignment at the inner domain
    // which is the key of the global unordered map base_pointers
    return ptr;
}

void mem_free(void *ptr) {
    if (ptr == nullptr) {
        return;
    }

    // get the value of key ptr which is a pointer that points to the same address
    // that ptr_base points to
    // and make org_ptr point to the same address so now ptr_base and org_ptr
    // points to the same address
    void *org_ptr = base_pointers[ptr];

#ifndef __INTEL_COMPILER
    // if the intel compiler is not defined free the org_ptr
    free(org_ptr);

#else
    // if the intel compiler is defined _mm_free the org_ptr
    _mm_free(org_ptr);
#endif
}
