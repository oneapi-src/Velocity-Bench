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

#ifndef MEMORY_MANAGER_MEMORY_ALLOCATOR_H
#define MEMORY_MANAGER_MEMORY_ALLOCATOR_H

#include <string>

/**
 * @generalnote
 * In malloc and mm__malloc they use as byte_size (size_t )
 * which equals unsigned long long void * malloc( size_t size );
 */

/**
 * @brief Allocates aligned memory and returns an aligned pointer with the requested
 * size.
 *
 * @param size_of_type
 * The size in bytes of a single object that this pointer should point to.
 * Normally given by sizeof(type).
 *
 * @param number_of_elements
 * The number of elements that our pointer should contain.
 *
 * @param name
 * A user given name for the pointer for tracking purposes.
 *
 * @return
 * A void aligned pointer with the given size allocated.
 */
void *mem_allocate(unsigned long long size_of_type,
                   unsigned long long number_of_elements, const std::string &name);

/**
 * @brief Allocates aligned memory and returns an aligned pointer with the requested
 * size.
 *
 * @param size_of_type
 * The size in bytes of a single object that this pointer should point to.
 * Normally given by sizeof(type).
 *
 * @param number_of_elements
 * The number of elements that our pointer should contain.
 *
 * @param name
 * A user given name for the pointer for tracking purposes.
 *
 * @param half_length_padding
 * A padding to be added to the alignment.
 *
 * @return
 * A void aligned pointer with the given size allocated.
 */
void *mem_allocate(unsigned long long size_of_type,
                   unsigned long long number_of_elements, const std::string &name,
                   uint half_length_padding);

/**
 * @brief Allocates aligned memory and returns an aligned pointer with the requested
 * size.
 *
 * @param size_of_type
 * The size in bytes of a single object that this pointer should point to.
 * Normally given by sizeof(type).
 *
 * @param number_of_elements
 * The number of elements that our pointer should contain.
 *
 * @param name
 * A user given name for the pointer for tracking purposes.
 *
 * @param half_length_padding
 * A padding to be added to the alignment.
 *
 * @param masking_allocation_factor
 * An offset to differentiate and make sure of different pointer caching.
 *
 * @return
 * A void aligned pointer with the given size allocated.
 */
void *mem_allocate(unsigned long long size_of_type,
                   unsigned long long number_of_elements, const std::string &name,
                   uint half_length_padding, uint masking_allocation_factor);

/**
 * @brief Frees an aligned memory block.
 *
 * @param ptr
 * The aligned void pointer to be freed.
 */
void mem_free(void *ptr);

#endif //MEMORY_MANAGER_MEMORY_ALLOCATOR_H
