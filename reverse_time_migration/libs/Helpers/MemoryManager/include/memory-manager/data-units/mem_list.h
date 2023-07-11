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


#ifndef MEMORY_MANAGER_LIST_MEM_H
#define MEMORY_MANAGER_LIST_MEM_H

#define STRING_LEN 300
#define STACK_SKIP 3

#ifndef MEM_STACK_SIZE
#define MEM_STACK_SIZE 10
#endif

#include <stdlib.h>

typedef enum ALLOCATION_TYPE {
    MALLOC = 0,
    NEW = 1,
    _MM_MALLOC = 2,
    A_REALLOC = 3,
    NEW_ARR = 4,
    CALLOC = 5
} allocation_type;
typedef enum DEALLOCATION_TYPE {
    FREE = 0,
    DELETE = 1,
    _MM_FREE = 2,
    D_REALLOC = 3,
    DELETE_ARR = 4
} deallocation_type;

int matching_calls(allocation_type alc, deallocation_type dealc);

void call_type_to_string(char *str, int type, int allocation);

/**
 * Memory Black data.
 **/
typedef struct MEM_ELEM {
    void *addr;
    // Name of the block if assigned by user.
    char name[STRING_LEN];
// Details about where it was allocated.
#if (MEM_TRACK >= 1)
    char malloc_file_name[STRING_LEN];
    char malloc_function_name[STRING_LEN];
    int malloc_line;
#endif
// Details about where it was freed.
#if (MEM_TRACK >= 2)
    char free_file_name[STRING_LEN];
    char free_function_name[STRING_LEN];
    int free_line;
#endif
// Stack trace at time of allocation and at time of deallocation.
#if (MEM_TRACK >= 3)
    char **malloc_frame;
    int malloc_frame_size;
    char **free_frame;
    int free_frame_size;
#endif
    // Size allocated of block.
    size_t size;
    allocation_type a_call;
    deallocation_type d_call;
    // Linked list links.
    struct MEM_ELEM *next;
    struct MEM_ELEM *prev;

} MEM_ELEM;

/* No need for explanation ? a struct to contain the MEMory LIST details */
typedef struct MEM_LIST {
    MEM_ELEM *start;
    MEM_ELEM *end;
    unsigned long long size;
} MEM_LIST;

/**
 * Initializes a list to begin using it.
 **/
void init_list(MEM_LIST *list);

/**
 * Add a block to a list.
 **/
void add_block(MEM_LIST *list, MEM_ELEM *elem);

/**
 * Get block with that address from the list.
 * Return Null if not found.
 **/
MEM_ELEM *get_block(MEM_LIST *list, void *addr);

/**
 * Removes the block with the assigned address <addr> from the current_blocks
 *list and returns a pointer to the block. If no block is found with that
 *address, NULL will be returned.
 **/
MEM_ELEM *free_block(MEM_LIST *current_blocks, void *addr);

/**
 * Deletes a list, with all the blocks in it.
 **/
void delete_list(MEM_LIST *list);

/**
 * Prints all block in a list.
 **/
void print_list(MEM_LIST *list);
/**
 * Creats a memory block and puts its allocation details in it. Returns that
 *block.
 **/
#if (MEM_TRACK == 1 || MEM_TRACK == 2)
MEM_ELEM *create_block(void *addr, size_t size, const char *file,
                       const char *func, int line, allocation_type acall);
#elif (MEM_TRACK == 3)
MEM_ELEM *create_block(void *addr, size_t size, const char *file,
                       const char *func, int line, char **stack,
                       size_t stack_size, allocation_type acall);
#else

MEM_ELEM *create_block(void *addr, size_t size, allocation_type acall);

#endif
/**
 * Adds deallocation data to a block.
 **/
#if (MEM_TRACK == 2)
void modify_block(MEM_ELEM *elem, const char *file, const char *func, int line,
                  deallocation_type dcall);
#elif (MEM_TRACK == 3)
void modify_block(MEM_ELEM *elem, const char *file, const char *func, int line,
                  char **stack, size_t stack_size, deallocation_type dcall);
#else

void modify_block(MEM_ELEM *elem, deallocation_type dcall);

#endif

/**
 * Assigns a name to a block.
 **/
void name_block(MEM_ELEM *elem, char *name);

/**
 * Deletes a block and frees all resources taken by it.
 **/
void delete_block(MEM_ELEM *elem);

/**
 * Prints a block in a well formatted way.
 **/
void print_block(MEM_ELEM *elem);

#endif