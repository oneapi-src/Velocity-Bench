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


#ifndef MEMORY_MANAGER_MEMORY_TRACKER_H
#define MEMORY_MANAGER_MEMORY_TRACKER_H

/**
 * Not memory tracking, then not compiling !.
 **/
#ifdef MEM_TRACK

#include <stdlib.h>

#define malloc(size) omalloc(size, __FILE__, __LINE__, __FUNCTION__)
#define free(ptr) ofree(ptr, __FILE__, __LINE__, __FUNCTION__)

#ifndef MEM_C_ONLY

void *operator new(size_t size);
void *operator new[](size_t size);

void operator delete(void *ptr);
void operator delete[](void *ptr);

#endif

#define _mm_malloc(size, align)                                                \
  omm_malloc(size, align, __FILE__, __LINE__, __FUNCTION__)

#define _mm_free(ptr) omm_free(ptr, __FILE__, __LINE__, __FUNCTION__)

#define realloc(ptr, new_size)                                                 \
  orealloc(ptr, new_size, __FILE__, __LINE__, __FUNCTION__)

#define calloc(num, size) ocalloc(num, size, __FILE__, __LINE__, __FUNCTION__)

/* Initializer function that will get executed before main */
void memory_track_startup(void) __attribute__((constructor));

/* Clean up function that will get executed after main */
void memory_track_cleanup(void) __attribute__((destructor));
/**
 * Gives a name to a pointer to keep track of it in the final output.
 **/
void name_ptr(void *ptr, char *name);
// Signal handler, will catch signals like segmentation fault and print the
// stack trace of it.
void seg_handler(int signum);

#ifdef __cplusplus
extern "C" {
#endif
// override malloc
extern void *omalloc(size_t size, const char *file, int l, const char *func);
// override realloc
extern void *orealloc(void *ptr, size_t new_size, const char *file, int l,
                      const char *func);
// override calloc
extern void *ocalloc(size_t num, size_t size, const char *file, int l,
                     const char *func);
// override free
extern void ofree(void *ptr, const char *file, int l, const char *func);

#ifndef MEM_NO_INTEL
// override _mm_alloc
extern void *omm_malloc(size_t size, size_t align, char const *file, int l,
                        char const *func);

// override mem_free
extern void omm_free(void *ptr, char const *file, int l, char const *func);
#endif
#ifdef __cplusplus
}
#endif
/**
 * Prints done data at time of checkpoint.
 **/
void mem_checkpoint(char *name);

/**
 * Registers c++ file to track.
 **/
void register_cpp_file(char *name);

#else

/**
 * Empty functions to prevent code dependencies of outer programs.s
 **/
inline void name_ptr(void *ptr, char *name) { ; }

inline void mem_checkpoint(char *name) { ; }

inline void register_cpp_file(char *name) { ; }

#endif

#endif //MEMORY_MANAGER_MEMORY_TRACKER_H
