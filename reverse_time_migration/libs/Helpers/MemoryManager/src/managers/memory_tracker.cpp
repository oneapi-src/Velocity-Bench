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


#include "memory-manager/managers/memory_tracker.h"

#ifdef MEM_TRACK

#include "logger.h"
#include "mem_list.h"
#include "mem_utils.h"
#include "string_list.h"

#include <execinfo.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef MEM_STACK_SIZE
#define MEM_STACK_SIZE 10
#endif

#ifndef MEM_LOG_NAME
#define MEM_LOG_NAME "memory_track.log"
#endif

#define MAX_MSG_LEN 1000

#ifdef MEM_TRACK_CONSOLE
#define RED "\x1B[31m"
#define GRN "\x1B[32m"
#define YEL "\x1B[33m"
#define BLU "\x1B[34m"
#define MAG "\x1B[35m"
#define CYN "\x1B[36m"
#define WHT "\x1B[37m"
#define RESET "\x1B[0m"
#else
#define RED ""
#define GRN ""
#define YEL ""
#define BLU ""
#define MAG ""
#define CYN ""
#define WHT ""
#define RESET ""
#endif

static unsigned long long total_bytes_allocated;
static unsigned long long bytes_allocated_remaining;
static unsigned long long potential_leaks;
static long long num_mallocs;
static char msg[MAX_MSG_LEN];
static MEM_LIST current_list;
static MEM_LIST free_list;
static MEM_LIST potential_leaks_list;
static string_list file_filter = {0, 0, 0};
#ifdef MEM_TRACK_REG
static char file_filter_disable = 0;
#else
static char file_filter_disable = 1;
#endif
static char system_open = 0;

void allocation_call(void *ptr, size_t size, char const *file, int line,
                     char const *func, char *func_name, allocation_type alloc);

void deallocation_call(void *ptr, char const *file, int line, char const *func,
                       void (*dealloc_func)(void *), char *func_name,
                       deallocation_type deloc);

void allocation_call(void *ptr, size_t size, char const *file, int line,
                     char const *func, char *func_name, allocation_type alloc) {
  if (ptr == NULL) {
    sprintf(msg, YEL "WARNING : %s FAILED at %s:%d:%s\n" RESET, func_name, file,
            line, func);
    log_msg(msg);
  } else {
    MEM_ELEM *elem;
#if (MEM_TRACK == 0)
    elem = create_block(ptr, size, alloc);
#elif (MEM_TRACK == 1 || MEM_TRACK == 2)
    elem = create_block(ptr, size, file, func, line, alloc);
#elif (MEM_TRACK == 3)
    char **strings;
    size_t size_f;
    strings = get_stack_trace(MEM_STACK_SIZE, &size_f);
    elem = create_block(ptr, size, file, func, line, strings, size_f, alloc);
#else
#error Memory Tracker Level is unsupported only 0 to 3 are supported.
#endif
    if (system_open) {
      add_block(&current_list, elem);
      total_bytes_allocated += size;
      bytes_allocated_remaining += size;
      num_mallocs++;
    } else {
      sprintf(msg,
              "System down allocation : pointer %p with size %lu allocated "
              "with %s in %s:%d:%s\n",
              ptr, size, func_name, file, line, func);
      log_msg(msg);
      delete_block(elem);
    }
  }
}

void deallocation_call(void *ptr, char const *file, int line, char const *func,
                       void (*dealloc_func)(void *), char *func_name,
                       deallocation_type deloc) {
  if (system_open) {
    MEM_ELEM *elem = free_block(&current_list, ptr);
    if (elem == NULL) {
      if (system_open) {
        sprintf(
            msg,
            RED
            "ERROR : %s issued at an invalid pointer %p  at %s:%d:%s\n" RESET,
            func_name, ptr, file, line, func);
        log_msg(msg);
      } else {
        sprintf(msg,
                "System down deallocation : pointer %p deallocated with %s in "
                "%s:%d:%s\n",
                ptr, func_name, file, line, func);
        log_msg(msg);
        if (dealloc_func != NULL) {
          (*dealloc_func)(ptr);
        }
      }
    } else {
      if (dealloc_func != NULL) {
        (*dealloc_func)(ptr);
      }
      size_t size = elem->size;
#if (MEM_TRACK == 2)
      modify_block(elem, file, func, line, deloc);
#elif (MEM_TRACK == 3)
      char **strings;
      size_t size_f;
      strings = get_stack_trace(MEM_STACK_SIZE, &size_f);
      modify_block(elem, file, func, line, strings, size_f, deloc);
#else
      modify_block(elem, deloc);
#endif
      if (matching_calls(elem->a_call, elem->d_call)) {
#if (MEM_TRACK >= 2)
#ifndef MEM_LEAKS_ONLY
        add_block(&free_list, elem);
#else
        delete_block(elem);
#endif
#else
        delete_block(elem);
#endif
      } else {
#if (MEM_TRACK >= 1)
        add_block(&potential_leaks_list, elem);
#else
        delete_block(elem);
#endif
        potential_leaks++;
      }
      num_mallocs--;
      bytes_allocated_remaining -= size;
    }
  } else {
    sprintf(msg,
            "System down de-allocation : pointer %p with de-allocated with %s "
            "in %s:%d:%s\n",
            ptr, func_name, file, line, func);
    log_msg(msg);
  }
}

#ifndef MEM_C_ONLY
void *operator new(size_t size) {
  char file[MAX_MSG_LEN] = "file-name";
  char ffile[MAX_MSG_LEN];
  char func[MAX_MSG_LEN] = "func-name";
  int l = 0;
  char trace[MAX_MSG_LEN];
  get_trace_line(1, trace);
  extract_info(file, func, &l, trace, 0);
  int itrace = 2;
  char prefix[6];
  char oldTrace[MAX_MSG_LEN];
  for (int i = 0; i < 5; i++) {
    prefix[i] = file[i];
  }
  prefix[5] = '\0';
  while (strcmp(file, "addr2line") == 0 || strcmp(prefix, "/usr/") == 0) {
    strncpy(oldTrace, trace, MAX_MSG_LEN);
    get_trace_line(itrace, trace);
    if (strcmp(trace, "") == 0 || strcmp(trace, oldTrace) == 0 ||
        file[strlen(file) - 1] == 'h') {
      break;
    }
    extract_info(file, func, &l, trace, 0);
    itrace++;
    for (int i = 0; i < 5; i++) {
      prefix[i] = file[i];
    }
    prefix[5] = '\0';
  }
  if (strcmp(trace, "") == 0) {
    get_trace_line(1, trace);
    extract_info(file, func, &l, trace, 0);
  }
  filter_file_name(ffile, file);
  void *ptr = (malloc)(size);
  if (file_filter_disable || slist_contains(&file_filter, ffile)) {
    allocation_call(ptr, size, file, l, func, "new", NEW);
  }
  return ptr;
}

void *operator new[](size_t size) {
  char file[MAX_MSG_LEN] = "file-name";
  char func[MAX_MSG_LEN] = "func-name";
  char ffile[MAX_MSG_LEN];
  int l = 0;
  char trace[MAX_MSG_LEN];
  get_trace_line(1, trace);
  extract_info(file, func, &l, trace, 0);
  int itrace = 2;
  char prefix[6];
  char oldTrace[MAX_MSG_LEN];
  for (int i = 0; i < 5; i++) {
    prefix[i] = file[i];
  }
  prefix[5] = '\0';
  while (strcmp(file, "addr2line") == 0 || strcmp(prefix, "/usr/") == 0) {
    strncpy(oldTrace, trace, MAX_MSG_LEN);
    get_trace_line(itrace, trace);
    if (strcmp(trace, "") == 0 || strcmp(trace, oldTrace) == 0 ||
        file[strlen(file) - 1] == 'h') {
      break;
    }
    extract_info(file, func, &l, trace, 0);
    itrace++;
    for (int i = 0; i < 5; i++) {
      prefix[i] = file[i];
    }
    prefix[5] = '\0';
  }
  if (strcmp(trace, "") == 0) {
    get_trace_line(1, trace);
    extract_info(file, func, &l, trace, 0);
  }
  filter_file_name(ffile, file);
  void *ptr = (malloc)(size);
  if (file_filter_disable || slist_contains(&file_filter, ffile)) {
    allocation_call(ptr, size, file, l, func, "new[]", NEW_ARR);
  }
  return ptr;
}

void operator delete(void *ptr) {
  char file[MAX_MSG_LEN] = "file-name";
  char func[MAX_MSG_LEN] = "func-name";
  char ffile[MAX_MSG_LEN];
  int l = 0;
  char trace[MAX_MSG_LEN];
  get_trace_line(1, trace);
  extract_info(file, func, &l, trace, 1);
  int itrace = 2;
  while (strcmp(file, "addr2line") == 0) {
    get_trace_line(itrace, trace);
    extract_info(file, func, &l, trace, 1);
    itrace++;
  }
  filter_file_name(ffile, file);
#ifdef MEM_TRACK_REG
  MEM_ELEM *elem = get_block(&current_list, ptr);
  if (elem != NULL || file_filter_disable ||
      slist_contains(&file_filter, ffile)) {
    deallocation_call(ptr, file, l, func, &free, "delete", DELETE);
  } else {
    (free)(ptr);
  }
#else
  deallocation_call(ptr, file, l, func, &free, "delete", DELETE);
#endif
}
void operator delete[](void *ptr) {
  char file[MAX_MSG_LEN] = "file-name";
  char func[MAX_MSG_LEN] = "func-name";
  char ffile[MAX_MSG_LEN];
  int l = 0;
  char trace[MAX_MSG_LEN];
  get_trace_line(1, trace);
  extract_info(file, func, &l, trace, 1);
  int itrace = 2;
  while (strcmp(file, "addr2line") == 0) {
    get_trace_line(itrace, trace);
    extract_info(file, func, &l, trace, 1);
    itrace++;
  }
  filter_file_name(ffile, file);
#ifdef MEM_TRACK_REG
  MEM_ELEM *elem = get_block(&current_list, ptr);
  if (elem != NULL || file_filter_disable ||
      slist_contains(&file_filter, ffile)) {
    deallocation_call(ptr, file, l, func, &free, "delete[]", DELETE_ARR);
  } else {
    (free)(ptr);
  }
#else
  deallocation_call(ptr, file, l, func, &free, "delete[]", DELETE_ARR);
#endif
}

#endif

void memory_track_startup(void) {
#ifdef MEM_TRACK_CONSOLE
  char ce = 1;
  char fe = 0;
#else
  char ce = 0;
  char fe = 1;
#endif
  init_logger(ce, fe, MEM_LOG_NAME);
  total_bytes_allocated = 0;
  bytes_allocated_remaining = 0;
  num_mallocs = 0;
  potential_leaks = 0;
  init_list(&current_list);
  init_list(&free_list);
  init_list(&potential_leaks_list);
  signal(SIGSEGV, seg_handler);
  init_slist(&file_filter, 2);
  system_open = 1;
}

void print_results(void) {
#if (MEM_TRACK >= 2)
  log_msg("===================================================================="
          "==============================================\n");
  if (free_list.size > 0) {
    log_msg(GRN "All Completed Allocations Data :\n" RESET);
    print_list(&free_list);
  } else {
    log_msg(GRN "No Completed Allocations\n" RESET);
  }
#endif
#if (MEM_TRACK >= 1)
  log_msg("===================================================================="
          "==============================================\n");
  if (potential_leaks_list.size > 0) {
    log_msg(RED "All Potenial Memory Leaks Data :\n" RESET);
    print_list(&potential_leaks_list);
  } else {
    log_msg(GRN "No Potenial Leaks Found !\n" RESET);
  } /**/
  log_msg("===================================================================="
          "==============================================\n");
  if (current_list.size > 0) {
    log_msg(RED "All Memory Leaks Data :\n" RESET);
    print_list(&current_list);
  } else {
    log_msg(GRN "No Memory Leaks Found !\n" RESET);
  }
#endif
  log_msg("===================================================================="
          "==============================================\n");
  sprintf(msg, BLU "Total bytes allocated during run : " RESET "%llu\n",
          total_bytes_allocated);
  log_msg(msg);
  sprintf(msg, RED "Bytes still allocated :" RESET " %llu\n",
          bytes_allocated_remaining);
  log_msg(msg);
  sprintf(msg, RED "Potential Leaks : " RESET "%llu\n", potential_leaks);
  log_msg(msg);
  sprintf(msg, YEL "Allocations - Deallocations : " RESET "%lld\n",
          num_mallocs);
  log_msg(msg);
  log_msg("===================================================================="
          "==============================================\n");
  /**/
}

void memory_track_cleanup(void) {
  system_open = 0;
  printf("Begining Final Report Formatting :\n\tCompleted allocations list "
         "size = %llu\n\tPotential leaks list size = %llu\n\tLeaks list size = "
         "%llu\n\tLeak to total allocation percentage : %f\n",
         free_list.size, potential_leaks_list.size, current_list.size,
         total_bytes_allocated == 0 ? 0.0
                                    : (((double)bytes_allocated_remaining) /
                                       ((double)total_bytes_allocated)) *
                                          100.0);
  sprintf(msg, "\n\n***********************************************************"
               "*******************************************************\n");
  log_msg(msg);
  sprintf(msg, "*************************************************** End Report "
               "***************************************************\n");
  log_msg(msg);
  sprintf(msg, "***************************************************************"
               "***************************************************\n\n\n");
  log_msg(msg);
  print_results();
  if (bytes_allocated_remaining == 0) {
    printf(msg, GRN "No Memory Leak Detected !\n" RESET);
    log_msg(msg);
  } else {
    sprintf(msg, RED "Memory Leaks Detected :  %llu !\n" RESET, num_mallocs);
    log_msg(msg);
  }
  delete_list(&current_list);
  delete_list(&free_list);
  delete_list(&potential_leaks_list);
  close_logger();
  destroy_slist(&file_filter);
}

void mem_checkpoint(char *name) {
  sprintf(msg, "Checkpoint Name : %s\n", name);
  log_msg(msg);
  print_results();
  delete_list(&free_list);
  delete_list(&potential_leaks_list);
  init_list(&free_list);
  init_list(&potential_leaks_list);
  log_flush();
}

void name_ptr(void *ptr, char *name) {
  MEM_ELEM *elem = get_block(&current_list, ptr);
  if (elem != NULL) {
    name_block(elem, name);
  }
}

void register_cpp_file(char *name) { add_slist(&file_filter, name); }

void seg_handler(int signum) {

  int i;
  char **strings;
  size_t size_f;
  strings = get_stack_trace(MEM_STACK_SIZE + 1, &size_f);
  char temp[MAX_MSG_LEN];
  log_msg("===================================================================="
          "==============================================\n");
  log_msg(RED "Segmentation fault detected !\n" RESET);
  for (i = 3; i < size_f; i++) {
    get_stack_trace_line(temp, strings[i]);
    if (strcmp(temp, "\0") == 0 && i == 0) {
      log_msg(YEL "No Support for 'addr2line' : only stack trace will be "
                  "printed\nUse objdump with -l flag to trace error\n" RESET);
    }
    if (i != 3) {
      decrement_line(temp);
    }
    sprintf(msg, WHT "\t\t%s\n" RESET YEL "\t\t\t%s\n" RESET, strings[i], temp);
    log_msg(msg);
  }
  (free)(strings);
  exit(signum);
}

void *omalloc(size_t size, const char *file, int l, const char *func) {
  void *ptr = (malloc)(size);
  allocation_call(ptr, size, file, l, func, "malloc", MALLOC);
  return ptr;
}

void *orealloc(void *ptr, size_t new_size, const char *file, int l,
               const char *func) {
  void *new_ptr = (realloc)(ptr, new_size);
  deallocation_call(ptr, file, l, func, NULL, "realloc", D_REALLOC);
  allocation_call(new_ptr, new_size, file, l, func, "realloc", A_REALLOC);
  return new_ptr;
}

void *ocalloc(size_t num, size_t size, const char *file, int l,
              const char *func) {
  void *ptr = (calloc)(num, size);
  size *= num;
  allocation_call(ptr, size, file, l, func, "calloc", CALLOC);
  return ptr;
}

void ofree(void *ptr, const char *file, int l, const char *func) {
  deallocation_call(ptr, file, l, func, &free, "free", FREE);
}

#ifdef __INTEL_COMPILER

#undef _mm_malloc
// override _mm_malloc
void *omm_malloc(size_t size, size_t align, char const *file, int l,
                 char const *func) {
  void *ptr = _mm_malloc(size, align);
  allocation_call(ptr, size, file, l, func, "_mm_malloc", _MM_MALLOC);
  return ptr;
}
#define _mm_malloc(size, align)                                                \
  omm_malloc(size, align, __FILE__, __LINE__, __FUNCTION__)

#undef _mm_free
// override _mm_free
void omm_free(void *ptr, char const *file, int l, char const *func) {
  deallocation_call(ptr, file, l, func, &_mm_free, "_mm_free", _MM_FREE);
}
#define _mm_free(ptr) omm_free(ptr, __FILE__, __LINE__, __FUNCTION__)

#endif

#endif
