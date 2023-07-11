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


#include <memory-manager/data-units/mem_list.h>

#include <memory-manager/utils//mem_utils.h>
#include <memory-manager/utils/logger.h>

#include <stdio.h>
#include <string.h>

void init_list(MEM_LIST *list) {
    list->start = NULL;
    list->end = NULL;
    list->size = 0;
}

int matching_calls(allocation_type alc, deallocation_type dealc) {
    if ((int) alc == (int) dealc) {
        return 1;
    }
    if ((alc == CALLOC || alc == MALLOC || alc == A_REALLOC) &&
        (dealc == FREE || dealc == D_REALLOC)) {
        return 1;
    }
    return 0;
}

void call_type_to_string(char *str, int type, int allocation) {
    if (allocation) {
        switch (type) {
            case MALLOC:
                strncpy(str, "malloc", MAX_MSG_LEN);
                break;
            case _MM_MALLOC:
                strncpy(str, "_mm_malloc", MAX_MSG_LEN);
                break;
            case A_REALLOC:
                strncpy(str, "realloc", MAX_MSG_LEN);
                break;
            case NEW:
                strncpy(str, "new", MAX_MSG_LEN);
                break;
            case CALLOC:
                strncpy(str, "calloc", MAX_MSG_LEN);
                break;
            case NEW_ARR:
                strncpy(str, "new[]", MAX_MSG_LEN);
                break;
            default:
                strncpy(str, "Unknown function", MAX_MSG_LEN);
        }
    } else {
        switch (type) {
            case FREE:
                strncpy(str, "free", MAX_MSG_LEN);
                break;
            case _MM_FREE:
                strncpy(str, "_mm_free", MAX_MSG_LEN);
                break;
            case D_REALLOC:
                strncpy(str, "realloc", MAX_MSG_LEN);
                break;
            case DELETE:
                strncpy(str, "delete", MAX_MSG_LEN);
                break;
            case DELETE_ARR:
                strncpy(str, "delete[]", MAX_MSG_LEN);
                break;
            default:
                strncpy(str, "Unknown function", MAX_MSG_LEN);
        }
    }
}

void add_block(MEM_LIST *list, MEM_ELEM *elem) {
    if (list->size == 0) {
        list->start = elem;
        list->end = elem;
        elem->next = NULL;
        elem->prev = NULL;
    } else {
        elem->next = NULL;
        elem->prev = list->end;
        list->end->next = elem;
        list->end = elem;
    }
    list->size++;
}

MEM_ELEM *get_block(MEM_LIST *list, void *addr) {
    // We search from the end : a newly allocated block is more probable to be
    // deallocated.
    MEM_ELEM *temp = list->end;
    while (temp != NULL) {
        if (temp->addr == addr) {
            break;
        }
        temp = temp->prev;
    }
    return temp;
}

MEM_ELEM *free_block(MEM_LIST *current_blocks, void *addr) {
    // Get block.
    MEM_ELEM *elem = get_block(current_blocks, addr);
    if (elem != NULL) {
        // Remove block from current_blocks list.
        if (current_blocks->size == 1) {
            init_list(current_blocks);
        } else {
            if (current_blocks->start == elem) {
                current_blocks->start = elem->next;
                current_blocks->start->prev = NULL;
            } else if (current_blocks->end == elem) {
                current_blocks->end = elem->prev;
                current_blocks->end->next = NULL;
            } else {
                elem->prev->next = elem->next;
                elem->next->prev = elem->prev;
            }
            current_blocks->size--;
        }
    }
    return elem;
}

void delete_list(MEM_LIST *list) {
    MEM_ELEM *temp = list->start;
    MEM_ELEM *temp2 = list->start;
    while (temp != NULL) {
        temp2 = temp->next;
        delete_block(temp);
        temp = temp2;
    }
}

void print_list(MEM_LIST *list) {
    if (list != NULL) {
        MEM_ELEM *temp = list->start;
        while (temp != NULL) {
            print_block(temp);
            temp = temp->next;
        }
    }
}

#if (MEM_TRACK == 1 || MEM_TRACK == 2)
MEM_ELEM *create_block(void *addr, size_t size, const char *file,
                       const char *func, int line, allocation_type acall) {
  MEM_ELEM *elem = (MEM_ELEM *)(malloc)(sizeof(MEM_ELEM));
  elem->addr = addr;
  elem->size = size;
#if (MEM_TRACK == 2)
  elem->free_line = -1;
#endif
  elem->a_call = acall;
  strncpy(elem->name, "\0", STRING_LEN);
  strncpy(elem->malloc_file_name, file, STRING_LEN);
  strncpy(elem->malloc_function_name, func, STRING_LEN);
  elem->malloc_line = line;
  return elem;
}
#elif (MEM_TRACK == 3)
MEM_ELEM *create_block(void *addr, size_t size, const char *file,
                       const char *func, int line, char **stack,
                       size_t stack_size, allocation_type acall) {
  MEM_ELEM *elem = (MEM_ELEM *)(malloc)(sizeof(MEM_ELEM));
  elem->addr = addr;
  elem->size = size;
  strncpy(elem->name, "\0", STRING_LEN);
  strncpy(elem->malloc_file_name, file, STRING_LEN);
  strncpy(elem->malloc_function_name, func, STRING_LEN);
  elem->malloc_line = line;
  elem->free_line = -1;
  elem->a_call = acall;
  elem->malloc_frame_size = stack_size;
  elem->malloc_frame = stack;
  return elem;
}
#else

MEM_ELEM *create_block(void *addr, size_t size, allocation_type acall) {
    MEM_ELEM *elem = (MEM_ELEM *) (malloc)(sizeof(MEM_ELEM));
    strncpy(elem->name, "\0", STRING_LEN);
    elem->addr = addr;
    elem->a_call = acall;
    elem->size = size;
    return elem;
}

#endif

#if (MEM_TRACK == 2)
void modify_block(MEM_ELEM *elem, const char *file, const char *func, int line,
                  deallocation_type dcall) {
  strncpy(elem->free_file_name, file, STRING_LEN);
  strncpy(elem->free_function_name, func, STRING_LEN);
  elem->free_line = line;
  elem->d_call = dcall;
}
#elif (MEM_TRACK == 3)
void modify_block(MEM_ELEM *elem, const char *file, const char *func, int line,
                  char **stack, size_t stack_size, deallocation_type dcall) {
  strncpy(elem->free_file_name, file, STRING_LEN);
  strncpy(elem->free_function_name, func, STRING_LEN);
  elem->free_line = line;
  elem->free_frame_size = stack_size;
  elem->free_frame = stack;
  elem->d_call = dcall;
}
#else

void modify_block(MEM_ELEM *elem, deallocation_type dcall) {
    elem->d_call = dcall;
}

#endif

void delete_block(MEM_ELEM *elem) {
#if (MEM_TRACK >= 3)
    (free)(elem->malloc_frame);
    if (elem->free_line != -1) {
      (free)(elem->free_frame);
    }
#endif
    (free)(elem);
}

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

void name_block(MEM_ELEM *elem, char *name) {
    strncpy(elem->name, name, STRING_LEN);
    elem->name[STRING_LEN - 1] = '\0';
}

void print_block(MEM_ELEM *elem) {
    static char msg[MAX_MSG_LEN];
    char trace_line[MAX_MSG_LEN];
    int i = 0;
#if (MEM_TRACK >= 0)
    if (strcmp(elem->name, "\0") != 0) {
        sprintf(msg, MAG "\tPointer Name : %s\n" RESET, elem->name);
        log_msg(msg);
    }
    sprintf(msg,
            MAG "\tMemory pointer : %p\n\t\t" GRN "Size Allocated : %lu\n" RESET,
            elem->addr, elem->size);
    log_msg(msg);
#endif
#if (MEM_TRACK >= 1)
    sprintf(msg,
            RED "\t\tAllocated In " RESET "(" GRN "FILE" RESET ":" CYN
                "LINE" RESET ":" BLU "FUNCTION" RESET ") : " GRN "%s" RESET
                ":" CYN "%d" RESET ":" BLU "%s" RESET " \n",
            elem->malloc_file_name, elem->malloc_line,
            elem->malloc_function_name);
    log_msg(msg);
    call_type_to_string(trace_line, elem->a_call, 1);
    sprintf(msg, CYN "\t\tAllocation method :" GRN " %s\n" RESET, trace_line);
    log_msg(msg);
#endif
    char file[MAX_MSG_LEN], func[MAX_MSG_LEN];
    int l;
#if (MEM_TRACK >= 3)
    log_msg(BLU "\t\tStack Trace At Allocation Call :\n" RESET);
    strcpy(file, "no_info");
    for (i = STACK_SKIP;
         i < elem->malloc_frame_size && file[strlen(file) - 1] != 'h'; i++) {
      extract_info(file, func, &l, elem->malloc_frame[i], 0);
      get_stack_trace_line(trace_line, elem->malloc_frame[i]);
      if (i != STACK_SKIP) {
        decrement_line(trace_line);
      }
      sprintf(msg, WHT "\t\t\t%s\n" YEL "\t\t\t\t%s\n" RESET,
              elem->malloc_frame[i], trace_line);
      log_msg(msg);
    }
#endif
#if (MEM_TRACK >= 2)
    if (elem->free_line != -1) {
      sprintf(msg,
              RED "\t\tDeallocated In " RESET "(" GRN "FILE" RESET ":" CYN
                  "LINE" RESET ":" BLU "FUNCTION" RESET ") : " GRN "%s" RESET
                  ":" CYN "%d" RESET ":" BLU "%s" RESET " \n",
              elem->free_file_name, elem->free_line, elem->free_function_name);
      log_msg(msg);
      call_type_to_string(trace_line, elem->d_call, 0);
      sprintf(msg, CYN "\t\tDeallocation method :" GRN " %s\n" RESET, trace_line);
      log_msg(msg);
    } else {
      log_msg(RED "\t\tNo Deallocation Was Issued or Info could not be obtained "
                  "according to where this data belongs !\n" RESET);
    }
#endif
#if (MEM_TRACK >= 3)
    if (elem->free_line != -1) {
      log_msg(BLU "\t\tStack Trace At Deallocation Call :\n" RESET);
      strcpy(file, "no_info");
      for (i = STACK_SKIP;
           i < elem->free_frame_size && file[strlen(file) - 1] != 'h'; i++) {
        extract_info(file, func, &l, elem->free_frame[i], 0);
        get_stack_trace_line(trace_line, elem->free_frame[i]);
        if (!(elem->d_call == D_REALLOC && i == STACK_SKIP)) {
          decrement_line(trace_line);
        }
        sprintf(msg, WHT "\t\t\t%s\n" YEL "\t\t\t\t%s\n" RESET,
                elem->free_frame[i], trace_line);
        log_msg(msg);
      }
    }
#endif
}
