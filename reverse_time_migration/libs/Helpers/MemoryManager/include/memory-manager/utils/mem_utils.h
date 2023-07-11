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


#ifndef MEMORY_MANAGER_MEM_UTILS_H
#define MEMORY_MANAGER_MEM_UTILS_H

#define MAX_MSG_LEN 1000

#include <stdlib.h>

/**
 * Extracts the various information from a trace line and returns them.
 **/
void extract_address(char *trace, char *file, char *function, char *offset,
                     char *address);

/**
 * Returns the starting address of the .text section in memory.
 **/
long int get_start_address(void);

/**
 * Processes the trace line to make it more beautiful.
 **/
void process_trace_line(char *traceline);

/**
 * Transforms a stack trace to contain a line number.
 **/
void get_stack_trace_line(char *line, char *trace);

/**
 * Given the required depth of the stack trace, it will return an array of
 *strings of a size stored in actual_size variable which at maximum is depth
 *+ 2.
 * + 2 because it assumes depth wanted is given without taking into account the
 *call of the function and its internal calls. The pointer it returns should be
 *freed to prevent memory leaking.
 **/
char **get_stack_trace(int depth, size_t *actual_size);

/**
 * Takes a processed stack trace that got transformed to contain line number and
 *decrements that line number.
 **/
void decrement_line(char *line);

/**
 * Get Trace line with the given offset in the stack trace.
 **/
void get_trace_line(int offset, char *trace);

/**
 * Extracts different info from the trace line.
 **/
void extract_info(char *filename, char *function, int *line, char *trace,
                  char decr);

/**
 * Gets the pure name of the trace.
 **/
void filter_file_name(char *filtered_name, char *name);

#endif
