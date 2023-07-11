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


#include <memory-manager/utils/mem_utils.h>

#include <execinfo.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DELIMITER "()[] +"
#define BIG_STACK_SIZE 1000

static long int start_section = -1;
static char msg[1000];
extern char _start;

long int get_start_address(void) {
    // If first call calculate start address.
    if (start_section == -1) {
        // Another way was by getting the biggest trace one can get, and calculating
        // from it. Uncomment this and comment the simple working way on your own
        // responsibility.

        /*
        void *array[BIG_STACK_SIZE + 1];
        size_t size_f;
        char **strings;
        size_f = backtrace (array, BIG_STACK_SIZE + 1);
        strings = backtrace_symbols (array, size_f);
        char address1[MAX_MSG_LEN], file1[MAX_MSG_LEN], function1[MAX_MSG_LEN],
        offset1[MAX_MSG_LEN]; extract_address(strings[size_f - 1], file1, function1,
        offset1, address1); start_section = strtol(address1, NULL, 0) -
        strtol(offset1, NULL, 0); (free)(strings);
        */
        start_section = (unsigned long) &_start;
    }
    return start_section;
}

void get_trace_line(int offset, char *trace) {
    char **strings;
    size_t size_f;
    strings = get_stack_trace(offset + 1, &size_f);
    if (offset + 2 < size_f) {
        strncpy(trace, strings[offset + 2], MAX_MSG_LEN);
    } else {
        strncpy(trace, "", MAX_MSG_LEN);
    }
    (free)(strings);
}

void extract_info(char *filename, char *function, int *line, char *trace,
                  char decr) {
    char offset[MAX_MSG_LEN];
    char address[MAX_MSG_LEN];
    char line_trace[MAX_MSG_LEN];
    int l = 0;
    extract_address(trace, filename, function, offset, address);
    get_stack_trace_line(line_trace, trace);
    if (decr) {
        decrement_line(line_trace);
    }

    char *token = strtok(line_trace, ":");
    int i = 0;
    while (token && i < 2) {
        if (i == 0) {
            strncpy(filename, token, MAX_MSG_LEN);
        } else {
            *line = atoi(token);
        }
        token = strtok(NULL, ":");
        i++;
    }
}

void filter_file_name(char *filtered_name, char *name) {
    int index = strlen(name) - 1;
    while (name[index] != '/' && index >= 0) {
        index--;
    }
    strncpy(filtered_name, name + index + 1, MAX_MSG_LEN);
}

void get_stack_trace_line(char *line, char *trace) {
    char address[MAX_MSG_LEN], file[MAX_MSG_LEN], function[MAX_MSG_LEN],
            offset[MAX_MSG_LEN];
    // Extract data from trace.
    extract_address(trace, file, function, offset, address);
    // Get addresses.
    long int address_i = strtol(address, NULL, 0);
    long int start_address = get_start_address();
    FILE *fp;
    char path[BIG_STACK_SIZE];
    // Create the command to get the line.
    sprintf(msg, "addr2line -e %s -j .text 0x%lx 2>&1\n", file,
            address_i - start_address);
    /* Open the command for reading. */
    fp = popen(msg, "r");
    if (fp == NULL) {
        strncpy(line, "\0", MAX_MSG_LEN);
    } else {
        /* Read the output. */
        while (fgets(path, sizeof(path) - 1, fp) != NULL) {
            sprintf(msg, "%s", path);
            strncpy(line, msg, MAX_MSG_LEN);
        }
        /* close */
        pclose(fp);
        if (line[0] == '?') {
            strncpy(line, "Information could not be obtained !!", MAX_MSG_LEN);
        } else {
            process_trace_line(line);
            if (line[strlen(line) - 1] == '\n') {
                line[strlen(line) - 1] = '\0';
            }
            strncat(line, ":", MAX_MSG_LEN);
            strncat(line, function, MAX_MSG_LEN);
            if (line[strlen(line) - 1] == '\n') {
                line[strlen(line) - 1] = '\0';
            }
        }
    }
}

void process_trace_line(char *traceline) {
    char s[MAX_MSG_LEN];
    strncpy(s, traceline, MAX_MSG_LEN);
    s[MAX_MSG_LEN - 1] = '\0';
    char *token = strtok(s, ":");
    strncpy(traceline, "", MAX_MSG_LEN);
    char temp[MAX_MSG_LEN];
    int i = 0;
    while (token && i < 2) {
        if (i != 0) {
            strncat(traceline, ":", MAX_MSG_LEN);
        };
        strncat(traceline, token, MAX_MSG_LEN);
        token = strtok(NULL, ":");
        i++;
    }
}

void extract_address(char *trace, char *file, char *function, char *offset,
                     char *address) {
    char s[MAX_MSG_LEN];
    strcpy(s, trace);
    char *token = strtok(s, DELIMITER);
    int i = 0;
    while (token && i < 4) {
        if (i == 0) {
            strncpy(file, token, MAX_MSG_LEN);
        } else if (i == 1) {
            strncpy(function, token, MAX_MSG_LEN);
        } else if (i == 2) {
            strncpy(offset, token, MAX_MSG_LEN);
        } else if (i == 3) {
            strncpy(address, token, MAX_MSG_LEN);
        }
        token = strtok(NULL, DELIMITER);
        i++;
    }

    sprintf(s, "c++filt %s", function);
    FILE *fp = popen(s, "r");
    char out[MAX_MSG_LEN];

    /* Read the output. */
    while (fgets(out, sizeof(out) - 1, fp) != NULL) {
        strncpy(function, out, MAX_MSG_LEN);
    }
    /* close */
    pclose(fp);
    if (function[strlen(function) - 1] == '\n') {
        function[strlen(function) - 1] = '\0';
    }
}

char **get_stack_trace(int depth, size_t *size_f) {
    void *array[depth + 2];
    char **strings;
    *size_f = backtrace(array, depth + 2);
    strings = backtrace_symbols(array, *size_f);
    return strings;
}

void decrement_line(char *line) {
    char s[MAX_MSG_LEN];
    strncpy(s, line, MAX_MSG_LEN);
    s[MAX_MSG_LEN - 1] = '\0';
    char *token = strtok(s, ":");
    strncpy(line, "", MAX_MSG_LEN);
    char temp[MAX_MSG_LEN];
    int i = 0;
    while (token) {
        if (i != 0) {
            strncat(line, ":", MAX_MSG_LEN);
        }
        strncpy(temp, token, MAX_MSG_LEN);
        temp[MAX_MSG_LEN - 1] = '\0';
        token = strtok(NULL, ":");
        if (i == 1) {
            int line_number = atoi(temp);
            line_number--;
            sprintf(temp, "%d", line_number);
        }
        strncat(line, temp, MAX_MSG_LEN);
        line[MAX_MSG_LEN - 1] = '\0';
        i++;
    }
    if (strcmp(line, "-1") == 0) {
        strncpy(line, "Information could not be obtained !", MAX_MSG_LEN);
    }
}
