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


#ifndef MEMORY_MANAGER_STRING_LIST_H
#define MEMORY_MANAGER_STRING_LIST_H

#define MULT_RATE 2

typedef struct {
    char **strings_list;
    int size;
    int max_size;
} string_list;

void init_slist(string_list *list, int size);

void add_slist(string_list *list, char *word);

char slist_contains(string_list *list, char *word);

void destroy_slist(string_list *list);

#endif //MEMORY_MANAGER_STRING_LIST_H
