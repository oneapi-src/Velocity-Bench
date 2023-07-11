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


#ifndef MEMORY_MANAGER_LOGGER_H
#define MEMORY_MANAGER_LOGGER_H

/**
 * Initalize Logger options (Should only be called once at the start):
 * - console_enable : if false(0) disables output on console. Otherwise, will
 *output on console.
 * - file_enable : if false(0) disables output on the file named file_name.
 *Otherwise, will output on the file.
 * - file_name : The name of the file to write to. Will overwrite the file.
 *Won't matter if file_enable is false.
 **/
void init_logger(char console_enable, char file_enable, char *file_name);

/*
 * - Prints the msg string to the valid streams.
 */
void log_msg(char *msg);

/*
 * Closes the logger and releases all taken resources.
 */
void close_logger(void);

/*
 * Flushs output streams
 */
void log_flush(void);

#endif
