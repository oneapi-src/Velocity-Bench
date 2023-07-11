/* 
 * Copyright (C) <2023> Intel Corporation
 * 
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License, as published
 * by the Free Software Foundation; either version 2 of the License,
 * or (at your option) any later version.
 *  
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *  
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, see <http://www.gnu.org/licenses/>.
 *  
 * 
 * SPDX-License-Identifier: GPL-2.0-or-later
 * 
 */ 

/*
 * Wrappers to emulate dlopen() on other systems like Windows
 */

#include "wraphelper.h"

#if defined(_WIN32)
void *wrap_dlopen(const char *filename)
{
    return (void *)LoadLibrary(filename);
}
void *wrap_dlsym(void *h, const char *sym)
{
    return (void *)GetProcAddress((HINSTANCE)h, sym);
}
int wrap_dlclose(void *h)
{
    /* FreeLibrary returns non-zero on success */
    return (!FreeLibrary((HINSTANCE)h));
}
#else
/* assume we can use dlopen itself... */
void *wrap_dlopen(const char *filename)
{
    return dlopen(filename, RTLD_NOW);
}
void *wrap_dlsym(void *h, const char *sym)
{
    return dlsym(h, sym);
}
int wrap_dlclose(void *h)
{
    return dlclose(h);
}
#endif
