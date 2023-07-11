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


/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 * File:   swapbyte.h
 * Author: krm
 *
 * Created on June 13, 2018, 12:19 PM
 */

#ifndef SWAPBYTE_H
#define SWAPBYTE_H

/* PC byte swapping */
void swap_short_2(short *tni2);

void swap_u_short_2(unsigned short *tni2);

void swap_int_4(int *tni4);

void swap_u_int_4(unsigned int *tni4);

void swap_long_4(long *tni4);

void swap_u_long_4(unsigned long *tni4);

void swap_float_4(float *tnf4);

void swap_double_8(double *tndd8);

#endif /* SWAPBYTE_H */
