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
 * File:   suheaders.cpp
 * Author: krm
 *
 * Created on June 7, 2018, 6:45 PM
 */

#include "suheaders.h"

void gethval(const segy *tr, int index, Value *valp) {
    char *tp = (char *) tr;

    switch (*(hdr[index].type)) {
        case 's':
            (void) strcpy(valp->s, tp + hdr[index].offs);
            break;
        case 'h':
            valp->h = *((short *) (tp + hdr[index].offs));
            break;
        case 'u':
            valp->u = *((unsigned short *) (tp + hdr[index].offs));
            break;
        case 'i':
            valp->i = *((int *) (tp + hdr[index].offs));
            break;
        case 'p':
            valp->p = *((unsigned int *) (tp + hdr[index].offs));
            break;
        case 'l':
            valp->l = *((long *) (tp + hdr[index].offs));
            break;
        case 'v':
            valp->v = *((unsigned long *) (tp + hdr[index].offs));
            break;
        case 'f':
            valp->f = *((float *) (tp + hdr[index].offs));
            break;
        case 'd':
            valp->d = *((double *) (tp + hdr[index].offs));
            break;
        default:
            std::cout << __FILE__ << ": " << __LINE__ << ": mysterious data type"
                      << std::endl;
            break;
    }

    return;
}

void puthval(segy *tr, int index, Value *valp) {
    char *tp = (char *) tr;

    switch (*(hdr[index].type)) {
        case 's':
            (void) strcpy(tp + hdr[index].offs, valp->s);
            break;
        case 'h':
            *((short *) (tp + hdr[index].offs)) = valp->h;
            break;
        case 'u':
            *((unsigned short *) (tp + hdr[index].offs)) = valp->u;
            break;
        case 'i':
            *((int *) (tp + hdr[index].offs)) = valp->i;
            break;
        case 'p':
            *((unsigned int *) (tp + hdr[index].offs)) = valp->p;
            break;
        case 'l':
            *((long *) (tp + hdr[index].offs)) = valp->l;
            break;
        case 'v':
            *((unsigned long *) (tp + hdr[index].offs)) = valp->v;
            break;
        case 'f':
            *((float *) (tp + hdr[index].offs)) = valp->f;
            break;
        case 'd':
            *((double *) (tp + hdr[index].offs)) = valp->d;
            break;
        default:
            std::cout << __FILE__ << ":" << __LINE__ << ": mysterious data type"
                      << std::endl;
            break;
    }

    return;
}

void getbhval(const bhed *bh, int index, Value *valp) {
    char *bhp = (char *) bh;

    switch (*(bhdr[index].type)) {
        case 'h':
            valp->h = *((short *) (bhp + bhdr[index].offs));
            break;
        case 'i':
            valp->i = *((int *) (bhp + bhdr[index].offs));
            break;
        default:
            std::cout << __FILE__ << ": " << __LINE__ << ": mysterious data type"
                      << std::endl;
            break;
            break;
    }

    return;
}

void putbhval(bhed *bh, int index, Value *valp) {
    char *bhp = (char *) bh;

    switch (*(bhdr[index].type)) {
        case 'h':
            *((short *) (bhp + bhdr[index].offs)) = valp->h;
            break;
        case 'i':
            *((int *) (bhp + bhdr[index].offs)) = valp->i;
            break;
        default:
            std::cout << __FILE__ << ":" << __LINE__ << ": mysterious data type"
                      << std::endl;
            break;
    }

    return;
}

void gettapehval(const tapesegy *tr, int index, Value *valp) {
    char *tp = (char *) tr;

    switch (*(tapehdr[index].type)) {
        case 'U':
            valp->h = (short) *((short *) (tp + tapehdr[index].offs));
            break;
        case 'P':
            valp->i = (int) *((int *) (tp + tapehdr[index].offs));
            break;
        default:
            std::cout << __FILE__ << ":" << __LINE__ << ": mysterious data type"
                      << std::endl;
            exit(EXIT_FAILURE);
            break;
    }

    return;
}

void puttapehval(tapesegy *tr, int index, Value *valp) {
    char *tp = (char *) tr;

    switch (*(tapehdr[index].type)) {
        case 'U':
            *((short *) (tp + tapehdr[index].offs)) = valp->h;
            break;
        case 'P':
            *((int *) (tp + tapehdr[index].offs)) = valp->i;
            break;
        default:
            std::cout << __FILE__ << ":" << __LINE__ << ": mysterious data type"
                      << std::endl;
            exit(EXIT_FAILURE);
            break;
    }

    return;
}

void gettapebhval(const tapebhed *tr, int index, Value *valp) {
    char *tp = (char *) tr;

    switch (*(tapebhdr[index].type)) {
        case 'U':
            valp->h = (short) *((short *) (tp + tapebhdr[index].offs));
            break;
        case 'P':
            valp->i = (int) *((int *) (tp + tapebhdr[index].offs));
            break;
        default:
            std::cout << __FILE__ << ":" << __LINE__ << ": mysterious data type"
                      << std::endl;
            exit(EXIT_FAILURE);
            break;
    }

    return;
}

void puttapebhval(tapebhed *bh, int index, Value *valp) {
    char *bhp = (char *) bh;

    switch (*(tapebhdr[index].type)) {
        case 'U':
            *((short *) (bhp + tapebhdr[index].offs)) = valp->h;
            break;
        case 'P':
            *((int *) (bhp + tapebhdr[index].offs)) = valp->i;
            break;
        default:
            std::cout << __FILE__ << ":" << __LINE__ << ": mysterious data type"
                      << std::endl;
            exit(EXIT_FAILURE);
            break;
    }

    return;
}

void swaphval(segy *tr, int index) {
    char *tp = (char *) tr;

    switch (*(hdr[index].type)) {
        case 'h':
            swap_short_2((short *) (tp + hdr[index].offs));
            break;
        case 'u':
            swap_u_short_2((unsigned short *) (tp + hdr[index].offs));
            break;
        case 'i':
            swap_int_4((int *) (tp + hdr[index].offs));
            break;
        case 'p':
            swap_u_int_4((unsigned int *) (tp + hdr[index].offs));
            break;
        case 'l':
            swap_long_4((long *) (tp + hdr[index].offs));
            break;
        case 'v':
            swap_u_long_4((unsigned long *) (tp + hdr[index].offs));
            break;
        case 'f':
            swap_float_4((float *) (tp + hdr[index].offs));
            break;
        case 'd':
            swap_double_8((double *) (tp + hdr[index].offs));
            break;
        default:
            std::cout << __FILE__ << ":" << __LINE__ << ": unsupported data type"
                      << std::endl;
            break;
    }

    return;
}

void swapbhval(bhed *bh, int index) {
    char *bhp = (char *) bh;

    switch (*(bhdr[index].type)) {
        case 'h':
            swap_short_2((short *) (bhp + bhdr[index].offs));
            break;
        case 'i':
            swap_int_4((int *) (bhp + bhdr[index].offs));
            break;
        default:
            std::cout << __FILE__ << ":" << __LINE__ << ": unsupported data type"
                      << std::endl;
            break;
    }

    return;
}

bool isequal(Value a, Value b, char *type) {
    switch (*type) {
        case 'h':
            return a.h == b.h;
        case 'u':
            return a.u == b.u;
        case 'l':
            return a.l == b.l;
        case 'v':
            return a.v == b.v;
        case 'i':
            return a.i == b.i;
        case 'p':
            return a.p == b.p;
        case 'f':
            return a.f == b.f;
        case 'd':
            return a.d == b.d;
        case 'U':
            return a.U == b.U;
        case 'P':
            return a.P == b.P;
        default:
            std::cout << __FILE__ << ":" << __LINE__ << ": unsupported data type"
                      << std::endl;
            return false;
    }
}

/* Display non-null header field values */
// void printheader(const segy *tp)
//{
//	int i;			/* index over header fields		*/
//	int j;			/* index over non-null header fields	*/
//	Value val;		/* value in header field		*/
//	cwp_String type;	/* ... its data type			*/
//	cwp_String key;		/* ... the name of the header field	*/
//	Value zeroval;		 /* zero value to compare with		*/
//
//	zeroval.l = 0;
//	j = 0;
//	for (i = 0; i < SU_NKEYS; i++) {
//		gethval(tp, i, &val);
//		key = getkey(i);
//		type = hdtype(key);
//		if (valcmp(type, val, zeroval)) { /* not equal to zero */
//			(void) printf(" %s=", key);
//			printfval(type, val);
//			if ((++j % 6) == 0) putchar('\n');
//		}
//	}
//	putchar('\n');
//
//	return;
//}
