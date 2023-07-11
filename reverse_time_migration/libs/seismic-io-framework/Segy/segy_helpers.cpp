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


#include "segy_helpers.h"
#include "suheaders.h"

void ebcdicToAscii(unsigned char *s, unsigned char *ascii) {
    //    cout << "in ascii=" << ascii << endl;
    //    cout << "in s=" << s << endl;
    while (*s) {
        *ascii = e2a[(int) (*s)];
        if (*ascii == 'C') {
            int character = (int) *(ascii + 1);
            if (character == 64) // the following character is a space
                *ascii = '\n';
        }
        s++;
        ascii++;
    }
}

void ebcdicToAscii(unsigned char *s) {
    while (*s) {
        //        cout << "normal" << endl;
        *s = e2a[(int) (*s)];
        if (*s == 'C') {
            int character = (int) *(s + 1);
            if (character == 64) // the following character is a space
                *s = '\n';
        }
        s++;
    }
}

void ibm_to_float(int from[], int to[], int n, int endian)
/***********************************************************************
ibm_to_float - convert between 32 bit IBM and IEEE floating numbers
 ************************************************************************
Input::
from    input vector
to    output vector, can be same as input vector
endian    byte order =0 little endian (DEC, PC's)
          =1 other systems
 *************************************************************************
Notes:
Up to 3 bits lost on IEEE -> IBM

Assumes sizeof(int) == 4

IBM -> IEEE may overflow or underflow, taken care of by
substituting large number or zero

Only integer shifting and masking are used.
 *************************************************************************
Credits: CWP: Brian Sumner,  c.1985
 *************************************************************************/
{
    int fconv, fmant, i, t;

    for (i = 0; i < n; ++i) {

        fconv = from[i];

        /* if little endian, i.e. endian=0 do this */
        if (endian == 0)
            fconv = (fconv << 24) | ((fconv >> 24) & 0xff) | ((fconv & 0xff00) << 8) |
                    ((fconv & 0xff0000) >> 8);

        if (fconv) {
            fmant = 0x00ffffff & fconv;
            /* The next two lines were added by Toralf Foerster */
            /* to trap non-IBM format data i.e. conv=0 data  */
            //            if (fmant == 0)
            //                warn("mantissa is zero data may not be in IBM FLOAT
            //                Format !");
            t = (int) ((0x7f000000 & fconv) >> 22) - 130;
            while (!(fmant & 0x00800000)) {
                --t;
                fmant <<= 1;
            }
            if (t > 254)
                fconv = (0x80000000 & fconv) | 0x7f7fffff;
            else if (t <= 0)
                fconv = 0;
            else
                fconv = (0x80000000 & fconv) | (t << 23) | (0x007fffff & fmant);
        }
        to[i] = fconv;
    }
    return;
}

/* Assumes sizeof(int) == 4 */
////void float_to_ibm(int from[], int to[], int n, int endian) {
////    /* float_to_ibm - convert between 32 bit IBM and IEEE floating numbers
////     *
////     * Credits:
////     *  CWP: Brian
////     *
////     * Parameters:
////     *    from  - input vector
////     *    to    - output vector, can be same as input vector
////     *    len   - number of floats in vectors
////     *    type  - conversion type
////     *
////     * Notes:
////     *  Up to 3 bits lost on IEEE -> IBM
////     *
////     *  IBM -> IEEE may overflow or underflow, taken care of by
////     *  substituting large number or zero
////     *
////     *  Only integer shifting and masking are used.
////     *
////     *  This routine assumes a big-endian machine.  If yours is little
////     *  endian you will need to reverse the bytes in ibm_to_float
////     *  with something like
////     *
////     *  fconv = from[i];
////     *  fconv = (fconv<<24) | ((fconv>>24)&0xff) |
////     *      ((fconv&0xff00)<<8) | ((fconv&0xff0000)>>8);
////     *
////     */
////    int fconv, fmant, i, t;
////    /** if little endian, i.e. endian=0 do this */
////    if (endian == 0)
////        fconv = (fconv << 24) | ((fconv >> 24) & 0xff) | ((fconv & 0xff00) << 8) |
////                ((fconv & 0xff0000) >> 8);
////
////    for (i = 0; i < n; ++i) {
////        fconv = from[i];
////        if (fconv) {
////            fmant = (0x007fffff & fconv) | 0x00800000;
////            t = (int) ((0x7f800000 & fconv) >> 23) - 126;
////            while (t & 0x3) {
////                ++t;
////                fmant >>= 1;
////            }
////            fconv = (0x80000000 & fconv) | (((t >> 2) + 64) << 24) | fmant;
////        }
////        to[i] = fconv;
////    }
////    return;
////}

void long_to_float(long from[], float to[], int n, int endian)
/****************************************************************************
Author: J.W. de Bruijn, May 1995
 ****************************************************************************/
{
    int i;

    if (endian == 0) {
        for (i = 0; i < n; ++i) {
            swap_long_4(&from[i]);
            to[i] = (float) from[i];
        }
    } else {
        for (i = 0; i < n; ++i) {
            to[i] = (float) from[i];
        }
    }
}

void short_to_float(short from[], float to[], int n, int endian)
/****************************************************************************
short_to_float - type conversion for additional SEG-Y formats
 *****************************************************************************
Author: Delft: J.W. de Bruijn, May 1995
Modified by: Baltic Sea Reasearch Institute: Toralf Foerster, March 1997
 ****************************************************************************/
{
    int i;

    if (endian == 0) {
        for (i = n - 1; i >= 0; --i) {
            swap_short_2(&from[i]);
            to[i] = (float) from[i];
        }
    } else {
        for (i = n - 1; i >= 0; --i)
            to[i] = (float) from[i];
    }
}

void integer1_to_float(signed char from[], float to[], int n)
/****************************************************************************
integer1_to_float - type conversion for additional SEG-Y formats
 *****************************************************************************
Author: John Stockwell,  2005
 ****************************************************************************/
{
    while (n--) {
        to[n] = from[n];
    }
}

void ieee2ibm(void *to, const void *from, int len) {
    unsigned fr; /* fraction */
    int exp;     /* exponent */
    int sgn;     /* sign */

    for (; len-- > 0; to = (char *) to + 4, from = (char *) from + 4) {
        /* split into sign, exponent, and fraction */
        fr = *(unsigned *) from; /* pick up value */
        sgn = fr >> 31;         /* save sign */
        fr <<= 1;               /* shift sign out */
        exp = fr >> 24;         /* save exponent */
        fr <<= 8;               /* shift exponent out */

        if (exp == 255) { /* infinity (or NAN) - map to largest */
            fr = 0xffffff00;
            exp = 0x7f;
            goto done;
        } else if (exp > 0) /* add assumed digit */
            fr = (fr >> 1) | 0x80000000;
        else if (fr == 0) /* short-circuit for zero */
            goto done;

        /* adjust exponent from base 2 offset 127 radix point after first digit
        to base 16 offset 64 radix point before first digit */
        exp += 130;
        fr >>= -exp & 3;
        exp = (exp + 3) >> 2;

        /* (re)normalize */
        while (fr < 0x10000000) { /* never executed for normalized input */
            --exp;
            fr <<= 4;
        }

        done:
        /* put the pieces back together and return it */
        fr = (fr >> 8) | (exp << 24) | (sgn << 31);
        *(unsigned *) to = htonl(fr);
    }
}

void tapebhed_to_bhed(const tapebhed *tapebhptr, bhed *bhptr)
/****************************************************************************
tapebhed_to_bhed -- converts the seg-y standard 2 byte and 4 byte
        integer header fields to, respectively, the
        machine's short and int types.
 *****************************************************************************
Input:
tapbhed     pointer to array of
 *****************************************************************************
Notes:
The present implementation assumes that these types are actually the "right"
size (respectively 2 and 4 bytes), so this routine is only a placeholder for
the conversions that would be needed on a machine not using this convention.
 *****************************************************************************
Author: CWP: Jack  K. Cohen, August 1994
 ****************************************************************************/
{
    int i;
    Value val;

    /* convert binary header, field by field */
    for (i = 0; i < BHED_NKEYS; ++i) {
        gettapebhval(tapebhptr, i, &val);
        putbhval(bhptr, i, &val);
    }
}

void tapesegy_to_segy(const tapesegy *tapetrptr, segy *trptr)
/****************************************************************************
tapesegy_to_segy -- converts the seg-y standard 2 byte and 4 byte
                    integer header fields to, respectively, the machine's
                    short and int types.
 *****************************************************************************
Input:
tapetrptr   pointer to trace in "tapesegy" (SEG-Y on tape) format

Output:
trptr       pointer to trace in "segy" (SEG-Y as in  SU) format
 *****************************************************************************
Notes:
Also copies float data byte by byte.  The present implementation assumes that
the integer types are actually the "right" size (respectively 2 and 4 bytes),
so this routine is only a placeholder for the conversions that would be needed
on a machine not using this convention.  The float data is preserved as
four byte fields and is later converted to internal floats by ibm_to_float
(which, in turn, makes additonal assumptions).
 *****************************************************************************
Author: CWP:Jack K. Cohen,  August 1994
 ****************************************************************************/
{
    int i;
    Value val;

    /* convert header trace header fields */
    for (i = 0; i < SEGY_NKEYS; ++i) {
        gettapehval(tapetrptr, i, &val);
        puthval(trptr, i, &val);
    }

    /* copy the optional portion */
    memcpy((char *) &(trptr->otrav) + 2, tapetrptr->unass, 60);

    /* copy data portion */
    memcpy(trptr->data, tapetrptr->data, sizeof(float) * SU_NFLTS);
}

void bhed_to_tapebhed(bhed *bhptr, tapebhed *tapebhptr) {
    int i;
    Value val;
    /* convert binary header, field by field */
    for (i = 0; i < BHED_NKEYS; ++i) {
        getbhval(bhptr, i, &val);
        puttapebhval(tapebhptr, i, &val);
    }
}

void segy_to_tapesegy(segy *trptr, tapesegy *tapetrptr, int nsegy) {
    int i;
    Value val;
    /* convert header trace header fields */
    for (i = 0; i < SEGY_NKEYS; ++i) {
        gethval(trptr, i, &val);
        puttapehval(tapetrptr, i, &val);
    }

    /* copy the optional portion */
    memcpy(tapetrptr->unass, (char *) &(trptr->otrav) + 2, 60);
    /* copy data portion */
    memcpy(tapetrptr->data, trptr->data, sizeof(float) * SU_NFLTS);
}
