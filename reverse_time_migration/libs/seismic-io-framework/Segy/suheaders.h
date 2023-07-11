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
 * File:   suheaders.h
 * Author: krm
 *
 * Created on June 7, 2018, 6:45 PM
 */

#ifndef SUHEADERS_H
#define SUHEADERS_H

#include "swapbyte.h"
#include <cstdlib>
#include <iostream>
#include <string.h>

#define SU_NFLTS 32767    /**< Arbitrary limit on data array size	*/
#define SEGY_HDRBYTES 240 /**< Bytes in the tape trace header	*/
#define EBCBYTES                                                               \
  3200 /**< Bytes in the card image EBCDIC block (Text Header)                 \
        */

#define SEGY_NKEYS 71 /* Number of mandated header fields  */
#define BHED_NKEYS 27 /* Number of mandated binary fields	*/
#define BNYBYTES 400  /* Bytes in the binary coded block  */
// we currently don't use those values: SEGY_NKEYS, BHED_NKEYS, BNYBYTES
// SEG-Y files have a lot of information.. some of them aren't usually utilized
// by our modules

/**
 * bhedtape - binary header
 - The SEG-Y file contains a lot of data in different formats and sizes
 - The structs that ends with 'tape' reads the headers' values as is
 - and then another struct casts these values to the types we need
 */

typedef struct {

    unsigned int jobid: 32; /**< job identification number */

    unsigned int lino: 32; /**< line number (only one line per reel) */

    unsigned int reno: 32; /**< reel number */

    unsigned int ntrpr: 16; /**< number of data traces per record */

    unsigned int nart: 16; /**< number of auxiliary traces per record */

    unsigned int hdt: 16; /**< sample interval (microsecs) for this reel */

    unsigned int dto: 16; /**< same for original field recording */

    unsigned int hns: 16; /**< number of samples per trace for this reel */

    unsigned int nso: 16; /**< same for original field recording */

    unsigned int format: 16; /** data sample format code:
                            - 1 = floating point (4 bytes)
                            - 2 = fixed point (4 bytes)
                            - 3 = fixed point (2 bytes)
                            - 4 = fixed point w/gain code (4 bytes)
                            */

    unsigned int fold: 16; /**< CDP fold expected per CDP ensemble */

    unsigned int tsort: 16; /**< Trace sorting code (i.e. type of ensemble) :
                               - -1 = Other (should be explained in user
                              Extended Textual File Header stanza
                               - 0 = Unknown
                               - 1 = As recorded (no sorting)
                               - 2 = CDP ensemble
                               - 3 = Single fold continuous profile
                               - 4 = Horizontally stacked
                               - 5 = Common source point
                               - 6 = Common receiver point
                               - 7 = Common offset point
                               - 8 = Common mid-point
                               - 9 = Common conversion point
                               */

    unsigned int vscode: 16; /**< vertical sum code:
                            - 1 = no sum
                            - 2 = two sum ...
                            - N = N sum (N = 32,767)
                            */

    unsigned int hsfs: 16; /**< sweep frequency at start */

    unsigned int hsfe: 16; /**< sweep frequency at end */

    unsigned int hslen: 16; /**< sweep length (ms) */

    unsigned int hstyp: 16; /**< sweep type code:
                           - 1 = linear
                           - 2 = parabolic
                           - 3 = exponential
                           - 4 = other
                           */

    unsigned int schn: 16; /**< trace number of sweep channel */

    unsigned int hstas: 16; /**< sweep trace taper length at start if \n
                      tapered (the taper starts at zero time
                      and is effective for this length) */

    unsigned int hstae: 16; /**< sweep trace taper length at end (the ending
                      taper starts at sweep length minus the taper
                      length at end) */

    unsigned int htatyp: 16; /**< sweep trace taper type code:
                            - 1 = linear
                            - 2 = cos-squared
                            - 3 = other
                            */

    unsigned int hcorr: 16; /**< correlated data traces code:
                           - 1 = no
                           - 2 = yes
                           */

    unsigned int bgrcv: 16; /**< binary gain recovered code:
                           - 1 = yes
                           - 2 = no
                           */

    unsigned int rcvm: 16; /**< amplitude recovery method code:
                          - 1 = none
                          - 2 = spherical divergence
                          - 3 = AGC
                          - 4 = other
                          */

    unsigned int mfeet: 16; /**< measurement system code:
                           - 1 = meters
                           - 2 = feet
                           */

    unsigned int polyt: 16; /**< impulse signal polarity code:
                           - 1 = increase in pressure or upward
                               geophone case movement gives
                               negative number on tape
                           - 2 = increase in pressure or upward
                               geophone case movement gives
                               positive number on tape
                           */

    unsigned int vpol: 16; /**< vibratory polarity code:
                          code    seismic signal lags pilot by
                          - 1       337.5 to  22.5 degrees
                          - 2        22.5 to  67.5 degrees
                          - 3        67.5 to 112.5 degrees
                          - 4       112.5 to 157.5 degrees
                          - 5       157.5 to 202.5 degrees
                          - 6       202.5 to 247.5 degrees
                          - 7       247.5 to 292.5 degrees
                          - 8       293.5 to 337.5 degrees
                          */

    unsigned char hunass[340]; /**< unassigned */

} tapebhed;

// C/C++ tip here we defined a struct and gave it a value at the same time
/** This array is used to convert the types when transforming the header_tape to
 * the header */
static struct {
    const char *key;
    const char *type;
    int offs;
} tapebhdr[] = {
        {"jobid",  "P", 0},
        {"lino",   "P", 4},
        {"reno",   "P", 8},
        {"ntrpr",  "U", 12},
        {"nart",   "U", 14},
        {"hdt",    "U", 16},
        {"dto",    "U", 18},
        {"hns",    "U", 20},
        {"nso",    "U", 22},
        {"format", "U", 24},
        {"fold",   "U", 26},
        {"tsort",  "U", 28},
        {"vscode", "U", 30},
        {"hsfs",   "U", 32},
        {"hsfe",   "U", 34},
        {"hslen",  "U", 36},
        {"hstyp",  "U", 38},
        {"schn",   "U", 40},
        {"hstas",  "U", 42},
        {"hstae",  "U", 44},
        {"htatyp", "U", 46},
        {"hcorr",  "U", 48},
        {"bgrcv",  "U", 50},
        {"rcvm",   "U", 52},
        {"mfeet",  "U", 54},
        {"polyt",  "U", 56},
        {"vpol",   "U", 58},
};

typedef struct {

    /**
     * tapesegy - trace identification header (trace header)
     * there is a trace header for each trace in the SEG-Y
     * 	*/

    unsigned int tracl: 32; /**< trace sequence number within line */

    unsigned int tracr: 32; /**< trace sequence number within reel */

    unsigned int fldr: 32; /**< field record number */

    unsigned int tracf: 32; /**< trace number within field record */

    unsigned int ep: 32; /**< energy source point number */

    unsigned int ensemble_number: 32; /**< CDP ensemble number */

    unsigned int cdpt: 32; /**< trace number within CDP ensemble */

    unsigned int trid: 16; /**< trace identification code:
                  - 1 = seismic data
                  - 2 = dead
                  - 3 = dummy
                  - 4 = time break
                  - 5 = uphole
                  - 6 = sweep
                  - 7 = timing
                  - 8 = water break
                  - 9---, N = optional use (N = 32,767)
                   */

    unsigned int nvs: 16; /**< number of vertically summed traces (see
                 vscode in bhed structure) */

    unsigned int nhs: 16; /**< number of horizontally summed traces (see
                 vscode in bhed structure) */

    unsigned int duse: 16; /**< data use:
                          1 = production
                          2 = test */

    unsigned int offset: 32; /**< distance from source point to receiver
                       group (negative if opposite to direction
                       in which the line was shot) */

    unsigned int gelev: 32; /**< receiver group elevation from sea level
                      (above sea level is positive) */

    unsigned int selev: 32; /**< source elevation from sea level
                      (above sea level is positive) */

    unsigned int sdepth: 32; /**< source depth (positive) */

    unsigned int gdel: 32; /**< datum elevation at receiver group */

    unsigned int sdel: 32; /**< datum elevation at source */

    unsigned int swdep: 32; /**< water depth at source */

    unsigned int gwdep: 32; /**< water depth at receiver group */

    unsigned int scalel: 16; /**< scale factor for previous 7 entries
                       with value plus or minus 10 to the
                       power 0, 1, 2, 3, or 4 (if positive,
                       multiply, if negative divide) */

    unsigned int scalco: 16; /**< scale factor for next 4 entries
                       with value plus or minus 10 to the
                       power 0, 1, 2, 3, or 4 (if positive,
                       multiply, if negative divide) */

    unsigned int sx: 32; /**< X source coordinate */

    unsigned int sy: 32; /**< Y source coordinate */

    unsigned int gx: 32; /**< X group coordinate */

    unsigned int gy: 32; /**< Y source coordinate */

    unsigned int counit: 16; /**< coordinate units code:
                            for previoius four entries
                            1 = length (meters or feet)
                            2 = seconds of arc (in this case, the
                            X values are unsigned intitude and the Y values
                            are latitude, a positive value designates
                            the number of seconds east of Greenwich
                            or north of the equator */

    unsigned int wevel: 16; /**< weathering velocity */

    unsigned int swevel: 16; /**< subweathering velocity */

    unsigned int sut: 16; /**< uphole time at source */

    unsigned int gut: 16; /**< uphole time at receiver group */

    unsigned int sstat: 16; /**< source static correction */

    unsigned int gstat: 16; /**< group static correction */

    unsigned int tstat: 16; /**< total static applied */

    unsigned int laga: 16; /**< lag time A, time in ms between end of 240-
                     byte trace identification header and time
                     break, positive if time break occurs after
                     end of header, time break is defined as
                     the initiation pulse which maybe recorded
                     on an auxiliary trace or as otherwise
                     specified by the recording system */

    unsigned int lagb: 16; /**< lag time B, time in ms between the time
                     break and the initiation time of the energy source,
                     may be positive or negative */

    unsigned int delrt: 16; /**< delay recording time, time in ms between
                      initiation time of energy source and time
                      when recording of data samples begins
                      (for deep water work if recording does not
                      start at zero time) */

    unsigned int muts: 16; /**< mute time--start */

    unsigned int mute: 16; /**< mute time--end */

    unsigned int ns: 16; /**< number of samples in this trace */

    unsigned int dt: 16; /**< sample interval; in micro-seconds */

    unsigned int gain: 16; /**< gain type of field instruments code:
                          1 = fixed
                          2 = binary
                          3 = floating point
                          4 ---- N = optional use */

    unsigned int igc: 16; /**< instrument gain constant */

    unsigned int igi: 16; /**< instrument early or initial gain */

    unsigned int corr: 16; /**< correlated:
                          1 = no
                          2 = yes */

    unsigned int sfs: 16; /**< sweep frequency at start */

    unsigned int sfe: 16; /**< sweep frequency at end */

    unsigned int slen: 16; /**< sweep length in ms */

    unsigned int styp: 16; /**< sweep type code:
                          1 = linear
                          2 = cos-squared
                          3 = other */

    unsigned int stas: 16; /**< sweep trace length at start in ms */

    unsigned int stae: 16; /**< sweep trace length at end in ms */

    unsigned int tatyp: 16; /**< taper type: 1=linear, 2=cos^2, 3=other */

    unsigned int afilf: 16; /**< alias filter frequency if used */

    unsigned int afils: 16; /**< alias filter slope */

    unsigned int nofilf: 16; /**< notch filter frequency if used */

    unsigned int nofils: 16; /**< notch filter slope */

    unsigned int lcf: 16; /**< low cut frequency if used */

    unsigned int hcf: 16; /**< high cut frequncy if used */

    unsigned int lcs: 16; /**< low cut slope */

    unsigned int hcs: 16; /**< high cut slope */

    unsigned int year: 16; /**< year data recorded */

    unsigned int day: 16; /**< day of year */

    unsigned int hour: 16; /**< hour of day (24 hour clock) */

    unsigned int minute: 16; /**< minute of hour */

    unsigned int sec: 16; /**< second of minute */

    unsigned int timbas: 16; /**< time basis code:
                            1 = local
                            2 = GMT
                            3 = other */

    unsigned int trwf: 16; /**< trace weighting factor, defined as 1/2^N
                     volts for the least sigificant bit */

    unsigned int grnors: 16; /**< geophone group number of roll switch
                       position one */

    unsigned int grnofr: 16; /**< geophone group number of trace one within
                       original field record */

    unsigned int grnlof: 16; /**< geophone group number of last trace within
                       original field record */

    unsigned int gaps: 16; /**< gap size (total number of groups dropped) */

    unsigned int otrav: 16; /**< overtravel taper code:
                           1 = down (or behind)
                           2 = up (or ahead) */

    unsigned char unass[60]; /**< unassigned */

    unsigned char data[SU_NFLTS][4]; /**< pseudo_float data[SU_NFLTS]; */

} tapesegy;

static struct {
    const char *key;
    const char *type;
    int offs;
} tapehdr[] = {
        /**< Same as tapebhdr*/
        {"tracl",  "P", 0},
        {"tracr",  "P", 4},
        {"fldr",   "P", 8},
        {"tracf",  "P", 12},
        {"ep",     "P", 16},
        {"cdp",    "P", 20},
        {"cdpt",   "P", 24},
        {"trid",   "U", 28},
        {"nvs",    "U", 30},
        {"nhs",    "U", 32},
        {"duse",   "U", 34},
        {"offset", "P", 36},
        {"gelev",  "P", 40},
        {"selev",  "P", 44},
        {"sdepth", "P", 48},
        {"gdel",   "P", 52},
        {"sdel",   "P", 56},
        {"swdep",  "P", 60},
        {"gwdep",  "P", 64},
        {"scalel", "U", 68},
        {"scalco", "U", 70},
        {"sx",     "P", 72},
        {"sy",     "P", 76},
        {"gx",     "P", 80},
        {"gy",     "P", 84},
        {"counit", "U", 88},
        {"wevel",  "U", 90},
        {"swevel", "U", 92},
        {"sut",    "U", 94},
        {"gut",    "U", 96},
        {"sstat",  "U", 98},
        {"gstat",  "U", 100},
        {"tstat",  "U", 102},
        {"laga",   "U", 104},
        {"lagb",   "U", 106},
        {"delrt",  "U", 108},
        {"muts",   "U", 110},
        {"mute",   "U", 112},
        {"ns",     "U", 114},
        {"dt",     "U", 116},
        {"gain",   "U", 118},
        {"igc",    "U", 120},
        {"igi",    "U", 122},
        {"corr",   "U", 124},
        {"sfs",    "U", 126},
        {"sfe",    "U", 128},
        {"slen",   "U", 130},
        {"styp",   "U", 132},
        {"stas",   "U", 134},
        {"stae",   "U", 136},
        {"tatyp",  "U", 138},
        {"afilf",  "U", 140},
        {"afils",  "U", 142},
        {"nofilf", "U", 144},
        {"nofils", "U", 146},
        {"lcf",    "U", 148},
        {"hcf",    "U", 150},
        {"lcs",    "U", 152},
        {"hcs",    "U", 154},
        {"year",   "U", 156},
        {"day",    "U", 158},
        {"hour",   "U", 160},
        {"minute", "U", 162},
        {"sec",    "U", 164},
        {"timbas", "U", 166},
        {"trwf",   "U", 168},
        {"grnors", "U", 170},
        {"grnofr", "U", 172},
        {"grnlof", "U", 174},
        {"gaps",   "U", 176},
        {"otrav",  "U", 178},
};

typedef union { /** storage for arbitrary type */
    char s[8];
    short h;
    unsigned short u;
    long l;
    unsigned long v;
    int i;
    unsigned int p;
    float f;
    double d;
    unsigned int U: 16;
    unsigned int P: 32;
} Value;

typedef struct {

    /**
     * segy - trace identification header (the one we usually use in the rest of
     the code
     * same as its tape counterpart (tapesegy) but with the format that suits our
       machine's (host) formats and coding needs
     * there is a trace header for each trace in the SEG-Y
     * */

    int tracl; /**< Trace sequence number within line
                --numbers continue to increase if the
                same line continues across multiple
                SEG Y files.
                byte# 1-4
              */

    int tracr; /**< Trace sequence number within SEG Y file
                ---each file starts with trace sequence
                one
                byte# 5-8
              */

    int fldr; /**< Original field record number
               byte# 9-12
            */

    int tracf; /**< Trace number within original field record
                byte# 13-16
             */

    int ep; /**< energy source point number
             ---Used when more than one record occurs
             at the same effective surface location.
             byte# 17-20
           */

    int ensemble_number; /**< Ensemble number (i.e. CDP, CMP, CRP,...)
                       byte# 21-24
                       */

    int cdpt; /**< trace number within the ensemble
               ---each ensemble starts with trace number one.
               byte# 25-28
             */

    short trid; /**< trace identification code:
              -1 = Other
               0 = Unknown
               1 = Seismic data
               2 = Dead
               3 = Dummy
               4 = Time break
               5 = Uphole
               6 = Sweep
               7 = Timing
               8 = Water break
               9 = Near-field gun signature
              10 = Far-field gun signature
              11 = Seismic pressure sensor
              12 = Multicomponent seismic sensor
                      - Vertical component
              13 = Multicomponent seismic sensor
                      - Cross-line component
              14 = Multicomponent seismic sensor
                      - in-line component
              15 = Rotated multicomponent seismic sensor
                      - Vertical component
              16 = Rotated multicomponent seismic sensor
                      - Transverse component
              17 = Rotated multicomponent seismic sensor
                      - Radial component
              18 = Vibrator reaction mass
              19 = Vibrator baseplate
              20 = Vibrator estimated ground force
              21 = Vibrator reference
              22 = Time-velocity pairs
              23 ... N = optional use
                      (maximum N = 32,767)

              Following are CWP id flags:

              109 = autocorrelation
              110 = Fourier transformed - no packing
                   xr[0],xi[0], ..., xr[N-1],xi[N-1]
              111 = Fourier transformed - unpacked Nyquist
                   xr[0],xi[0],...,xr[N/2],xi[N/2]
              112 = Fourier transformed - packed Nyquist
                   even N:
                   xr[0],xr[N/2],xr[1],xi[1], ...,
                      xr[N/2 -1],xi[N/2 -1]
                      (note the exceptional second entry)
                   odd N:
                   xr[0],xr[(N-1)/2],xr[1],xi[1], ...,
                      xr[(N-1)/2 -1],xi[(N-1)/2 -1],xi[(N-1)/2]
                      (note the exceptional second & last entries)
              113 = Complex signal in the time domain
                   xr[0],xi[0], ..., xr[N-1],xi[N-1]
              114 = Fourier transformed - amplitude/phase
                   a[0],p[0], ..., a[N-1],p[N-1]
              115 = Complex time signal - amplitude/phase
                   a[0],p[0], ..., a[N-1],p[N-1]
              116 = Real part of complex trace from 0 to Nyquist
              117 = Imag part of complex trace from 0 to Nyquist
              118 = Amplitude of complex trace from 0 to Nyquist
              119 = Phase of complex trace from 0 to Nyquist
              121 = Wavenumber time domain (k-t)
              122 = Wavenumber frequency (k-omega)
              123 = Envelope of the complex time trace
              124 = Phase of the complex time trace
              125 = Frequency of the complex time trace
              130 = Depth-Range (z-x) traces
              201 = Seismic data packed to bytes (by supack1)
              202 = Seismic data packed to 2 bytes (by supack2)
                 byte# 29-30
              */

    short nvs; /**< Number of vertically summed traces yielding
                this trace. (1 is one trace,
                2 is two summed traces, etc.)
                byte# 31-32
              */

    short nhs; /**< Number of horizontally summed traces yielding
                this trace. (1 is one trace
                2 is two summed traces, etc.)
                byte# 33-34
              */

    short duse; /**< Data use:
                      1 = Production
                      2 = Test
                 byte# 35-36
               */

    int offset; /**< Distance from the center of the source point
                 to the center of the receiver group
                 (negative if opposite to direction in which
                 the line was shot).
                 byte# 37-40
               */

    int gelev; /**< Receiver group elevation from sea level
                (all elevations above the Vertical datum are
                positive and below are negative).
                byte# 41-44
              */

    int selev; /**< Surface elevation at source.
                byte# 45-48
              */

    int sdepth; /**< Source depth below surface (a positive number).
                 byte# 49-52
               */

    int gdel; /**< Datum elevation at receiver group.
               byte# 53-56
            */

    int sdel; /**< Datum elevation at source.
               byte# 57-60
            */

    int swdep; /**< Water depth at source.
                byte# 61-64
             */

    int gwdep; /**< Water depth at receiver group.
                byte# 65-68
             */

    short scalel; /**< Scalar to be applied to the previous 7 entries
                   to give the real value.
                   Scalar = 1, +10, +100, +1000, +10000.
                   If positive, scalar is used as a multiplier,
                   if negative, scalar is used as a divisor.
                   byte# 69-70
                 */

    short scalco; /**< Scalar to be applied to the next 4 entries
                   to give the real value.
                   Scalar = 1, +10, +100, +1000, +10000.
                   If positive, scalar is used as a multiplier,
                   if negative, scalar is used as a divisor.
                   byte# 71-72
                 */

    int sx; /**< Source coordinate - X
             byte# 73-76
          */

    int sy; /**< Source coordinate - Y
             byte# 77-80
          */

    int gx; /**< Group coordinate - X
             byte# 81-84
          */

    int gy; /**< Group coordinate - Y
             byte# 85-88
          */

    short counit; /**<
                * Coordinate units: (for previous 4 entries and
                        for the 7 entries before scalel)
                   - 1 = Length (meters or feet)
                   - 2 = Seconds of arc
                   - 3 = Decimal degrees
                   - 4 = Degrees, minutes, seconds (DMS)

                * In case 2, the X values are longitude and
                  the Y values are latitude, a positive value designates
                  the number of seconds east of Greenwich
                        or north of the equator

                * In case 4, to encode +-DDDMMSS
                  counit = +-DDD*10^4 + MM*10^2 + SS,
                  with scalco = 1. To encode +-DDDMMSS.ss
                  counit = +-DDD*10^6 + MM*10^4 + SS*10^2
                  with scalco = -100.
                     byte# 89-90
                */

    short wevel; /**< Weathering velocity. byte# 91-92 */

    short swevel; /**< Subweathering velocity. byte# 93-94 */

    short sut; /**< Uphole time at source in milliseconds. byte# 95-96 */

    short gut; /**< Uphole time at receiver group in milliseconds. byte# 97-98 */

    short sstat; /**< Source static correction in milliseconds.byte# 99-100 */

    short gstat; /**< Group static correction  in milliseconds. byte# 101-102 */

    short tstat; /**< Total static applied  in milliseconds.
                           (Zero if no static has been applied.)
                           byte# 103-104
                        */

    short laga; /**< Lag time A, time in ms between end of 240-
                                   byte trace identification header and time
                                   break, positive if time break occurs after
                                   end of header, time break is defined as
                                   the initiation pulse which maybe recorded
                                   on an auxiliary trace or as otherwise
                                   specified by the recording system
                                   byte# 105-106
                           */

    short lagb; /**< lag time B, time in ms between the time break
                               and the initiation time of the energy source,
                               may be positive or negative
                               byte# 107-108
                      */

    short delrt; /**< delay recording time, time in ms between
                                        initiation time of energy source and
                  time when recording of data samples begins (for deep water
                  work if recording does not start at zero time) byte# 109-110
                               */

    short muts; /**< mute time--start byte# 111-112 */

    short mute; /**< mute time--end byte# 113-114 */

    unsigned short ns; /**< number of samples in this trace byte# 115-116 */

    unsigned short dt; /**< sample interval; in micro-seconds byte# 117-118 */

    short gain; /**< gain type of field instruments code:
                                      - 1 = fixed
                                      - 2 = binary
                                      - 3 = floating point
                                      - 4 ---- N = optional use
                               byte# 119-120
                              */

    short igc; /**< instrument gain constant byte# 121-122 */

    short igi; /**< instrument early or initial gain byte# 123-124 */

    short corr; /**< correlated:
                                      - 1 = no
                                      - 2 = yes
                              byte# 125-126
                      */

    short sfs; /**< sweep frequency at start byte# 127-128 */

    short sfe; /**< sweep frequency at end byte# 129-130 */

    short slen; /**< sweep length in ms byte# 131-132 */

    short styp; /**< sweep type code:
                                      - 1 = linear
                                      - 2 = cos-squared
                                      - 3 = other
                              byte# 133-134
                      */

    short stas; /**< sweep trace length at start in ms. byte# 135-136 */

    short stae; /**< sweep trace length at end in ms. byte# 137-138 */

    short tatyp; /**< taper type: 1=linear, 2=cos^2, 3=other. byte# 139-140 */

    short afilf; /**< alias filter frequency if used. byte# 141-142 */

    short afils; /**< alias filter slope byte# 143-144 */

    short nofilf; /**< notch filter frequency if used.byte# 145-146 */

    short nofils; /**< notch filter slope. byte# 147-148 */

    short lcf; /**< low cut frequency if used. byte# 149-150 */

    short hcf; /**< high cut frequncy if used. byte# 151-152 */

    short lcs; /**< low cut slope
                byte# 153-154
             */

    short hcs; /**< high cut slope
                byte# 155-156
             */

    short year; /**< year data recorded
                 byte# 157-158
              */

    short day; /**< day of year
                byte# 159-160
             */

    short hour; /**< hour of day (24 hour clock)
                 byte# 161-162
              */

    short minute; /**< minute of hour
                   byte# 163-164
                */

    short sec; /**< second of minute
                byte# 165-166
             */

    short timbas; /**< time basis code:
                                                - 1 = local
                                                - 2 = GMT
                                                - 3 = other
                                        byte# 167-168
                                */

    short trwf; /**< trace weighting factor, defined as 1/2^N
                               volts for the least sigificant bit
                               byte# 169-170
                      */

    short grnors; /**< geophone group number of roll switch
                                                position one
                                         byte# 171-172
                                */

    short grnofr; /**< geophone group number of trace one within
                                                original field record
                                         byte# 173-174
                                */

    short grnlof; /**< geophone group number of last trace within
                                                original field record
                                        byte# 175-176
                                */

    short gaps; /**< gap size (total number of groups dropped)
                                   byte# 177-178
                           */

    short otrav; /**< overtravel taper code:
                                               - 1 = down (or behind)
                                               - 2 = up (or ahead)
                                       byte# 179-180
                               */

#ifdef SLTSU_SEGY_H /* begin Unocal SU segy.h differences */

    /** cwp local assignments
     *
     */

    float d1; /**< sample spacing for non-seismic data
                       byte# 181-184
                    */

    float f1; /**< first sample location for non-seismic data
                       byte# 185-188
                    */

    float d2; /**< sample spacing between traces
                       byte# 189-192
                    */

    float f2; /**< first trace location
                       byte# 193-196
                    */

    float ungpow; /**< negative of power used for dynamic
                   range compression
                   byte# 197-200
                */

    float unscale; /**< reciprocal of scaling factor to normalize
                    range
                    byte# 201-204
                 */

    short mark; /**< mark selected traces
                         byte# 205-206
                      */

    /** SLTSU local assignments */

    short mutb; /**< mute time at bottom (start time)
                         bottom mute ends at last sample
                         byte# 207-208
                      */
    float dz;   /**< depth sampling interval in (m or ft)
                      if =0.0, input are time samples
                         byte# 209-212
                      */

    float fz; /**< depth of first sample in (m or ft)
                       byte# 213-116
                    */

    short n2; /**< number of traces per cdp or per shot
                       byte# 217-218
                    */

    short shortpad; /**< alignment padding
                     byte# 219-220
                  */

    int ntr; /**< number of traces
                      byte# 221-224
                   */

    /**< SLTSU local assignments end */

    short unass[8]; /**< unassigned
                     byte# 225-240
                  */

#else

    /** cwp local assignments */
    float d1; /**< sample spacing for non-seismic data
               byte# 181-184
            */

    float f1; /**< first sample location for non-seismic data
               byte# 185-188
            */

    float d2; /**< sample spacing between traces
               byte# 189-192
            */

    float f2; /**< first trace location
               byte# 193-196
            */

    float ungpow; /**< negative of power used for dynamic
                   range compression
                   byte# 197-200
                */

    float unscale; /**< reciprocal of scaling factor to normalize
                    range
                    byte# 201-204
                 */

    int ntr; /**< number of traces
              byte# 205-208
           */

    short mark; /**< mark selected traces
                 byte# 209-210
              */

    short shortpad; /**< alignment padding
                     byte# 211-212
                  */

    short unass[14]; /**< unassigned--NOTE: last entry causes
              a break in the word alignment, if we REALLY
              want to maintain 240 bytes, the following
              entry should be an odd number of short/UINT2
              OR do the insertion above the "mark" keyword
              entry
              byte# 213-240
           */
#endif

    float data[SU_NFLTS];

} segy;

typedef struct {

    /**
     * bhed - binary header (the one we usually use in the rest of the code
     * same as its tape counterpart (tapebhed) but with the format that suits our
       machine's (host) formats and coding needs
     */

    int jobid; /**< job identification number */

    int lino; /**< line number (only one line per reel) */

    int reno; /**< reel number */

    short ntrpr; /**< number of data traces per record */

    short nart; /**< number of auxiliary traces per record */

    unsigned short hdt; /**< sample interval in micro secs for this reel */

    unsigned short dto; /**< same for original field recording */

    unsigned short hns; /**< number of samples per trace for this reel */

    unsigned short nso; /**< same for original field recording */

    short format; /**< data sample format code:
                                                - 1 = floating point, 4 byte (32
                   bits)
                                                - 2 = fixed point, 4 byte (32
                   bits)
                                                - 3 = fixed point, 2 byte (16
                   bits)
                                                - 4 = fixed point w/gain code, 4
                   byte (32 bits)
                                                - 5 = IEEE floating point, 4
                   byte (32 bits)
                                                - 8 = two's complement integer,
                   1 byte (8 bits)
                                */

    short fold; /**< CDP fold expected per CDP ensemble */

    short tsort; /**< Trace sorting code (i.e. type of ensemble) :
                   - -1 = Other (should be explained in user Extended Textual
                  File Header stanza
                   - 0 = Unknown
                   - 1 = As recorded (no sorting)
                   - 2 = CDP ensemble
                   - 3 = Single fold continuous profile
                   - 4 = Horizontally stacked
                   - 5 = Common source point
                   - 6 = Common receiver point
                   - 7 = Common offset point
                   - 8 = Common mid-point
                   - 9 = Common conversion point
             */

    short vscode; /**< vertical sum code:
                                                - 1 = no sum
                                                - 2 = two sum ...
                                                - N = N sum (N = 32,767)
                                 */

    short hsfs; /**< sweep frequency at start */

    short hsfe; /**< sweep frequency at end */

    short hslen; /**< sweep length (ms) */

    short hstyp; /**< sweep type code:
                                               - 1 = linear
                                               - 2 = parabolic
                                               - 3 = exponential
                                               - 4 = other
                                       */

    short schn; /**< trace number of sweep channel */

    short hstas; /**< sweep trace taper length at start if
                                        tapered (the taper starts at zero time
                                        and is effective for this length)
                               */

    short hstae; /**<  sweep trace taper length at end (the ending
                                         taper starts at sweep length minus the
                  taper length at end)
                                */

    short htatyp; /**< sweep trace taper type code:
                                                - 1 = linear
                                                - 2 = cos-squared
                                                - 3 = other
                                 */

    short hcorr; /**< correlated data traces code:
                                               - 1 = no
                                               - 2 = yes
                                */

    short bgrcv; /**< binary gain recovered code:
                                               - 1 = yes
                                               - 2 = no
                                */

    short rcvm; /**< amplitude recovery method code:
                                      - 1 = none
                                      - 2 = spherical divergence
                                      - 3 = AGC
                                      - 4 = other
                              */

    short mfeet; /**< measurement system code:
                                               - 1 = meters
                                               - 2 = feet
                                */

    short polyt; /**< impulse signal polarity code:
                                               - 1 = increase in pressure or
                  upward geophone case movement gives negative number on tape
                                               - 2 = increase in pressure or
                  upward geophone case movement gives positive number on tape
                            */

    short vpol; /**< vibratory polarity code:
                                      code seismic signal lags pilot by
                                      - 1	337.5 to  22.5 degrees
                                      - 2	 22.5 to  67.5 degrees
                                      - 3	 67.5 to 112.5 degrees
                                      - 4	112.5 to 157.5 degrees
                                      - 5	157.5 to 202.5 degrees
                                      - 6	202.5 to 247.5 degrees
                                      - 7	247.5 to 292.5 degrees
                                      - 8	293.5 to 337.5 degrees
                              */

    short hunass[170]; /**< unassigned */

} bhed;

static struct {
    const char *key;
    const char *type;
    int offs;
} hdr[] = {
        /**< Same as tapebhdr*/
        {"tracl",    "i", 0},
        {"tracr",    "i", 4},
        {"fldr",     "i", 8},
        {"tracf",    "i", 12},
        {"ep",       "i", 16},
        {"cdp",      "i", 20},
        {"cdpt",     "i", 24},
        {"trid",     "h", 28},
        {"nvs",      "h", 30},
        {"nhs",      "h", 32},
        {"duse",     "h", 34},
        {"offset",   "i", 36},
        {"gelev",    "i", 40},
        {"selev",    "i", 44},
        {"sdepth",   "i", 48},
        {"gdel",     "i", 52},
        {"sdel",     "i", 56},
        {"swdep",    "i", 60},
        {"gwdep",    "i", 64},
        {"scalel",   "h", 68},
        {"scalco",   "h", 70},
        {"sx",       "i", 72},
        {"sy",       "i", 76},
        {"gx",       "i", 80},
        {"gy",       "i", 84},
        {"counit",   "h", 88},
        {"wevel",    "h", 90},
        {"swevel",   "h", 92},
        {"sut",      "h", 94},
        {"gut",      "h", 96},
        {"sstat",    "h", 98},
        {"gstat",    "h", 100},
        {"tstat",    "h", 102},
        {"laga",     "h", 104},
        {"lagb",     "h", 106},
        {"delrt",    "h", 108},
        {"muts",     "h", 110},
        {"mute",     "h", 112},
        {"ns",       "u", 114},
        {"dt",       "u", 116},
        {"gain",     "h", 118},
        {"igc",      "h", 120},
        {"igi",      "h", 122},
        {"corr",     "h", 124},
        {"sfs",      "h", 126},
        {"sfe",      "h", 128},
        {"slen",     "h", 130},
        {"styp",     "h", 132},
        {"stas",     "h", 134},
        {"stae",     "h", 136},
        {"tatyp",    "h", 138},
        {"afilf",    "h", 140},
        {"afils",    "h", 142},
        {"nofilf",   "h", 144},
        {"nofils",   "h", 146},
        {"lcf",      "h", 148},
        {"hcf",      "h", 150},
        {"lcs",      "h", 152},
        {"hcs",      "h", 154},
        {"year",     "h", 156},
        {"day",      "h", 158},
        {"hour",     "h", 160},
        {"minute",   "h", 162},
        {"sec",      "h", 164},
        {"timbas",   "h", 166},
        {"trwf",     "h", 168},
        {"grnors",   "h", 170},
        {"grnofr",   "h", 172},
        {"grnlof",   "h", 174},
        {"gaps",     "h", 176},
        {"otrav",    "h", 178},
        {"d1",       "f", 180},
        {"f1",       "f", 184},
        {"d2",       "f", 188},
        {"f2",       "f", 192},
        {"ungpow",   "f", 196},
        {"unscale",  "f", 200},
        {"ntr",      "i", 204},
        {"mark",     "h", 208},
        {"shortpad", "h", 210},
};

static struct {
    const char *key;
    const char *type;
    int offs;
} bhdr[] = {/**< Same as tapebhdr*/
        {"jobid",  "i", 0},
        {"lino",   "i", 4},
        {"reno",   "i", 8},
        {"ntrpr",  "h", 12},
        {"nart",   "h", 14},
        {"hdt",    "h", 16},
        {"dto",    "h", 18},
        {"hns",    "h", 20},
        {"nso",    "h", 22},
        {"format", "h", 24},
        {"fold",   "h", 26},
        {"tsort",  "h", 28},
        {"vscode", "h", 30},
        {"hsfs",   "h", 32},
        {"hsfe",   "h", 34},
        {"hslen",  "h", 36},
        {"hstyp",  "h", 38},
        {"schn",   "h", 40},
        {"hstas",  "h", 42},
        {"hstae",  "h", 44},
        {"htatyp", "h", 46},
        {"hcorr",  "h", 48},
        {"bgrcv",  "h", 50},
        {"rcvm",   "h", 52},
        {"mfeet",  "h", 54},
        {"polyt",  "h", 56},
        {"vpol",   "h", 58}};

void gethval(const segy *tr, int index, Value *valp);

void puthval(segy *tr, int index, Value *valp);

void getbhval(const bhed *bh, int index, Value *valp);

void putbhval(bhed *bh, int index, Value *valp);

void gettapebhval(const tapebhed *tapetr, int index, Value *valp);

void puttapebhval(tapebhed *tapetr, int index, Value *valp);

void gettapehval(const tapesegy *tapetr, int index, Value *valp);

void puttapehval(tapesegy *tapetr, int index, Value *valp);

void swaphval(segy *tr, int index);

void swapbhval(bhed *bh, int index);

bool isequal(Value a, Value b, char *type);

#endif /* SUHEADERS_H */
