/*
 * EasyWave - A realtime tsunami simulation program with GPU support.
 * Copyright (C) 2014  Andrey Babeyko, Johannes Spazier
 * GFZ German Research Centre for Geosciences (http://www.gfz-potsdam.de)
 *
 * Parts of this program (especially the GPU extension) were developed
 * within the context of the following publicly funded project:
 * - TRIDEC, EU 7th Framework Programme, Grant Agreement 258723
 *   (http://www.tridec-online.eu)
 *
 * Licensed under the EUPL, Version 1.1 or - as soon they will be approved by
 * the European Commission - subsequent versions of the EUPL (the "Licence"),
 * complemented with the following provision: For the scientific transparency
 * and verification of results obtained and communicated to the public after
 * using a modified version of the work, You (as the recipient of the source
 * code and author of this modified version, used to produce the published
 * results in scientific communications) commit to make this modified source
 * code available in a repository that is easily and freely accessible for a
 * duration of five years after the communication of the obtained results.
 *
 * You may not use this work except in compliance with the Licence.
 *
 * You may obtain a copy of the Licence at:
 * https://joinup.ec.europa.eu/software/page/eupl
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the Licence is distributed on an "AS IS" basis,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the Licence for the specific language governing permissions and
 * limitations under the Licence.
 */

/*
 * Modifications Copyright (C) 2023 Intel Corporation
 * 
 * This Program is subject to the terms of the European Union Public License 1.2
 * 
 * If a copy of the license was not distributed with this file, you can obtain one at 
 * https://joinup.ec.europa.eu/sites/default/files/custom-page/attachment/2020-03/EUPL-1.2%20EN.txt
 * 
 * SPDX-License-Identifier: EUPL-1.2
 */

// last modified 11.07.2012
#ifndef UTIL_H
#define UTIL_H

#include <stdio.h>
#include <math.h>

#define MaxFileRecordLength 16384
#define RealMax             1.e+30
#define RealMin             1.e-30

#define My_PI     3.14159265358979
#define g2r(x)    (((double)(x)) * My_PI / 180)
#define r2g(x)    (((double)(x)) / My_PI * 180)
#define cosdeg(x) cos(g2r(x))
#define sindeg(x) sin(g2r(x))
#define tandeg(x) tan(g2r(x))

#define My_max(a, b) (((a) > (b)) ? (a) : (b))
#define My_min(a, b) (((a) < (b)) ? (a) : (b))

// Class Message
#define MSG_OUTSCRN 1
#define MSG_OUTFILE 2
class cMsg
{
  protected:
    int  enabled;
    int  channel;
    char fname[64];

  public:
    cMsg();
    ~cMsg();
    void enable();
    void disable();
    void setchannel(int newchannel);
    void setfilename(const char *newfilename);
    int  print(const char *fmt, ...);
};
extern cMsg Msg;

// Class Error message
class cErrMsg : public cMsg
{

  public:
    cErrMsg();
    ~cErrMsg();
    int post(const char *fmt, ...);
};
extern cErrMsg Err;

// Class Log
class cLog : public cMsg
{
  private:
    int timestamp_enabled;

  public:
    cLog();
    ~cLog();
    void start(const char *filename);
    void timestamp_enable();
    void timestamp_disable();
    int  print(const char *fmt, ...);
};
extern cLog Log;

/***  Error messages and logging  ***/
char *utlCurrentTime();

// Following error functions are outdated. Use global Err object instead.
int   utlPostError(const char *message);
char *utlErrMsgEndOfFile(const char *fname);
char *utlErrMsgReadFile(const char *fname, int line, const char *expected);

/*** Command-line processing ***/
int utlCheckCommandLineOption(int argc, char **argv, const char *option, int letters_to_compare);

/***  String handling  ***/
int   utlStringReadOption(char *record, char *option_name, char *contents);
int   utlReadSequenceOfInt(char *line, int *value);
int   utlReadSequenceOfDouble(char *line, double *value);
int   utlPickUpWordFromString(char *string, int n, char *word);
char *utlPickUpWordFromString(char *string, int n);
char *utlPickUpWordFromString(char *string, char *pos, char *word);
int   utlCountWordsInString(char *record);
char *utlWords2String(int nwords, char **word);
int   utlSubString(char *logstr, int p1, int p2, char *substr);
void  utlPrintToString(char *string, int position, char *insert);

/***  File handling  ***/
int   utlFindNextInputField(FILE *fp);
int   utlReadNextRecord(FILE *fp, char *record, int *line);
int   utlReadNextDataRecord(FILE *fp, char *record, int *line);
char *utlFileFindRecord(char *fname, char *pattern, char *record);
int   utlFileParceToString(FILE *fp, char *pattern);
int   utlGetNumberOfRecords(const char *fname);
int   utlGetNumberOfDataRecords(const char *fname);
int   utlFileReadOption(char *fname, char *option_name, char *contents);
char *utlFileAddExtension(const char *fname, const char *addext);
int   utlReadXYdata(char *fname, double **x, double **y);

/***   MATH & GEOMETRY   ***/
int utlPointInPolygon(int nvrtx, double *xvrtx, double *yvrtx, double x, double y);

/***  O T H E R S    ***/
void  utlTimeSplit(double ctime, int &nHour, int &nMin, int &nSec, int &nMsec);
char *utlTimeSplitString(double ctime);
char *utlTimeSplitString(int ctime);
int   utlTimeHour(double timesec);
int   utlTimeHour(int timesec);
int   utlTimeMin(double timesec);
int   utlTimeMin(int timesec);
int   utlTimeSec(double timesec);
int   utlTimeSec(int timesec);

#endif // UTIL_H
