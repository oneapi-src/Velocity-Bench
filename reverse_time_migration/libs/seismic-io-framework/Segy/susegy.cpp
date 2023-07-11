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


#include "susegy.h"
#include <set>

using namespace std;

namespace suselect {

    bool all(segy *trace, vector<SEGYelement> *check_elements) { return true; }

    bool ifequal(segy *trace, vector<SEGYelement> *check_elements) {
        Value trace_value;
        for (auto &element : *check_elements) {
            trace_value = trace->*(element.member);
            if (element.GetType() == 'i')
                //                cout << "trace_value " << trace_value.i << ", element"
                //                << element.value.i << endl;

                //            if (element.first.first == 'h')
                //                cout << "trace_value " << trace_value.h << ", element"
                //                << element.second.h << endl;

                //        if (cnt_x >10) exit(EXIT_FAILURE);
                if (!element.IsEqual(*trace)) {
                    return false;
                }
        }
        return true;
    }

    int GetSelected(segy *trace, vector<SEGYelement> *check_elements) {
        Value trace_value;
        for (auto &element : *check_elements) {
            trace_value = trace->*(element.member);
            if (element.GetType() == 'i')
                //                cout << "trace_value " << trace_value.i << ", element"
                //                << element.value.i << endl;

                //            if (element.first.first == 'h')
                //                cout << "trace_value " << trace_value.h << ", element"
                //                << element.second.h << endl;

                //        if (cnt_x >10) exit(EXIT_FAILURE);
                return trace_value.i;
        }
        return -1;
    }

    bool first(segy *trace, vector<SEGYelement> *check_elements) {
        if (trace->tracr == 1)
            return true;
        else
            return false;
    }
} // namespace suselect

vector<uint> SUSegy::GetUniqueOccurences(string filename, vector<SEGYelement> *select_element, uint min_threshold,
                                         uint max_threshold) {
    if (!file.is_open())
        ReadBinaryHeader(filename, false);
    if (!file.is_open()) {
        cout << "File '" << filename << "' could not be opened, will be skipped" << std::endl;
        return vector<uint>();
    }
    memset((char *) bh.hunass, 0, 340);
    set<uint> unique_ids;
    int itr = 0;
    //cout << "nsegy " << nsegy << endl;
    while (itr < INT_MAX) {
        tapesegy tapetrace;
        segy trace;
        file.read((char *) &tapetrace, nsegy); // reading a single trace
        if (!file) { // error in reading (reaching the end of the file)
            break;
        }
        tapesegy_to_segy(&tapetrace, &trace); // converts to easily readable format
        // (oblivious to the type int/short)
        if (endian == 0)
            for (int i = 0; i < SEGY_NKEYS; ++i)
                swaphval(&trace, i); // swaps the bytes if the machine is little endian
        uint unique_value = suselect::GetSelected(&trace, select_element);
        if (unique_value >= min_threshold && unique_value <= max_threshold) {
            unique_ids.insert(unique_value);
        }
    }

    file.close();
    vector<uint> results;
    results.insert(results.end(), unique_ids.begin(), unique_ids.end());
    return results;
}

void SUSegy::ReadBinaryHeader(std::string const &filename, bool exit_on_error) {

    file.open(filename.c_str(), ifstream::binary);
    //cout << filename << endl;
    if (!file.is_open()) {
        if (exit_on_error) {
            cout << "ERROR:: file '" << filename << "' doesn't exit" << endl;
            exit(EXIT_FAILURE);
        } else {
            return;
        }
    }
    memset(ebcdictextheader, 0, EBCBYTES + 1);
    memset(asciitextheader, 0, EBCBYTES + 1);

    /* reading the ebcdic header and converting it to ascii */
    file.read(ebcdictextheader, EBCBYTES);
    ebcdicToAscii((unsigned char *) ebcdictextheader,
                  (unsigned char *) asciitextheader);
    // cout << "text header\n" << asciitextheader << endl;
    // Printing text header
    //    for(int i=0; i<40; i++){
    //        for(int j=0; j<80; j++){
    //            printf("%c", asciitextheader[i*80 + j]);
    //        }
    //        printf("\n");
    //    }
    /* reading the binary header */
    memset(binaryheader, 0, BNYBYTES + 1);
    file.read(binaryheader, BNYBYTES);
    tapebhed tapebh;
    memset((void *) &tapebh, 0, BNYBYTES);
    memcpy((void *) &tapebh, binaryheader, BNYBYTES);
    memset((void *) &tapebh.hunass, 0, 340);
    // ofstream file2;

    /*file2.open("bintapedump", std::ofstream::out);
    file2.write((char*) &tapebh, sizeof(tapebh));
    file2.close();*/
    memset(&bh, 0, BNYBYTES);
    tapebhed_to_bhed(&tapebh, &bh);

    /*file2.open("binreadump", std::ofstream::out);
    file2.write((char*) &bh, sizeof(binaryheader));
    file2.close();*/

    /* if little endian machine, swap bytes in binary header */
    union {
        short s;
        char c[2];
    } testend; // testing if the system is little or big endian
    testend.s = 1;
    endian = (testend.c[0] == '\0') ? 1 : 0;
    if (endian == 0)
        for (int i = 0; i < BHED_NKEYS; ++i)
            swapbhval(&bh, i);

    /* size of whole trace in bytes */
    //cout << "bh.format " << bh.format << endl;
    switch (bh.format) {
        case 8:
            nsegy = bh.hns + SEGY_HDRBYTES;
            break;
        case 3:
            nsegy = bh.hns * 2 + SEGY_HDRBYTES;
            break;
        case 1:
        case 2:
        case 5:
        default:
            nsegy = bh.hns * 4 + SEGY_HDRBYTES;
    }

    /* number of extended text headers */
    nextended = *((short *) (((unsigned char *) &tapebh) + 304));
    if (endian == 0)
        swap_short_2((short *) &nextended);
    if (nextended > 0) { /* number of extended text headers > 0 */
        for (int i = 0; i < nextended; i++) {
            /* cheat -- an extended text header is same size as
             * EBCDIC header */
            /* Read the bytes from the tape for one xhdr into the
             * buffer */
            char ebcbuf_ebcdic[EBCBYTES + 1] = {0};
            char ebcbuf_ascii[EBCBYTES + 1] = {0};
            file.read(ebcbuf_ebcdic, EBCBYTES);
            extendedebcdictextheader.push_back(ebcbuf_ebcdic);
            ebcdicToAscii((unsigned char *) ebcbuf_ebcdic,
                          (unsigned char *) ebcbuf_ascii);
            extendedasciitextheader.push_back(ebcbuf_ascii);
        }
    }
    /*  printf("Bin Header Attributes: \n");

      printf("bh.format = %d\n", bh.format);
      printf("bh.tsort = %d\n", bh.tsort);
      printf("bh.hdt = %d\n", bh.hdt);
      printf("bh.hns = %d\n", bh.hns);
      printf("bh.mfeet = %d\n", bh.mfeet);
      cout <<   "bh.ntrpr = "   << bh.ntrpr << endl;

      */
}

void SUSegy::ReadHeadersAndTraces(
        std::string const &filename, vector<SEGYelement> *check_elements,
        bool (*check_func)(segy *trace, vector<SEGYelement> *check_elements)) {

    if (!file.is_open())
        ReadBinaryHeader(filename);
    memset((char *) bh.hunass, 0, 340);

    int itr = 0;
    //cout << "nsegy " << nsegy << endl;
    while (itr < INT_MAX) {
        tapesegy tapetrace;
        segy trace;
        file.read((char *) &tapetrace, nsegy); // reading a single trace
        if (!file) { // error in reading (reaching the end of the file)
            break;
        }
        tapesegy_to_segy(&tapetrace, &trace); // converts to easily readable format
        // (oblivious to the type int/short)
        if (endian == 0)
            for (int i = 0; i < SEGY_NKEYS; ++i)
                swaphval(&trace, i); // swaps the bytes if the machine is little endian
        if (bh.hns != trace.ns)
            //  dout("warning::the number of samples in the binary header is " <<
            //  bh.hns << " and in the trace is " << trace.ns);
            itr++; // increase the number of read traces;
        /*  if (itr== 1 ){
             cout << "trace length is " << trace.ns << endl ;

                   ofstream file;
                   file.open("bintracenew", std::ofstream::out);
                   file.write((char*) &trace.data, trace.ns*4);
                   file.close();

           }*/
        switch (bh.format) {
            case 1:
                /* Convert IBM floats to native floats */
                ibm_to_float((int *) trace.data, (int *) trace.data, bh.hns, endian);
                break;
            case 2:
                /* Convert 4 byte integers to native floats */
                long_to_float((long *) trace.data, (float *) trace.data, bh.hns, endian);
                break;
            case 3:
                /* Convert 2 byte integers to native floats */
                short_to_float((short *) trace.data, (float *) trace.data, bh.hns, endian);
                break;
            case 5:
                /* IEEE floats.  Byte swap if necessary. */
                if (endian == 0)
                    for (int i = 0; i < bh.hns; ++i)
                        swap_float_4(&trace.data[i]);
                break;
            case 8:
                /* Convert 1 byte integers to native floats */
                integer1_to_float((signed char *) trace.data, (float *) trace.data, bh.hns);
                break;
        }

        /* Apply trace weighting. */
        int trcwt = (bh.format == 1 || bh.format == 5) ? 0 : 1;
        if (trcwt && trace.trwf != 0) {
            float scale = pow(2.0, -trace.trwf);
            for (int i = 0; i < bh.hns; ++i) {
                trace.data[i] *= scale;
            }
        }

        trace.ns = bh.hns;
        shot_ids.insert(trace.fldr);
        //        cout << "trace ensemble number " << trace.fldr << endl;
        if (check_func(
                &trace,
                check_elements)) { // check_func is a callback function that decides
            // whether to add this trace or not
            //            cout << "new element is pushed" << endl;
            traces.push_back(trace);
        }
        //        cout << "reading sx = "<< trace.sx <<endl;
        //        exit(EXIT_FAILURE);
    }

    file.close();
    //cout << "finished reading successfully" << endl;
    //    ofstream file0;
    //    file0.open("bindump0", std::ofstream::out);
    //    file0.write((char*) &bh, sizeof(bhed));
    //    file0.close();
}

void SUSegy::WriteHeadersAndTraces(string filename) {

    ofstream fileW;
    tapebhed tapebh;
    tapesegy tapetr;

    memset((void *) &tapebh, 0, BNYBYTES);
    memset((void *) &tapetr, 0, sizeof(tapesegy));

    fileW.open(filename.c_str(), std::ofstream::out);
    fileW.write(ebcdictextheader, EBCBYTES); // writing the text header

    /** writing the binary header */
    bh.format = 1; ///< the format in which the segy file is written

    switch (bh.format) {
        case 8:
            nsegy = bh.hns + SEGY_HDRBYTES;
            break;
        case 3:
            nsegy = bh.hns * 2 + SEGY_HDRBYTES;
            break;
        case 1:
        case 2:
        case 5:
        default:
            nsegy = bh.hns * 4 + SEGY_HDRBYTES;
    }

    /* if little endian machine, swap bytes in binary header */
    union {
        short s;
        char c[2];
    } testend; // testing if the system is little or big endian
    testend.s = 1;
    endian = (testend.c[0] == '\0') ? 1 : 0;
    if (endian == 0)
        for (int i = 0; i < BHED_NKEYS; ++i)
            swapbhval(&bh, i);
    bhed_to_tapebhed(&bh, &tapebh);     ///< convert from longs/shorts to bytes
    fileW.write((char *) &bh, BNYBYTES); // writing the text header

    /** Copy traces from the class to the file */
    int itr = 0;

    for (vector<segy>::iterator tr = traces.begin(); tr != traces.end();
         tr++, itr++) {

        segy *trace = &*tr;
        unsigned short ns = trace->ns;

        if (endian == 0) {
            for (int i = 0; i < SEGY_NKEYS; ++i)
                swaphval(trace, i); ///< swaps the bytes if the machine is little endian
        };
        /** Set/check trace header words */
        if (trace->ns != bh.hns && le32toh(trace->ns) != bh.hns) {
            cout << "ERROR:: conflict (note: the values may be swapped due to "
                    "endianness): trace.ns = "
                 << trace->ns << endl;
        }
        /* Convert and write desired traces */
        /* Convert internal floats to IBM floats */
        //        float_to_ibm((int *) trace->data, (int *) trace->data, ns,
        //        endian);
        unsigned char
                dataibm[SU_NFLTS * sizeof(float)]; /**< pseudo_float data[SU_NFLTS]; */
        for (int i = 0; i < ns; i++) {
            if (trace->data[i] == 0.00) { // changed; each trace is located on a row
                trace->data[i] = 0.000000;
            }
        }

        ieee2ibm(trace->data, trace->data, ns);

        /* Convert from longs/shorts to bytes*/
        segy_to_tapesegy(trace, &tapetr, nsegy);
        //        memset(tapetr.unass, 0, 60);
        /* Write the trace to tape */
        // if(itr == 1000) {cout << "value in tapetr "<< *(float*)&tapetr.data[1000]
        // << endl;

        //                          cout << "value intrace" << trace->data[1000] <<
        //                          endl;}

        fileW.write((char *) &tapetr, nsegy);
    }

    //  cout << "finished writing" << endl;
}

float *SUSegy::Arrange(float **ptr_base, char *name) {
    float *ptr = (float *) malloc(bh.hns * traces.size() * sizeof(float));
    int j = 0;
    for (auto const &trace : traces) {
        for (int i = 0; i < bh.hns; i++) {
            ptr[i * traces.size() + j] = trace.data[i];
        }
        j++;
    }
    return ptr;
}

SUSegy::SUSegy() {

    //    size_t nsegy;
    //    bool endian;

    //    vector<segy> traces; // contains the traces (headers + data) of that
    //    file
    //

    memset(ebcdictextheader, 0, EBCBYTES + 1);
    memset(asciitextheader, 0, EBCBYTES + 1);
    memset(binaryheader, 0, BNYBYTES + 1);

    memset(&bh, 0, sizeof(bh));
    memset(ebcdictextheader, 0, EBCBYTES + 1);
    memset(asciitextheader, 0, EBCBYTES + 1);

    memset(&nextended, 0, sizeof(short));

    nsegy = 0;
    endian = false;
}

SUSegy::~SUSegy() {
    file.close();
    // cout << __FUNCTION__ << endl;
    // fflush(stdout);
}

SUSegy::SUSegy(string filename) {

    memset(ebcdictextheader, 0, EBCBYTES + 1);
    memset(asciitextheader, 0, EBCBYTES + 1);
    memset(binaryheader, 0, BNYBYTES + 1);

    memset(&bh, 0, sizeof(bh));
    memset(ebcdictextheader, 0, EBCBYTES + 1);
    memset(asciitextheader, 0, EBCBYTES + 1);
    memset((char *) &nextended, 0, sizeof(short));

    ReadHeadersAndTraces(filename, NULL, suselect::all);
}
