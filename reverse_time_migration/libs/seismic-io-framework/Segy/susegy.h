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


#ifndef SUSEGY_H
#define SUSEGY_H

#include <bitset>
#include <endian.h>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>

#include <limits.h>
#include <math.h>

#include "segyelement.h"
#include "suheaders.h"

#include "segy_helpers.h"

namespace suselect {
    struct compare {
        bool operator()(const int &lhs, const int &rhs) const { return lhs < rhs; }
    };

    bool ifequal(segy *trace, std::vector<SEGYelement> *check_elements);

    bool all(segy *trace, std::vector<SEGYelement> *check_elements);

    bool first(segy *trace, std::vector<SEGYelement> *check_elements);
} // namespace suselect

class SUSegy {
private:
    std::vector<char *>
            extendedebcdictextheader; // there is a possibility for an extended header
    // after the binary header
    std::vector<char *> extendedasciitextheader;
    char binaryheader[BNYBYTES + 1];

protected:
    std::ifstream file;
    size_t nsegy;
    bool endian;

public:
    char ebcdictextheader[EBCBYTES + 1]; // called ebcbuf in SeisUnix; reads in
    // ebcidic format
    char asciitextheader[EBCBYTES + 1];  // converted text header to ascii format
    short nextended;                     // the number of extended text headers

    std::vector<segy> traces; // contains the traces (headers + data) of that file

    std::set<int, suselect::compare>
            shot_ids; // the id number of all shots in the file

    bhed bh; // binary header

    SUSegy();

    SUSegy(std::string filename);

    ~SUSegy();

    void print_extended_header() {
        for (auto header : extendedasciitextheader)
            std::cout << "header: \n" << header << std::endl;
    }

    size_t getNsegy() { return nsegy; }

    void Ascii2ebc(unsigned char *s) {
        while (*s) {
            *s = a2e[(int) (*s)];
            s++;
        }
    }

    void setasciitextheader(char textheader[EBCBYTES + 1]) {
        memset(ebcdictextheader, 0, EBCBYTES + 1);
        memset(asciitextheader, 0, EBCBYTES + 1);
        mempcpy(asciitextheader, textheader, EBCBYTES);
        mempcpy(ebcdictextheader, textheader, EBCBYTES);
        Ascii2ebc((unsigned char *) ebcdictextheader);
    }

    int ntraces() { return this->traces.size(); }

    void clear() { // empties the susegy from any previous data that may exists
        memset(ebcdictextheader, 0, (EBCBYTES + 1) * sizeof(char));
        memset(asciitextheader, 0, (EBCBYTES + 1) * sizeof(char));
        memset(binaryheader, 0, (BNYBYTES + 1) * sizeof(char));
        nextended = 0;
        traces.clear();
        for (auto str : extendedasciitextheader)
            delete str;
        for (auto str : extendedebcdictextheader)
            delete str;
        extendedasciitextheader.clear();
        extendedebcdictextheader.clear();
        if (file.is_open()) {
            file.clear();
            file.close();
        }
        nsegy = 0;
    }

    template<typename MEMBER>
    void print(MEMBER member) {
        int cnt = 0;
        int min = INT_MAX;
        int max = 0;
        std::cout << "the number of elements " << this->traces.size() << std::endl;
        for (segy trace : this->traces) {
            //    std::cout << "counter = " << cnt << ", " << trace.*member <<
            //    std::endl;
            cnt++;
            if (trace.*member > max)
                max = trace.*member;
            if (trace.*member < min)
                min = trace.*member;
            // if (trace.*member == 2503 ) continue;
            // break
        }
        std::cout << "min is : " << min << " , max is " << max << std::endl;
        std::cout << "total count " << cnt << std::endl;
    }

    std::vector<uint>
    GetUniqueOccurences(std::string filename, std::vector<SEGYelement> *select_element, uint min_threshold, uint max_threshold);

    void ReadHeadersAndTraces(
            std::string const &filename, std::vector<SEGYelement> *check_elements,
            bool (*check_func)(
                    segy *trace,
                    std::vector<SEGYelement> *check_elements)); // reads a segy file
    void WriteHeadersAndTraces(
            std::string filename); // writes the segy data into a segy file
    void ReadBinaryHeader(
            std::string const &filename, bool exit_on_error = true); // reads the binary header and store it in bh
    float *Arrange(float **ptr_base, char *name = NULL);

    // other functions:

    /*

    void ReadTracesHeader(string file_name,string sort_type, );

    vector <general_traces> ReadTraces(string file_name, string sort_type,);

    void WriteBinaryHeaderToFile(string file_name, bhead * Bh);

    void WriteTracesHeaderToFile(string file_name, string sort_type,  ) ; //
    traces heades structure ??

    void WriteTracesDataToFile(string file_name , string sort_type , float *
    traces_data);

    */
};

#endif // SUSEGY_H
