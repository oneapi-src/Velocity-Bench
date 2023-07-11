/*
 * 
 * Modifications Copyright (C) 2023 Intel Corporation
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * 
 * of this software and associated documentation files (the "Software"),
 * 
 * to deal in the Software without restriction, including without limitation
 * 
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * 
 * and/or sell copies of the Software, and to permit persons to whom
 * 
 * the Software is furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included
 * 
 * in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * 
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * 
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * 
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
 * 
 * OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * 
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
 * 
 * OR OTHER DEALINGS IN THE SOFTWARE.
 * 
 * SPDX-License-Identifier: MIT
 */


#ifndef PROFILER
#define PROFILER

#include "Timer.h"

#include <string>
#include <map>
#include <vector>

class Profiler
{
private:
    std::map<std::string, Timer>       m_mapTimerName2Timer;
    std::vector<std::string>           m_vTimerNames;
    bool                         const m_bDisplayInactiveTimers;

public:
    Profiler           (Profiler const &RHS)  = delete;
    Profiler           (Profiler const &&RHS) = delete;
    Profiler &operator=(Profiler const &RHS)  = delete;

    explicit Profiler(bool const bShowInactiveTimers);
    Profiler();
   ~Profiler();

    void AddTimer         (std::string const &sTimerName);
    void AddTimer         (std::string const &sTimerName, bool const bSupressAddedMessage);
    void AddMultipleTimers(std::string const &sCommaSeparatedNames);

    void AccumulateTimer(std::string const &sTimerName, Timer const &RHS); 
    void AccumulateTimer(std::string const &sTimerName, std::chrono::steady_clock::duration const &RHS); 

    void StartTimer(std::string const &sTimerName);
    void StopTimer (std::string const &sTimerName);

    void PrintTimingStats();
};

#endif
