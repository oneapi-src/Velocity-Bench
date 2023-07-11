/*
MIT License

Copyright (c) 2015 University of West Bohemia

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

/*
MIT License

Modifications Copyright (C) 2023 Intel Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

SPDX-License-Identifier: MIT License
*/

#ifndef __MY_STOP_WATCH_H_INCLUDED__
#define __MY_STOP_WATCH_H_INCLUDED__


#include <inttypes.h>

//usefull definitions for simple use:
#define MYSTOPWATCH_START MyStopWatch cl; cl.start();
#define MYSTOPWATCH_STOP_AND_PRINT cl.stop(); printf("Elapsed time: %f s\n", cl.getTime());
#define MYSTOPWATCH_STOP cl.stop();
#define MYSTOPWATCH_PRINT printf("Elapsed time: %f s\n", cl.getTime());

#ifdef _MSC_VER 
#include <Windows.h>
class WindowsStopWatch;

typedef WindowsStopWatch MyStopWatch;

class WindowsStopWatch {
    unsigned nofCycles;
    uint64_t timeAccum;
    
    LARGE_INTEGER curStart;
    LARGE_INTEGER curStop;

    static inline uint64_t clockResulution() {
            LARGE_INTEGER foo;
            QueryPerformanceFrequency(&foo);
            return foo.QuadPart;
    };
    
    static inline uint64_t diffFileTime(LARGE_INTEGER &a, LARGE_INTEGER &b) 
    {
        return b.QuadPart - a.QuadPart;
    };

public:
    WindowsStopWatch() : nofCycles(0), timeAccum(0UL) {};
    
    inline void start() 
    { 
        QueryPerformanceCounter(&curStart);
    };

    inline void stop() 
    { 
        QueryPerformanceCounter(&curStop);
        timeAccum += diffFileTime(curStart, curStop);
        nofCycles +=1;
    };

    inline void stopstart() 
    { 
        QueryPerformanceCounter(&curStop);
        timeAccum += diffFileTime(curStart, curStop);
        nofCycles +=1;
        curStart = curStop;
    };

    inline void reset() 
    {
        nofCycles = 0;
        timeAccum = 0;
    };

    inline const float getTime() const
    {
        return (float) timeAccum / (float)clockResulution();
    }
    
    inline const uint64_t getTicks() const
    {
        return (uint64_t)timeAccum;
    }

    inline const float avgTime() const
    {
        return getTime() / (float)nofCycles;
    }

    inline float getCurrentTime()
    {
    QueryPerformanceCounter(&curStop);
        return (float)((double)diffFileTime(curStart, curStop) / clockResulution());
    }
};
#else
#include <sys/time.h>
class PosixStopWatch;

typedef PosixStopWatch MyStopWatch;


class PosixStopWatch {
    unsigned nofCycles;
    uint64_t timeAccum;
    
    timeval curStop;
    timeval curStart;

    static inline uint64_t clockResulution() {
        static const unsigned usec_per_sec = 1000000;
        return usec_per_sec;
    };

    static inline void getClocks(timeval &time) {
        gettimeofday(&time, NULL);
    };
    
    static inline uint64_t timeDiff(timeval start, timeval stop) {
        uint64_t ticks1 = start.tv_sec * clockResulution() + start.tv_usec;
        uint64_t ticks2 = stop.tv_sec * clockResulution() + stop.tv_usec;
        
        return ticks2-ticks1;
    };

public:
    PosixStopWatch() : nofCycles(0), timeAccum(0UL) {};
    
    inline void start() 
    { 
        getClocks(curStart);
    };

    inline void stop() 
    { 
        getClocks(curStop);

        timeAccum += timeDiff(curStart, curStop);
        nofCycles +=1;
    };

    inline void reset() 
    {
        nofCycles = 0;
        timeAccum = 0;
    };

    inline const float getTime() const
    {    
        return (float) timeAccum / (float)clockResulution();
    }
    
    inline const uint64_t getTicks() const
    {
            return (uint64_t)timeAccum;
    }

    inline const float avgTime() const
    {
        return getTime() / (float)nofCycles;
    }

    inline float getCurrentTime()
    {
        getClocks(curStop);
        return (float)((double)timeDiff(curStart, curStop) / clockResulution());
    }
};

#endif
#endif
