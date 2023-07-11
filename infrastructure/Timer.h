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


#ifndef TIMER_H
#define TIMER_H

#include <chrono>
#include <string>

class Timer
{
public:
    enum class Units { SECONDS, MILLISECONDS, MICROSECONDS, NANOSECONDS };
private:
    enum class TimerState { ZERO, IDLE, RUNNING }; // ZERO: Timer was created, never started, IDLE: Timer has a duration, RUNNING: self explanatory...

    std::string const m_sTimerName;
    TimerState        m_TimerState;
    
    std::chrono::steady_clock::time_point m_tStart;
    std::chrono::steady_clock::duration   m_tTotalDuration;
public:
    Timer(std::string const &sTimerName);
   ~Timer();

    // Note: Operators below only add durations
    Timer            (Timer const &RHS) = default; 
    Timer &operator= (Timer const &RHS);
    Timer  operator+ (Timer const &RHS);
    Timer  operator- (Timer const &RHS);
    Timer &operator+=(Timer const &RHS);

    Timer &operator+=(std::chrono::steady_clock::duration const &RHSDuration);


    void Start();
    void Stop ();
    void Reset();

    double GetTime()                     const;  // Defaults to seconds
    double GetTime(Units const TimeUnit) const;

    bool        WasNeverActive() const { return m_TimerState == TimerState::ZERO; }
    std::string GetTimerName()   const { return m_sTimerName; } 
    
    unsigned long GetTimeInNanoSecs() const;

    std::string GetTimeAsString(Units const TimeUnit) const;
    std::string GetTimeAsString()                     const; 
};

#endif
