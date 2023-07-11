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



#include "Timer.h"
#include <iostream>

#define CPP_MODULE "TIMR"
#include "Logging.h"

Timer::Timer(std::string const &sTimerName)
    : m_sTimerName(sTimerName)
    , m_TimerState(TimerState::ZERO)
    , m_tStart    ()
    , m_tStop     ()
{
    Reset();
}

Timer::~Timer() 
{
    if (m_TimerState == TimerState::RUNNING) {
        LOG_WARNING("Timer " << m_sTimerName << " is still running!");
    }
}

void Timer::Start()
{
    if (m_TimerState == TimerState::RUNNING) {
        LOG_WARNING("Calling Start() on Timer " << m_sTimerName << " that is already running!");
        return;
    }

    if (m_TimerState == TimerState::IDLE) {
        LOG_WARNING("Overwriting previous timer data on " << m_sTimerName << " timer");
    }

    m_TimerState = TimerState::RUNNING;
    m_tStart     = std::chrono::steady_clock::now();
}

void Timer::Stop()
{
    if (m_TimerState == TimerState::IDLE) {
        LOG_WARNING("Calling Stop() on Timer " << m_sTimerName << " that was not running!");
        return;
    }

    if (m_TimerState == TimerState::ZERO) {
        return;
    }

    m_tStop      = std::chrono::steady_clock::now();
    m_TimerState = TimerState::IDLE;
}

void Timer::Reset() 
{ 
    m_TimerState = TimerState::ZERO; 
    m_tStart     = {};
    m_tStop      = {};
}

double Timer::GetTime() const
{
    return GetTime(Units::SECONDS);
}

double Timer::GetTime(Units const TimeUnit) const
{
    LOG_ASSERT(m_TimerState != TimerState::RUNNING, "Calling GetTime() while timer is still running");
    //std::chrono::system_clock::duration tDuration = (m_tStop - m_tStart);
    switch (TimeUnit) {
        case Units::MILLISECONDS: return std::chrono::duration<double, std::milli>(m_tStop - m_tStart).count();
        case Units::MICROSECONDS: return std::chrono::duration<double, std::micro>(m_tStop - m_tStart).count();
        case Units::NANOSECONDS:  return std::chrono::duration<double, std::nano> (m_tStop - m_tStart).count();
        default: /* SECONDS */    return std::chrono::duration<double>(m_tStop - m_tStart).count();
    }
}

std::string Timer::GetTimeAsString() const
{
    return std::to_string(GetTime()) + "s";
}

std::string Timer::GetTimeAsString(unsigned int iterations) const
{
    return std::to_string(GetTime()/iterations) + "s";
}

std::string Timer::GetTimeAsString(Units const TimeUnit) const
{
    switch (TimeUnit) {
        case Units::MILLISECONDS: return std::to_string(GetTime(TimeUnit)) + "ms"; 
        case Units::MICROSECONDS: return std::to_string(GetTime(TimeUnit)) + "us"; 
        case Units::NANOSECONDS:  return std::to_string(GetTime(TimeUnit)) + "ns"; 
        default: return std::to_string(GetTime(TimeUnit)) + "s"; /* SECONDS */    
    }
}
