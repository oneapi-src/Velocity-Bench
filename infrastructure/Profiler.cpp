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


#include "Profiler.h"
#include "Utilities.h"

#define CPP_MODULE "PROF"
#include "Logging.h"

#include <iomanip>

Profiler::Profiler(bool const bShowInactiveTimers)
    : m_mapTimerName2Timer()
    , m_vTimerNames()
    , m_bDisplayInactiveTimers(bShowInactiveTimers)
{
    // LOG("Profiler created. Inactive timers are " << (m_bDisplayInactiveTimers ? "shown" : "not shown"));  // This causes crashing in CUDA for some reason
}

Profiler::Profiler() : Profiler{false} {}

Profiler::~Profiler(){}

void Profiler::AddMultipleTimers(std::string const &sTimerNames)
{
    std::vector<std::string> const vTimerNames(Utility::TokenizeString(sTimerNames, ','));
    for (auto const &sTimerName : vTimerNames)
        AddTimer(sTimerName, true);
    LOG("Added multiple timers " << sTimerNames);
}

void Profiler::AddTimer(std::string const &sTimerName)
{
    AddTimer(sTimerName, false);
}

void Profiler::AddTimer(std::string const &sTimerName, bool const bSupressAddedMessage)
{
    std::map<std::string, Timer>::const_iterator itcFoundTimer(m_mapTimerName2Timer.find(sTimerName));
    if (itcFoundTimer != m_mapTimerName2Timer.end())
        LOG_ERROR("Timer name " << sTimerName << " was already included");
    m_mapTimerName2Timer.insert({sTimerName, Timer(sTimerName)});
    m_vTimerNames.emplace_back(sTimerName);
    
    if (!bSupressAddedMessage)
        LOG("Added timer " << sTimerName);
}

void Profiler::AccumulateTimer(std::string const &sTimerName, Timer const &RHS)
{
    std::map<std::string, Timer>::iterator itFoundTimer(m_mapTimerName2Timer.find(sTimerName));
    if (itFoundTimer == m_mapTimerName2Timer.end())
        LOG_ERROR("Unable to find timer " << sTimerName << " to accumulate");
    itFoundTimer->second += RHS;
}

void Profiler::AccumulateTimer(std::string const &sTimerName, std::chrono::steady_clock::duration const &RHS)
{
    std::map<std::string, Timer>::iterator itFoundTimer(m_mapTimerName2Timer.find(sTimerName));
    if (itFoundTimer == m_mapTimerName2Timer.end())
        LOG_ERROR("Unable to find timer " << sTimerName << " to accumulate");
    itFoundTimer->second += RHS;
}

void Profiler::StartTimer(std::string const &sTimerName)
{
    std::map<std::string, Timer>::iterator itFoundTimer(m_mapTimerName2Timer.find(sTimerName));
    if (itFoundTimer == m_mapTimerName2Timer.end())
        LOG_ERROR("Unable to find timer " << sTimerName << " to start");
    itFoundTimer->second.Start();
}

void Profiler::StopTimer(std::string const &sTimerName)
{
    std::map<std::string, Timer>::iterator itFoundTimer(m_mapTimerName2Timer.find(sTimerName));
    if (itFoundTimer == m_mapTimerName2Timer.end())
        LOG_ERROR("Unable to find timer " << sTimerName << " to stop");
    itFoundTimer->second.Stop();
}

void Profiler::PrintTimingStats()
{
    unsigned int uiLongestTimerName(16); // " Timing results " is length 16
    for (auto const &TimerName : m_vTimerNames)
        if (TimerName.length() > uiLongestTimerName)
            uiLongestTimerName = TimerName.length();

    LOG("*********************************");
    LOG(std::left << std::setw(uiLongestTimerName + 1) << " Timing Results | Elapsed time" << (m_bDisplayInactiveTimers ? " (N/A - denotes inactive)" : ""));
    LOG("*********************************");
    for (auto const &TimerName : m_vTimerNames) {
        std::map<std::string, Timer>::const_iterator itcFoundTimer(m_mapTimerName2Timer.find(TimerName));
        if (itcFoundTimer->second.WasNeverActive() && !m_bDisplayInactiveTimers)
            continue;
        LOG(std::setw(uiLongestTimerName) << itcFoundTimer->second.GetTimerName() << "| " <<  (!itcFoundTimer->second.WasNeverActive() ? itcFoundTimer->second.GetTimeAsString() : "N/A"));
    }
    LOG("*********************************");
}
