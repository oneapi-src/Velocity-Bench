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

#include "TestBenchBase.h"

#define CPP_MODULE "TBEN"
#include "Logging.h"

TestBenchBase::TestBenchBase()
    : m_cmdLineParser()
{
    m_cmdLineParser.AddSetting("-n",  "Number of Iterations", false, "1", CommandLineParser::InputType_t::INTEGER, 1 /* One operand */);

    // Flags
    m_cmdLineParser.AddSetting("--noverification", "Disables verification", false);

    // FFT specific settings
    m_cmdLineParser.AddSetting("-i",           "Input FFT data file",  true,  "",         CommandLineParser::InputType_t::STRING, 1);
    m_cmdLineParser.AddSetting("-t",           "Acceptable tolerance", false, "0.000001", CommandLineParser::InputType_t::DOUBLE, 1); 
    m_cmdLineParser.AddSetting("--nofastfail", "Disable Fast fail",    false);
	m_cmdLineParser.AddSetting("-s",           "support threshold",    false, "0.02", CommandLineParser::InputType_t::DOUBLE, 1);
}

unsigned int TestBenchBase::GetNumberOfIterations() const
{
    int const iNumberOfIterations(m_cmdLineParser.GetIntegerSetting("-n"));
    if (iNumberOfIterations < 1) {
        LOG_ERROR("Invalid input for number of iterations " << iNumberOfIterations);
    }
    return static_cast<unsigned int>(iNumberOfIterations);
}

bool TestBenchBase::ParseCommandLineArguments(int const argc, char const *argv[])
{
    if (!m_cmdLineParser.Parse(argc, argv)) {
        LOG_ERROR("Failed to parse command line arguments");
    }

    return true;
}
