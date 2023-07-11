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


#ifndef COMMAND_LINE_PARSER_H
#define COMMAND_LINE_PARSER_H

#include <map>
#include <string>
#include <vector>

namespace VelocityBench{ // SS: tmp fix to avoid name collision with cv namespace

class CommandLineParser
{
public:
    enum class InputType_t { STRING, INTEGER, DOUBLE, NONE };
private:
    struct ArgumentSetting {
        // These const members should not change throughout the life of the struct
        std::string        const m_sDescription;
        unsigned int       const m_uiNumberOfArgumentSettings;
        bool               const m_bRequired;
        InputType_t        const m_InputType;
        bool                     m_bWasSet;
       std::vector<std::string> m_vRetrievedArgumentSettings;

        ArgumentSetting(std::string const &sDescription, unsigned int const uiNumberOfArgumentSettings, bool const bRequired, InputType_t const InputType, std::string const &sDefaultValue)
            : m_sDescription              (sDescription)
            , m_uiNumberOfArgumentSettings(uiNumberOfArgumentSettings)
            , m_bRequired                 (bRequired)
            , m_InputType                 (InputType)
            , m_bWasSet                   (false)
            , m_vRetrievedArgumentSettings()
        {
            m_vRetrievedArgumentSettings.emplace_back(sDefaultValue);
        }
    };

    std::map<std::string, ArgumentSetting> m_mapSettings2ArgumentSettings;
    std::map<std::string, std::string>     m_mapDescription2Commands;
private:
    std::string              SearchAndGetSetting(std::string const &sSetting) const; 
    std::vector<std::string> SearchAndGetSetting(std::string const &sSetting, bool const bCheckOnlyOneArgumentSetting) const; 
    std::vector<std::string> ConvertArgArrayToVector(int const uiNumberOfArguments, char const *cArguments[]) const;
    bool                     AreRequiredSettingsGiven() const;
    bool                     ExtractSettingValues(std::vector<std::string> const &vArguments, unsigned int const uiArgumentSettingStartPosition, std::string const &sSetting, ArgumentSetting &CurrentArgumentSetting);
    bool                     ValidateIntegerInput(std::string const &sInputToCheck, bool const bNegativeNumberAllowed) const;
public:
    CommandLineParser           (CommandLineParser const &RHS) = delete;
    CommandLineParser &operator=(CommandLineParser const &RHS) = delete;

    CommandLineParser() : m_mapSettings2ArgumentSettings(), m_mapDescription2Commands() { AddSetting("-h", "Displays help message", false); }
   ~CommandLineParser(){}

    void AddSetting(std::string const &sSetting, std::string const &sDescription, bool const bRequired);
    void AddSetting(std::string const &sSetting, std::string const &sDescription, bool const bRequired, std::string const &sDefaultValue, InputType_t const InputType, unsigned int const uiNumberOfArgumentSettings);

    std::string              GetSetting         (std::string const &sSetting) const;
    int                      GetIntegerSetting  (std::string const &sSetting) const;
    unsigned int             GetUIntegerSetting (std::string const &sSetting) const;
    double                   GetDoubleSetting   (std::string const &sSetting) const;
    std::vector<std::string> GetArgumentSettings(std::string const &sSetting) const;

    bool IsSet(std::string const &sSetting) const;
    bool Parse(int const argc, char const *argv[]);

    void DisplayUsage();
};

}// namespace VelocityBench

#endif


