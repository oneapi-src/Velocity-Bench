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


#include "CommandLineParser.h"
#include "Utilities.h"

#include <iomanip>

#define CPP_MODULE "CMDL"
#include "Logging.h"

void VelocityBench::CommandLineParser::AddSetting(std::string const &sSetting, 
                                   std::string const &sDescription, 
                                   bool        const  bRequired)
{
    AddSetting(sSetting, sDescription, bRequired, "", InputType_t::NONE, 0);
}

void VelocityBench::CommandLineParser::AddSetting(std::string  const &sSetting, 
                                   std::string  const &sDescription, 
                                   bool         const  bRequired, 
                                   std::string  const &sDefaultValue, 
                                   InputType_t  const  InputType, 
                                   unsigned int const  uiNumberOfSettings)
{
    if (sSetting.front() != '-') {
        LOG_ERROR("Invalid setting " << sSetting << ". Should contain a '-' or '--' at front");
    }

    if (bRequired && !sDefaultValue.empty()) {
        LOG_ERROR("Required setting " << sSetting << " should not have a default value");
    }

    std::map<std::string, ArgumentSetting>::const_iterator itFind(m_mapSettings2ArgumentSettings.find(sSetting));
    if (itFind != m_mapSettings2ArgumentSettings.end()) {
        LOG_ERROR("Setting " << sSetting << " was already registered");
    }
    m_mapSettings2ArgumentSettings.insert({sSetting, ArgumentSetting(sDescription, uiNumberOfSettings, bRequired, InputType, sDefaultValue)}); 
}

bool VelocityBench::CommandLineParser::IsSet(std::string const &sSetting) const
{
    std::map<std::string, ArgumentSetting>::const_iterator itFind(m_mapSettings2ArgumentSettings.find(sSetting));
    if (itFind == m_mapSettings2ArgumentSettings.end())
        LOG_ERROR(sSetting << " was not registered!");
    return itFind->second.m_bWasSet;
}

bool VelocityBench::CommandLineParser::AreRequiredSettingsGiven() const
{
    for (auto const &ArgumentArgumentSetting : m_mapSettings2ArgumentSettings) {
        if (!ArgumentArgumentSetting.second.m_bWasSet && ArgumentArgumentSetting.second.m_bRequired) {
            LOG_ERROR("Required setting " << ArgumentArgumentSetting.first << " was not received!");
        }
    }
    return true;
}

bool VelocityBench::CommandLineParser::ExtractSettingValues(std::vector<std::string> const &vArguments, 
                                             unsigned int             const  uiArgumentSettingStartPosition, 
                                             std::string              const &sSetting, 
                                             ArgumentSetting                &CurrentArgumentSetting)
{
    CurrentArgumentSetting.m_vRetrievedArgumentSettings.clear(); // Clear any default values that were stored

    if (uiArgumentSettingStartPosition + CurrentArgumentSetting.m_uiNumberOfArgumentSettings > vArguments.size()) { // Check if we are within the vector's range before operand extraction
        LOG_ERROR("Insufficient number of operands required for " << sSetting);
    }

    for (unsigned int uiCurrentArgumentSetting = uiArgumentSettingStartPosition; uiCurrentArgumentSetting < uiArgumentSettingStartPosition + CurrentArgumentSetting.m_uiNumberOfArgumentSettings; ++uiCurrentArgumentSetting) {
        if (CurrentArgumentSetting.m_InputType == InputType_t::STRING && vArguments[uiCurrentArgumentSetting].front() == '-') { // ArgumentSettings should not start with '-' 
            LOG_ERROR("Not enough operands for setting " << sSetting);
        }
        //std::cout << sSetting << " obtained value \'" << vArguments[uiCurrentArgumentSetting] << "\'" << std::endl;
        CurrentArgumentSetting.m_vRetrievedArgumentSettings.emplace_back(vArguments[uiCurrentArgumentSetting]);
    }

    //std::cout << "vector size : " << CurrentArgumentSetting.m_vRetrievedArgumentSettings.size() << " expected : " << CurrentArgumentSetting.m_uiNumberOfArgumentSettings << std::endl;
    if (CurrentArgumentSetting.m_vRetrievedArgumentSettings.size() != CurrentArgumentSetting.m_uiNumberOfArgumentSettings) {
        LOG_ERROR("Number of operands parsed does not match expected for setting " << sSetting);
    }
    return true;
}

std::vector<std::string> VelocityBench::CommandLineParser::ConvertArgArrayToVector(int const uiNumberOfArguments, char const *cArguments[]) const
{
    std::vector<std::string> vParsedInputs(uiNumberOfArguments - 1, "");
    for (int uiArgument = 1; uiArgument < uiNumberOfArguments; ++uiArgument) {
        std::string const sArgument(cArguments[uiArgument]);
        vParsedInputs[uiArgument - 1] = sArgument;
    }
    return vParsedInputs;
}

std::vector<std::string> VelocityBench::CommandLineParser::SearchAndGetSetting(std::string const &sSetting, bool const bCheckOnlyOneArgumentSetting) const 
{
    std::map<std::string, ArgumentSetting>::const_iterator itSetting(m_mapSettings2ArgumentSettings.find(sSetting)); 
    if (itSetting == m_mapSettings2ArgumentSettings.end())
        throw std::runtime_error("Unable to obtain operand for setting " + sSetting + " because it is not registered in the command line arguments");

    if (itSetting->second.m_uiNumberOfArgumentSettings == 0)
        throw std::runtime_error("Arguments were not expected for setting " + sSetting);
    
    if (!itSetting->second.m_bWasSet && itSetting->second.m_vRetrievedArgumentSettings.empty())
        throw std::runtime_error("Argument " + sSetting + " does not have a default value to retrieve");

    if (bCheckOnlyOneArgumentSetting && itSetting->second.m_vRetrievedArgumentSettings.size() > 1)
        throw std::runtime_error("There are more than one operands for setting " + sSetting);

    return itSetting->second.m_vRetrievedArgumentSettings;
}

std::string VelocityBench::CommandLineParser::SearchAndGetSetting(std::string const &sSetting) const 
{ 
    return SearchAndGetSetting(sSetting, true).front();
}

bool VelocityBench::CommandLineParser::ValidateIntegerInput(std::string const &sSettingToCheck, bool const bNegativeNumberAllowed) const
{
    std::string const sInputToCheck(SearchAndGetSetting(sSettingToCheck));
    if (!bNegativeNumberAllowed && sInputToCheck.front() == '-')
        LOG_ERROR("Cannot convert " << sInputToCheck << " for setting " << sSettingToCheck << " because it does not allow for negative inputs");

    if (!Utility::IsInteger(sInputToCheck))
        LOG_ERROR("Cannot convert " << sInputToCheck << " for setting " << sSettingToCheck << " because it is not a number");

    return true;
}

int VelocityBench::CommandLineParser::GetIntegerSetting(std::string const &sSetting) const
{
    if (!ValidateIntegerInput(sSetting, true /* Can be negative */))
        LOG_ERROR("Unable to validate input setting " << sSetting);
    return std::stoi(SearchAndGetSetting(sSetting)); // can throw
}

unsigned int VelocityBench::CommandLineParser::GetUIntegerSetting(std::string const &sSetting) const
{
    if (!ValidateIntegerInput(sSetting, false /* Must be positive */))
        LOG_ERROR("Unable to validate input setting " << sSetting);
    return std::stoul(SearchAndGetSetting(sSetting), nullptr, 0);
}

double VelocityBench::CommandLineParser::GetDoubleSetting(std::string const &sSetting) const
{
    std::string::size_type sSize_t;
    return std::stod(SearchAndGetSetting(sSetting), &sSize_t /* Alias for size_t type */); // can throw
}

std::string VelocityBench::CommandLineParser::GetSetting(std::string const &sSetting) const
{
    return SearchAndGetSetting(sSetting);
}

std::vector<std::string> VelocityBench::CommandLineParser::GetArgumentSettings(std::string const &sSetting) const
{
    return SearchAndGetSetting(sSetting, false /* No one operand check */); 
}

bool VelocityBench::CommandLineParser::Parse(int const argc, char const *argv[])
{
    std::vector<std::string> const vParsedInputs(ConvertArgArrayToVector(argc, argv));
    unsigned int             const uiNumberOfArguments(vParsedInputs.size());
    unsigned int                   uiCurrentArgument(0);
    std::vector<std::string>       vUnknownInputs;
    do {
        if (vParsedInputs.empty())
            break;

        std::string const sCurrentArgument(vParsedInputs[uiCurrentArgument]); // Convenience
        //LOG("uiCurrentArgument : " << uiCurrentArgument << " Arg : " << sCurrentArgument); 
        if (sCurrentArgument.front() != '-') { // It is assumed that a setting starts with a dash or double dash
            vUnknownInputs.emplace_back(sCurrentArgument);
        } else {
            
            if (sCurrentArgument == "-h") { // Gracefully exit when the user looks for the help message
                DisplayUsage();
                exit(EXIT_SUCCESS);
            }

            std::map<std::string, ArgumentSetting>::iterator itArgumentSetting(m_mapSettings2ArgumentSettings.find(sCurrentArgument));
            if (itArgumentSetting == m_mapSettings2ArgumentSettings.end()) {
                LOG_ERROR("Unknown setting " << sCurrentArgument);
            }

            if (itArgumentSetting->second.m_bWasSet) {
                LOG_ERROR("Multiple settings received for " << sCurrentArgument);
            }
            itArgumentSetting->second.m_bWasSet = true; // Make sure only one setting was used 
            
            if (itArgumentSetting->second.m_uiNumberOfArgumentSettings != 0) { 
                if (!ExtractSettingValues(vParsedInputs, uiCurrentArgument + 1 /* beginning of operand setting */, sCurrentArgument, itArgumentSetting->second))
                    return false;
                uiCurrentArgument = uiCurrentArgument + itArgumentSetting->second.m_uiNumberOfArgumentSettings; // Move the current argument counter to the very last operand setting
            }
        }
        ++uiCurrentArgument; // move along ...
    } while (uiCurrentArgument < uiNumberOfArguments);

    if (!vUnknownInputs.empty()) {
        std::string sUnknownInputs;
        for (auto const &sElem : vUnknownInputs)
            sUnknownInputs += sElem + " ";
        LOG_ERROR("Received unknown inputs " << sUnknownInputs);
    }

    if (!AreRequiredSettingsGiven()) {
        LOG("Insufficient number of arguments");
        DisplayUsage();
        return false;
    }

    return true;
}

void VelocityBench::CommandLineParser::DisplayUsage()
{
    unsigned int uiLongestSetting(0), uiLongestDescription(0);
    for (auto const &Setting2ArgumentSetting : m_mapSettings2ArgumentSettings) {
        if (Setting2ArgumentSetting.first.length() > uiLongestSetting)
            uiLongestSetting = Setting2ArgumentSetting.first.length();

        if (Setting2ArgumentSetting.second.m_sDescription.length() > uiLongestDescription)
            uiLongestDescription = Setting2ArgumentSetting.second.m_sDescription.length();

    }
    LOG("Available input settings"); 
    LOG("------------------------");
    for (auto const &Setting2ArgumentSetting : m_mapSettings2ArgumentSettings) {
        LOG(std::left << std::setw(uiLongestSetting + 1) <<  Setting2ArgumentSetting.first << ": " << std::left << std::setw(uiLongestSetting + uiLongestDescription + 1) << Setting2ArgumentSetting.second.m_sDescription << (Setting2ArgumentSetting.second.m_bRequired ? " (REQUIRED)" : " (OPTIONAL)"));// << std::endl;
    }
    LOG("------------------------");
}

