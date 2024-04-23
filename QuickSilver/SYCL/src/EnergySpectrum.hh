#ifndef ENERGYSPECTRUM_HH
#define ENERGYSPECTRUM_HH
#include <string>
#include <vector>
#include <cstdint>

class MonteCarlo;

class EnergySpectrum
{
public:
    EnergySpectrum(std::string name, uint64_t size) : _fileName(std::move(name)), _censusEnergySpectrum(size, 0){};
    void UpdateSpectrum(MonteCarlo *monteCarlo);
    void PrintSpectrum(MonteCarlo *monteCarlo);

private:
    std::string _fileName;
    std::vector<uint64_t> _censusEnergySpectrum;
};
//------------------------------------------------------------------------------------------------------------
class EnergySpectrum_d
{
public:
    EnergySpectrum_d(std::string name, uint64_t size) : _fileName(std::move(name))/*, _censusEnergySpectrum(size, 0)*/{
        _censusEnergySpectrum = new uint64_t[size];
    };
    void UpdateSpectrum(MonteCarlo *monteCarlo);
    void PrintSpectrum(MonteCarlo *monteCarlo);

private:
    std::string _fileName;
    //std::vector<uint64_t> _censusEnergySpectrum;
    uint64_t* _censusEnergySpectrum;
    int _censusEnergySpectrumSize;
};

#endif
