/*
Copyright 2019 Advanced Micro Devices

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


#ifndef MONTECARLO_HH
#define MONTECARLO_HH

#include "QS_Vector.hh"
#include "MC_Domain.hh"
#include "MC_Location.hh"
#include "Parameters.hh"

class MC_RNG_State;
class NuclearData;
class NuclearData_d;
class MaterialDatabase;
class ParticleVaultContainer;
class ParticleVaultContainer_d;
class Tallies;
class Tallies_d;
class MC_Processor_Info;
class MC_Time_Info;
class MC_Particle_Buffer;
class MC_Fast_Timer_Container;
class MC_Domain;
class Material_d;

void copyParticleVaultContainerH2D(ParticleVaultContainer* pvc, ParticleVaultContainer_d* pvc_d);
void copyParticleVaultContainerD2H(ParticleVaultContainer_d* pvc_d, ParticleVaultContainer* pvc);
void copyTalliesDevice(Tallies* tallies, Tallies_d* tallies_d);
void copyTalliesHost(Tallies_d* tallies_d, Tallies* tallies);
void copyMC_Time_InfoDevice(MC_Time_Info* mc_time_info, MC_Time_Info* mc_time_info_d);
void copyMC_Time_InfoHost(MC_Time_Info* mc_time_info_d, MC_Time_Info* mc_time_info);
void copyMonteCarloDevice(MonteCarlo* mc, MonteCarlo_d* mc_d);
void copyMonteCarloHost(MonteCarlo_d* mc_d, MonteCarlo* mc);
class MonteCarlo
{
public:

   MonteCarlo(const Parameters& params);
   ~MonteCarlo();

public:

   void clearCrossSectionCache();

   qs_vector<MC_Domain> domain;
   MC_Domain_d * domain_d;
   int domainSize;

    Parameters _params;
    NuclearData* _nuclearData;
    ParticleVaultContainer* _particleVaultContainer;
    MaterialDatabase* _materialDatabase;
    Tallies *_tallies;
    MC_Time_Info *time_info;
    MC_Fast_Timer_Container *fast_timer;
    MC_Processor_Info *processor_info;
    MC_Particle_Buffer *particle_buffer;
    Material_d * _material_d;
    NuclearData_d* _nuclearData_d;

    double source_particle_weight;

private:
   // Disable copy constructor and assignment operator
   MonteCarlo(const MonteCarlo&);
   MonteCarlo& operator=(const MonteCarlo&);
};

class MonteCarlo_d
{
public:
   MC_Domain_d * domain_d;
   ParticleVaultContainer_d* _particleVaultContainer_d;
   Tallies_d *_tallies_d; 
   MC_Time_Info *time_info_d; 
   Material_d * _material_d;
   NuclearData_d* _nuclearData_d;
};
#endif
