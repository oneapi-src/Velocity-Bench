/*
Modifications Copyright (C) 2023 Intel Corporation

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


SPDX-License-Identifier: BSD-3-Clause
*/

/*
Copyright 2019 Advanced Micro Devices

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <sycl/sycl.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include "utils.hh"
#include "Parameters.hh"
#include "utilsMpi.hh"
#include "MonteCarlo.hh"
#include "initMC.hh"
#include "Tallies.hh"
#include "PopulationControl.hh"
#include "ParticleVaultContainer.hh"
#include "ParticleVault.hh"
#include "MC_Particle_Buffer.hh"
#include "MC_Processor_Info.hh"
#include "MC_Time_Info.hh"
#include "macros.hh"
#include "MC_Fast_Timer.hh"
#include "MC_SourceNow.hh"
#include "SendQueue.hh"
#include "NVTX_Range.hh"
#include "cudaUtils.hh"
#include "cudaFunctions.hh"
#include "qs_assert.hh"
#include "CycleTracking.hh"
#include "CoralBenchmark.hh"
#include "EnergySpectrum.hh"

#include "git_hash.hh"
#include "git_vers.hh"

#ifdef HAVE_MPI
#include <mpi.h>
#endif
#include "utilsMpi.hh"

#define SIMD_WIDTH 16

#ifdef HAVE_SYCL
sycl::property_list q_prop{sycl::ext::oneapi::property::queue::discard_events(), sycl::property::queue::in_order()};
sycl::queue sycl_device_queue(q_prop);
#endif

void gameOver();
void cycleInit(bool loadBalance);
void cycleTracking(MonteCarlo *monteCarlo, uint64_cu *, uint64_cu *);
void cycleFinalize();

inline void copyParticleVault_h2d(ParticleVault_d* P_d, ParticleVault* P)
{
    int _particlesSize = P->size();
    int _particlesCap = P->_particles._capacity;
    MC_Base_Particle* _particles_d = sycl::malloc_device<MC_Base_Particle>(_particlesCap, sycl_device_queue);
    if(_particlesSize ==  0) {    
	    sycl_device_queue.memset(&(P_d->_particlesSize), 0, sizeof(int));
	   sycl_device_queue.memset(_particles_d, 0, _particlesCap*sizeof(MC_Base_Particle));
    }
    else{
       sycl_device_queue.memcpy(_particles_d, P->_particles._data, _particlesCap*sizeof(MC_Base_Particle));
        sycl_device_queue.memcpy(&(P_d->_particlesSize), &(P->_particles._size), sizeof(int));
    }
    sycl_device_queue.memcpy(&(P_d->_particles), &_particles_d, sizeof(MC_Base_Particle*));
}

void copyParticleVault_d2h(ParticleVault* P, ParticleVault_d* P_d)
{
    int _particlesCap = P->_particles._capacity;
    sycl_device_queue.memcpy(&(P->_particles._size), &(P_d->_particlesSize), sizeof(int));
    MC_Base_Particle* _particles_add;
    sycl_device_queue.memcpy(&_particles_add, &(P_d->_particles), sizeof(MC_Base_Particle*));
    sycl_device_queue.memcpy(P->_particles._data, _particles_add, _particlesCap*sizeof(MC_Base_Particle)).wait();
    sycl::free(_particles_add, sycl_device_queue);
}

void copySendQueueDevice(SendQueue* sq, SendQueue_d* sq_d)
{
    int _dataSize = sq->size();
    sendQueueTuple* _data_d = sycl::malloc_device<sendQueueTuple>(_dataSize, sycl_device_queue);
    sycl_device_queue.memcpy(_data_d, sq->get_data_pointer(), _dataSize*sizeof(sendQueueTuple));
    sycl_device_queue.memcpy(&(sq_d->_dataSize), &(sq->_data._size), sizeof(int));
    sycl_device_queue.memcpy(&(sq_d->_data), &(_data_d), sizeof(int));
}

void copySendQueueHost(SendQueue_d* sq_d, SendQueue* sq)
{
    sendQueueTuple* _data;
    sycl_device_queue.memcpy(&_data, &(sq_d->_data), sizeof(sendQueueTuple*)).wait();
    int _dataSize = sq->_data.size();
    sycl_device_queue.memcpy(sq->get_data_pointer(), _data, _dataSize*sizeof(sendQueueTuple)).wait();
    sycl::free(_data, sycl_device_queue);
}

void copyMonteCarloDevice_first(MonteCarlo* mc, MonteCarlo_d* mc_d)
{
    Tallies_d *tallies_d = sycl::malloc_device<Tallies_d>(1, sycl_device_queue);
    sycl_device_queue.memcpy(&(mc_d->_tallies_d), &(tallies_d), sizeof(Tallies_d*));


    sycl_device_queue.memcpy(&(tallies_d->_num_balance_replications), &(mc->_tallies->_num_balance_replications), sizeof(int));
    sycl_device_queue.memcpy(&(tallies_d->_num_flux_replications), &(mc->_tallies->_num_flux_replications), sizeof(int));
    sycl_device_queue.memcpy(&(tallies_d->_num_cellTally_replications), &(mc->_tallies->_num_cellTally_replications), sizeof(int));

    int _scalarFluxDomainSize = mc->_tallies->_scalarFluxDomain.size();
    ScalarFluxDomain_d *_scalarFluxDomain_d = sycl::malloc_device<ScalarFluxDomain_d>(_scalarFluxDomainSize, sycl_device_queue);
    sycl_device_queue.memcpy(&(tallies_d->_scalarFluxDomainSize), &(_scalarFluxDomainSize), sizeof(int));
    sycl_device_queue.memcpy(&(tallies_d->_scalarFluxDomain), &(_scalarFluxDomain_d), sizeof(ScalarFluxDomain_d*));

    for(int i=0; i<_scalarFluxDomainSize; i++)
    {
        int _taskSize = mc->_tallies->_scalarFluxDomain[i]._task.size();
        ScalarFluxTask_d *_task = sycl::malloc_device<ScalarFluxTask_d>(_taskSize,sycl_device_queue);
        sycl_device_queue.memcpy(&(_scalarFluxDomain_d[i]._taskSize), &(_taskSize), sizeof(int));
        sycl_device_queue.memcpy(&(_scalarFluxDomain_d[i]._task), &(_task), sizeof(ScalarFluxTask_d*));
        for(int j=0; j<_taskSize; j++)
        {
            int _cellSize = mc->_tallies->_scalarFluxDomain[i]._task[j]._cell.size();
            ScalarFluxCell *_cell = sycl::malloc_device<ScalarFluxCell>(_cellSize, sycl_device_queue);
            sycl_device_queue.memcpy(&(_task[j]._cellSize), &(_cellSize), sizeof(int));
            sycl_device_queue.memcpy(&(_task[j]._cell), &(_cell), sizeof(ScalarFluxCell*));
            for(int k=0; k<_cellSize; k++)
            {
                int _size = mc->_tallies->_scalarFluxDomain[i]._task[j]._cell[k].size();
                double *_group = sycl::malloc_device<double>(_size, sycl_device_queue);
                sycl_device_queue.memcpy(_group, mc->_tallies->_scalarFluxDomain[i]._task[j]._cell[k]._group, _size*sizeof(double));
                sycl_device_queue.memcpy(&(_cell[k]._size), &(_size), sizeof(int));
                sycl_device_queue.memcpy(&(_cell[k]._group), &(_group), sizeof(double *));
            }
        }
    }
}

void copyMonteCarloDevice_part(MonteCarlo* mc, MonteCarlo_d* mc_d)
{
    sycl_device_queue.memcpy(&(mc_d->domain_d), &(mc->domain_d), sizeof(MC_Domain_d*));
    sycl_device_queue.memcpy(&(mc_d->_material_d), &(mc->_material_d), sizeof(Material_d*));
    sycl_device_queue.memcpy(&(mc_d->_nuclearData_d), &(mc->_nuclearData_d), sizeof(NuclearData_d*));

    MC_Time_Info* time_info_d = sycl::malloc_device<MC_Time_Info>(1, sycl_device_queue);
    sycl_device_queue.memcpy(time_info_d, mc->time_info, sizeof(MC_Time_Info));
    sycl_device_queue.memcpy(&(mc_d->time_info_d), &(time_info_d), sizeof(MC_Time_Info*));

    ParticleVaultContainer_d* _particleVaultContainer_d = sycl::malloc_device<ParticleVaultContainer_d>(1, sycl_device_queue);
    sycl_device_queue.memcpy(&(mc_d->_particleVaultContainer_d), &(_particleVaultContainer_d), sizeof(ParticleVaultContainer_d*));
    sycl_device_queue.memcpy(&(_particleVaultContainer_d->_vaultSize), &(mc->_particleVaultContainer->_vaultSize), sizeof(uint64_t));
    sycl_device_queue.memcpy(&(_particleVaultContainer_d->_numExtraVaults), &(mc->_particleVaultContainer->_numExtraVaults), sizeof(uint64_t));
    sycl_device_queue.memcpy(&(_particleVaultContainer_d->_extraVaultIndex), &(mc->_particleVaultContainer->_extraVaultIndex), sizeof(uint64_cu));
    int _extraVaultSize = mc->_particleVaultContainer->_extraVault.size();
    sycl_device_queue.memcpy(&(_particleVaultContainer_d->_extraVaultSize), &(_extraVaultSize), sizeof(int));
    ParticleVault_d ** _extraVault=sycl::malloc_device<ParticleVault_d*>(_extraVaultSize, sycl_device_queue);
    sycl_device_queue.memcpy(&(_particleVaultContainer_d->_extraVault), &(_extraVault), sizeof(ParticleVault**));
    for(int i=0;i<_extraVaultSize;i++)
    {
	    ParticleVault_d *tmp = sycl::malloc_device<ParticleVault_d>(1, sycl_device_queue);
	    copyParticleVault_h2d(tmp, mc->_particleVaultContainer->_extraVault[i]);
	    sycl_device_queue.memcpy(&(_extraVault[i]), &(tmp), sizeof(ParticleVault*));
    }
}

void copyMonteCarloHost_part(MonteCarlo_d* mc_d, MonteCarlo* mc)
{
    MC_Time_Info *tmp;
    sycl_device_queue.memcpy(&tmp, &(mc_d->time_info_d), sizeof(MC_Time_Info*)).wait();
    sycl_device_queue.memcpy(mc->time_info, tmp, sizeof(MC_Time_Info)).wait();
    sycl::free(tmp,sycl_device_queue);

    ParticleVaultContainer_d* tmp_p;
    sycl_device_queue.memcpy(&tmp_p, &(mc_d->_particleVaultContainer_d), sizeof(ParticleVaultContainer_d*)).wait();
    sycl_device_queue.memcpy(&(mc->_particleVaultContainer->_vaultSize), &(tmp_p->_vaultSize), sizeof(uint64_t)).wait();
    sycl_device_queue.memcpy(&(mc->_particleVaultContainer->_numExtraVaults), &(tmp_p->_numExtraVaults), sizeof(uint64_t)).wait();   
    sycl_device_queue.memcpy(&(mc->_particleVaultContainer->_extraVaultIndex), &(tmp_p->_extraVaultIndex), sizeof(uint64_cu)).wait();

    ParticleVault_d **tmp_ev;
    sycl_device_queue.memcpy(&(mc->_particleVaultContainer->_extraVault._size), &(tmp_p->_extraVaultSize), sizeof(int)).wait();
    sycl_device_queue.memcpy(&tmp_ev, &(tmp_p->_extraVault), sizeof(ParticleVault_d **)).wait();
    for(int i=0;i<mc->_particleVaultContainer->_extraVault.size();i++)
    {
	ParticleVault_d *tmp_0;
	sycl_device_queue.memcpy(&tmp_0, &(tmp_ev[i]), sizeof(ParticleVault_d *)).wait();
        copyParticleVault_d2h(mc->_particleVaultContainer->_extraVault[i], tmp_0);
	sycl::free(tmp_0, sycl_device_queue);
    }
    sycl::free(tmp_ev, sycl_device_queue);
    sycl::free(tmp_p, sycl_device_queue);
}

void copyMonteCarloHost_last(MonteCarlo_d* mc_d, MonteCarlo* mc)
{
    Tallies_d *tmp_t;
    sycl_device_queue.memcpy(&tmp_t, &(mc_d->_tallies_d), sizeof(Tallies_d*)).wait();
    ScalarFluxDomain_d* tmp_s;
    sycl_device_queue.memcpy(&(tmp_s), &(tmp_t->_scalarFluxDomain), sizeof(ScalarFluxDomain_d*)).wait();
    int _scalarFluxDomainSize = mc->_tallies->_scalarFluxDomain.size();
    for(int i=0;i<_scalarFluxDomainSize;i++)
    {
        ScalarFluxTask_d* tmp_task;
        sycl_device_queue.memcpy(&(tmp_task), &(tmp_s[i]._task), sizeof(ScalarFluxTask_d*)).wait();
        int _taskSize = mc->_tallies->_scalarFluxDomain[i]._task.size();
        for(int j=0; j<_taskSize; j++)
        {
            ScalarFluxCell* tmp_cell;
            sycl_device_queue.memcpy(&(tmp_cell), &(tmp_task[j]._cell), sizeof(ScalarFluxCell*)).wait();
            int _cellSize = mc->_tallies->_scalarFluxDomain[i]._task[j]._cell.size();
            for(int k=0; k<_cellSize; k++)
            {
                double *tmp_d;
                sycl_device_queue.memcpy(&(tmp_d), &(tmp_cell[k]._group), sizeof(double*)).wait();
                int _size = mc->_tallies->_scalarFluxDomain[i]._task[j]._cell[k].size();
                sycl_device_queue.memcpy(mc->_tallies->_scalarFluxDomain[i]._task[j]._cell[k]._group, tmp_d, _size*sizeof(double)).wait();
                sycl::free(tmp_d, sycl_device_queue);
            }
            sycl::free(tmp_cell, sycl_device_queue);
        }
        sycl::free(tmp_task, sycl_device_queue);
    }
    sycl::free(tmp_s, sycl_device_queue);
    sycl::free(tmp_t, sycl_device_queue);
}

void setGPU()
{

    int rank;
    MPI_Comm comm_mc_world(MPI_COMM_WORLD);

    int Ngpus = 1;

    mpiComm_rank(comm_mc_world, &rank);
    int GPUID = rank % Ngpus;
}

using namespace std;

MonteCarlo *mcco = NULL;

int main(int argc, char **argv)
{
    mpiInit(&argc, &argv);
    printBanner(GIT_VERS, GIT_HASH);
    std::cout << "Loading params" << std::endl;
    Parameters params = getParameters(argc, argv);
    const string &filename = params.simulationParams.inputFile;
    ifstream inp_file(filename.c_str());
    if (!inp_file.good())
    {
        return -1;
    }
    std::cout << "Finished loading params" << std::endl;
    printParameters(params, cout);
    std::cout << "setting GPU" << std::endl;
    setGPU();

#ifdef HAVE_SYCL
    // HOIST INTO SETUP FUNCTION EVENTUALLY
    char *devchar = std::getenv("QS_DEVICE");
    std::string devname = (devchar == NULL ? "None" : devchar);
    if (devname == "CPU")
    {
        sycl_device_queue = sycl::cpu_selector{};
    }
    else if (devname == "GPU")
    {
        sycl_device_queue = sycl::gpu_selector{};
    }
    else
    {
        std::cout << "QS_DEVICE must be CPU, GPU or HOST" << std::endl;
        std::abort();
    }
#endif

    // mcco stores just about everything.
    std::cout << "setting parameters" << std::endl;
    mcco = initMC(params);

    int myRank, nRanks;
    mpiComm_rank(MPI_COMM_WORLD, &myRank);

    std::cout << "Started copyMaterialDatabase_device" << std::endl;
    copyMaterialDatabase_device(mcco);
    std::cout << "Finished copyMaterialDatabase_device" << std::endl;
    copyNuclearData_device(mcco->_nuclearData, mcco->_nuclearData_d);
    std::cout << "Finished copyNuclearData_device" << std::endl;
    copyDomainDevice(mcco->_nuclearData->_numEnergyGroups, mcco->domain, mcco->domain_d, mcco->domainSize);
    std::cout << "Finished copyDomainDevice" << std::endl;

    mpiBarrier(MPI_COMM_WORLD);
    int loadBalance = params.simulationParams.loadBalance;

    MC_FASTTIMER_START(MC_Fast_Timer::main); // this can be done once mcco exist.

    const int nSteps = params.simulationParams.nSteps;

    // allocate arrays to hold counters in pinned memory on the host and on the device.
    int replications = mcco->_tallies->GetNumBalanceReplications();
    uint64_cu *tallies;
    tallies = (uint64_cu *)sycl::malloc_host(sizeof(uint64_cu) * NUM_TALLIES *
                                                 replications,
                                             sycl_device_queue);
    uint64_cu *tallies_d;
    tallies_d = (uint64_cu *)sycl::malloc_device(sizeof(uint64_cu) *
                                                     NUM_TALLIES * replications,
                                                 sycl_device_queue);
    for (int il = 0; il < replications; il++)
    {
        for (int j1 = 0; j1 < NUM_TALLIES; j1++)
        {
            tallies[NUM_TALLIES * il + j1] = 0;
        }
    }
    sycl_device_queue
        .memcpy(tallies_d, tallies,
                sizeof(uint64_cu) * NUM_TALLIES * replications)
        .wait();
    for (int ii = 0; ii < nSteps; ++ii)
    {
        cycleInit(bool(loadBalance));      
        cycleTracking(mcco, tallies, tallies_d);
        cycleFinalize();

        mcco->fast_timer->Last_Cycle_Report(
            params.simulationParams.cycleTimers,
            mcco->processor_info->rank,
            mcco->processor_info->num_processors,
            mcco->processor_info->comm_mc_world);
    }

    MC_FASTTIMER_STOP(MC_Fast_Timer::main);

    gameOver();

    coralBenchmarkCorrectness(mcco, params);

    sycl::free(tallies, sycl_device_queue);
    sycl::free(tallies_d, sycl_device_queue);

#ifdef HAVE_UVM
    mcco->~MonteCarlo();
    gpuFree(mcco);
#else
    delete mcco;
#endif

    mpiFinalize();

    return 0;
}

void gameOver()
{
    mcco->fast_timer->Cumulative_Report(mcco->processor_info->rank,
                                        mcco->processor_info->num_processors,
                                        mcco->processor_info->comm_mc_world,
                                        mcco->_tallies->_balanceCumulative._numSegments);
    mcco->_tallies->_spectrum.PrintSpectrum(mcco);
}

void cycleInit(bool loadBalance)
{
    MC_FASTTIMER_START(MC_Fast_Timer::cycleInit);
    mcco->clearCrossSectionCache();
    mcco->_tallies->CycleInitialize(mcco);
    mcco->_particleVaultContainer->swapProcessingProcessedVaults();
    mcco->_particleVaultContainer->collapseProcessed();
    mcco->_particleVaultContainer->collapseProcessing();

    mcco->_tallies->_balanceTask[0]._start = mcco->_particleVaultContainer->sizeProcessing();

    mcco->particle_buffer->Initialize();

    MC_SourceNow(mcco);

    PopulationControl(mcco, loadBalance); // controls particle population

    RouletteLowWeightParticles(mcco); // Delete particles with low statistical weight

    MC_FASTTIMER_STOP(MC_Fast_Timer::cycleInit);
}

#if defined(HAVE_SYCL)

void CycleTrackingKernel(MonteCarlo_d *monteCarlo, int num_particles, ParticleVault_d *processingVault, ParticleVault_d *processedVault, uint64_cu *tallies,
                         sycl::nd_item<3> item_ct1,
                         uint8_t *local)
{
    int global_index = getGlobalThreadID(item_ct1);
    int local_index = getLocalThreadID(item_ct1);
    int replications = monteCarlo->_tallies_d->GetNumBalanceReplications();

    auto values_l = (int *)local;
    if (local_index < replications * NUM_TALLIES)
    {
        values_l[local_index] = 0;
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

    if (global_index < num_particles)
    {
        CycleTrackingGuts(monteCarlo, global_index, processingVault, processedVault, &values_l[0]);
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);
    if (local_index < replications * NUM_TALLIES)
    {
#if defined(HAVE_SYCL)
        ATOMIC_ADD(tallies[local_index], (uint64_cu)values_l[local_index]);
#else
        __atomic_fetch_add(&(tallies[local_index]), (uint64_t)values_l[local_index], __ATOMIC_RELAXED);
#endif
    }
}
#endif

void cycleTracking(MonteCarlo *monteCarlo, uint64_cu *tallies, uint64_cu *tallies_d)
{
    MC_FASTTIMER_START(MC_Fast_Timer::cycleTracking);

    bool done = false;

    // Determine whether or not to use GPUs if they are available (set for each MPI rank)
    ExecutionPolicy execPolicy = getExecutionPolicy(monteCarlo->processor_info->use_gpu);

    ParticleVaultContainer &my_particle_vault = *(monteCarlo->_particleVaultContainer);

    // Post Inital Receives for Particle Buffer
    monteCarlo->particle_buffer->Post_Receive_Particle_Buffer(my_particle_vault.getVaultSize());

    // Get Test For Done Method (Blocking or non-blocking
    MC_New_Test_Done_Method::Enum new_test_done_method = monteCarlo->particle_buffer->new_test_done_method;

    int l5 = 0;

    const int replications = monteCarlo->_tallies->GetNumBalanceReplications();
    MonteCarlo_d *monteCarlo_d = sycl::malloc_device<MonteCarlo_d>(1, sycl_device_queue);
    copyMonteCarloDevice_first(monteCarlo, monteCarlo_d);
    do
    {
        int particle_count = 0; // Initialize count of num_particles processed

        while (!done)
        {
            uint64_t fill_vault = 0;

            for (uint64_t processing_vault = 0; processing_vault < my_particle_vault.processingSize(); processing_vault++)
            {
//                MC_FASTTIMER_START(MC_Fast_Timer::cycleTracking_Kernel);
                uint64_t processed_vault = my_particle_vault.getFirstEmptyProcessedVault();

                ParticleVault *processingVault = my_particle_vault.getTaskProcessingVault(processing_vault);
                ParticleVault *processedVault = my_particle_vault.getTaskProcessedVault(processed_vault);	

                int numParticles = processingVault->size();

                if (numParticles != 0)
                {
                    NVTX_Range trackingKernel("cycleTracking_TrackingKernel"); // range ends at end of scope

                    // The tracking kernel can run
                    // * As a cuda kernel
                    // * As an OpenMP 4.5 parallel loop on the GPU
                    // * As an OpenMP 3.0 parallel loop on the CPU
                    // * AS a single thread on the CPU.
                    
                    switch (execPolicy)
                    {
                    case gpuWithCUDA:
                    {
#if defined(HAVE_SYCL)

                        const size_t N = numParticles;
                        unsigned int wg_size = 256;
                        unsigned int num_wgs = (N + wg_size - 1) / wg_size;

                ParticleVault_d *processingVault_d = sycl::malloc_device<ParticleVault_d>(1, sycl_device_queue);
                ParticleVault_d *processedVault_d = sycl::malloc_device<ParticleVault_d>(1, sycl_device_queue);

                copyParticleVault_h2d(processingVault_d, processingVault);
                copyParticleVault_h2d(processedVault_d, processedVault);
                copyMonteCarloDevice_part(monteCarlo, monteCarlo_d);
                sycl_device_queue.wait();

                MC_FASTTIMER_START(MC_Fast_Timer::cycleTracking_Kernel);		

                        sycl_device_queue.submit([&](sycl::handler &cgh)
                                                 {              
             sycl::accessor<uint8_t, 1, sycl::access_mode::read_write, sycl::access::target::local>
              local_acc_ct1(
                  sycl::range<1>(NUM_TALLIES * replications * sizeof(int)),
                  cgh);

              cgh.parallel_for(
                  sycl::nd_range<3>(sycl::range<3>(1, 1, num_wgs) *
                                        sycl::range<3>(1, 1, wg_size),
                                    sycl::range<3>(1, 1, wg_size)),
                  [=](sycl::nd_item<3> item_ct1)
#if !defined(USE_NVIDIA_BACKEND) && !defined(USE_AMDHIP_BACKEND)
		  [[intel::reqd_sub_group_size(SIMD_WIDTH)]]
#endif
		  {
                    CycleTrackingKernel(monteCarlo_d, numParticles,
                                        processingVault_d, processedVault_d,
                                        tallies_d, item_ct1,
                                        local_acc_ct1.get_pointer());                   
                  }); })
                            .wait();
                            
                  MC_FASTTIMER_STOP(MC_Fast_Timer::cycleTracking_Kernel);

                        sycl_device_queue
                            .memcpy(tallies, tallies_d, NUM_TALLIES * sizeof(uint64_cu) * replications)
			    .wait();
                  
            copyParticleVault_d2h(processingVault, processingVault_d);
	        copyParticleVault_d2h(processedVault, processedVault_d);
            copyMonteCarloHost_part(monteCarlo_d, monteCarlo);
            
			sycl::free(processingVault_d, sycl_device_queue);
			sycl::free(processedVault_d, sycl_device_queue);            
#endif                  
                    }
                    break;

                    case gpuWithOpenMP:
                    {

                        std::cout << " this isn't supported with hip yet " << std::endl;
                    }
                    break;

                   /* case cpu:
#include "mc_omp_parallel_for_schedule_static.hh"
                        for (int particle_index = 0; particle_index < numParticles; particle_index++)
                        {
                            CycleTrackingGuts(monteCarlo, particle_index, processingVault, processedVault, (int *)tallies);
                        }
                        break;
                    */
                    default:
                        qs_assert(false);

                    } // end switch

                    // Add in counters from GPU kernel
                    for (int il = 0; il < replications; il++)
                    {
                        monteCarlo->_tallies->_balanceTask[il]._numSegments += tallies[NUM_TALLIES * il + 0];
                        tallies[NUM_TALLIES * il + 0] = 0;
                        monteCarlo->_tallies->_balanceTask[il]._escape += tallies[NUM_TALLIES * il + 1];
                        tallies[NUM_TALLIES * il + 1] = 0;
                        monteCarlo->_tallies->_balanceTask[il]._census += tallies[NUM_TALLIES * il + 2];
                        tallies[NUM_TALLIES * il + 2] = 0;
                        monteCarlo->_tallies->_balanceTask[il]._collision += tallies[NUM_TALLIES * il + 3];
                        tallies[NUM_TALLIES * il + 3] = 0;
                        monteCarlo->_tallies->_balanceTask[il]._scatter += tallies[NUM_TALLIES * il + 4];
                        tallies[NUM_TALLIES * il + 4] = 0;
                        monteCarlo->_tallies->_balanceTask[il]._absorb += tallies[NUM_TALLIES * il + 5];
                        tallies[NUM_TALLIES * il + 5] = 0;
                        monteCarlo->_tallies->_balanceTask[il]._fission += tallies[NUM_TALLIES * il + 6];
                        tallies[NUM_TALLIES * il + 6] = 0;
                        monteCarlo->_tallies->_balanceTask[il]._produce += tallies[NUM_TALLIES * il + 7];
                        tallies[NUM_TALLIES * il + 7] = 0;
                    }
                    sycl_device_queue
                        .memcpy(tallies_d, tallies,
                                sizeof(uint64_cu) * NUM_TALLIES * replications)
                        .wait();
		}
                particle_count += numParticles;
//                MC_FASTTIMER_STOP(MC_Fast_Timer::cycleTracking_Kernel);

                MC_FASTTIMER_START(MC_Fast_Timer::cycleTracking_MPI);

                // Next, communicate particles that have crossed onto
                // other MPI ranks.
                NVTX_Range cleanAndComm("cycleTracking_clean_and_comm");

                SendQueue &sendQueue = *(my_particle_vault.getSendQueue());
                monteCarlo->particle_buffer->Allocate_Send_Buffer(sendQueue);

                // Move particles from send queue to the send buffers

                for (int index = 0; index < sendQueue.size(); index++)
                {
                    sendQueueTuple &sendQueueT = sendQueue.getTuple(index);
                    MC_Base_Particle mcb_particle;

                    processingVault->getBaseParticleComm(mcb_particle, sendQueueT._particleIndex);

                    int buffer = monteCarlo->particle_buffer->Choose_Buffer(sendQueueT._neighbor);
                    if (buffer >= 0)
                        monteCarlo->particle_buffer->Buffer_Particle(mcb_particle, buffer);
                }

                monteCarlo->particle_buffer->Send_Particle_Buffers(); // post MPI sends

                processingVault->clear(); // remove the invalid particles
                sendQueue.clear();

                // Move particles in "extra" vaults into the regular vaults.
                my_particle_vault.cleanExtraVaults();

                // receive any particles that have arrived from other ranks
                monteCarlo->particle_buffer->Receive_Particle_Buffers(fill_vault);

                MC_FASTTIMER_STOP(MC_Fast_Timer::cycleTracking_MPI);

            } // for loop on vaults

            MC_FASTTIMER_START(MC_Fast_Timer::cycleTracking_MPI);

            NVTX_Range collapseRange("cycleTracking_Collapse_ProcessingandProcessed");
            my_particle_vault.collapseProcessing();
            my_particle_vault.collapseProcessed();
            collapseRange.endRange();

            // Test for done - blocking on all MPI ranks
            NVTX_Range doneRange("cycleTracking_Test_Done_New");
            done = monteCarlo->particle_buffer->Test_Done_New(new_test_done_method);
            doneRange.endRange();

            MC_FASTTIMER_STOP(MC_Fast_Timer::cycleTracking_MPI);

        } // while not done: Test_Done_New()

        // Everything should be done normally.
        done = monteCarlo->particle_buffer->Test_Done_New(MC_New_Test_Done_Method::Blocking);
        if(done)
        {
            copyMonteCarloHost_last(monteCarlo_d, monteCarlo);
            sycl::free(monteCarlo_d, sycl_device_queue);
        }

    } while (!done);

    // Make sure to cancel all pending receive requests
    monteCarlo->particle_buffer->Cancel_Receive_Buffer_Requests();
    // Make sure Buffers Memory is Free
    monteCarlo->particle_buffer->Free_Buffers();

    MC_FASTTIMER_STOP(MC_Fast_Timer::cycleTracking);
}

void cycleFinalize()
{
    MC_FASTTIMER_START(MC_Fast_Timer::cycleFinalize);

    mcco->_tallies->_balanceTask[0]._end = mcco->_particleVaultContainer->sizeProcessed();

    // Update the cumulative tally data.
    mcco->_tallies->CycleFinalize(mcco);

    mcco->time_info->cycle++;

    mcco->particle_buffer->Free_Memory();

    MC_FASTTIMER_STOP(MC_Fast_Timer::cycleFinalize);
}
