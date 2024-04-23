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

#include <cuda_profiler_api.h>
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

void gameOver();
void cycleInit(bool loadBalance);
void cycleTracking(MonteCarlo *monteCarlo, uint64_cu *, uint64_cu *);
void cycleFinalize();

inline void copyParticleVault_h2d(ParticleVault_d* P_d, ParticleVault* P)
{
    int _particlesSize = P->size();
    int _particlesCap = P->_particles._capacity;
    //MC_Base_Particle* _particles_d = sycl::malloc_device<MC_Base_Particle>(_particlesCap, sycl_device_queue);
    MC_Base_Particle* _particles_d;
    safeCall(cudaMalloc((void **)&_particles_d, sizeof(MC_Base_Particle) * _particlesCap));
    if(_particlesSize ==  0) {    
	    //sycl_device_queue.memset(&(P_d->_particlesSize), 0, sizeof(int));
        safeCall(cudaMemset(&(P_d->_particlesSize), 0, sizeof(int)));
	   //sycl_device_queue.memset(_particles_d, 0, _particlesCap*sizeof(MC_Base_Particle));
       safeCall(cudaMemset((_particles_d), 0, _particlesCap*sizeof(MC_Base_Particle)));
    }
    else{
       //sycl_device_queue.memcpy(_particles_d, P->_particles._data, _particlesCap*sizeof(MC_Base_Particle));
       safeCall(cudaMemcpy(_particles_d, P->_particles._data, sizeof(MC_Base_Particle) * _particlesCap, cudaMemcpyHostToDevice));
        //sycl_device_queue.memcpy(&(P_d->_particlesSize), &(P->_particles._size), sizeof(int));
       safeCall(cudaMemcpy(&(P_d->_particlesSize), &(P->_particles._size), sizeof(int), cudaMemcpyHostToDevice));
    }
    //sycl_device_queue.memcpy(&(P_d->_particles), &_particles_d, sizeof(MC_Base_Particle*));
    safeCall(cudaMemcpy(&(P_d->_particles), &_particles_d, sizeof(MC_Base_Particle*), cudaMemcpyHostToDevice));
}

void copyParticleVault_d2h(ParticleVault* P, ParticleVault_d* P_d)
{
    int _particlesCap = P->_particles._capacity;
    //sycl_device_queue.memcpy(&(P->_particles._size), &(P_d->_particlesSize), sizeof(int));
    safeCall(cudaMemcpy(&(P->_particles._size), &(P_d->_particlesSize), sizeof(int), cudaMemcpyDeviceToHost));
    MC_Base_Particle* _particles_add;
    //sycl_device_queue.memcpy(&_particles_add, &(P_d->_particles), sizeof(MC_Base_Particle*));
    safeCall(cudaMemcpy(&_particles_add, &(P_d->_particles), sizeof(MC_Base_Particle*), cudaMemcpyDeviceToHost));
    //sycl_device_queue.memcpy(P->_particles._data, _particles_add, _particlesCap*sizeof(MC_Base_Particle)).wait();
    safeCall(cudaMemcpy(P->_particles._data, _particles_add, _particlesCap*sizeof(MC_Base_Particle), cudaMemcpyDeviceToHost));
    safeCall(cudaDeviceSynchronize());
    //free(_particles_add, sycl_device_queue);
    cudaFree(_particles_add);
}

void copySendQueueDevice(SendQueue* sq, SendQueue_d* sq_d)
{
    int _dataSize = sq->size();
    //sendQueueTuple* _data_d = sycl::malloc_device<sendQueueTuple>(_dataSize, sycl_device_queue);
    sendQueueTuple* _data_d;
    safeCall(cudaMalloc((void **)&_data_d, sizeof(sendQueueTuple) * _dataSize));
    //sycl_device_queue.memcpy(_data_d, sq->get_data_pointer(), _dataSize*sizeof(sendQueueTuple));
    safeCall(cudaMemcpy(_data_d, sq->get_data_pointer(), sizeof(sendQueueTuple)*_dataSize, cudaMemcpyHostToDevice));
    //sycl_device_queue.memcpy(&(sq_d->_dataSize), &(sq->_data._size), sizeof(int));
    safeCall(cudaMemcpy(&(sq_d->_dataSize), &(sq->_data._size), sizeof(int), cudaMemcpyHostToDevice));
    //sycl_device_queue.memcpy(&(sq_d->_data), &(_data_d), sizeof(int));
    safeCall(cudaMemcpy(&(sq_d->_data), &(_data_d), sizeof(int), cudaMemcpyHostToDevice));
}

void copySendQueueHost(SendQueue_d* sq_d, SendQueue* sq)
{
    SendQueue_d* sq_h = (SendQueue_d*) malloc(sizeof(SendQueue_d));
    //sycl_device_queue.memcpy(sq_h, sq_d, sizeof(SendQueue_d)).wait();
    safeCall(cudaMemcpy(sq_h, sq_d, sizeof(SendQueue_d), cudaMemcpyDeviceToHost));
    safeCall(cudaDeviceSynchronize());
    int _dataSize = sq_h->_dataSize;
    //sycl_device_queue.memcpy(sq->get_data_pointer(), sq_h->_data, _dataSize*sizeof(sendQueueTuple)).wait();
    safeCall(cudaMemcpy(sq->get_data_pointer(), sq_h->_data, _dataSize*sizeof(sendQueueTuple), cudaMemcpyDeviceToHost));
    safeCall(cudaDeviceSynchronize());
    //sycl::free(sq_h->_data, sycl_device_queue);
    cudaFree(sq_h->_data);
    sq->_data.setsize(_dataSize);
    free(sq_h);
}

void copyParticleVaultContainerH2D(ParticleVaultContainer* pvc, ParticleVaultContainer_d* pvc_d)
{
    //sycl_device_queue.memcpy(&(pvc_d->_vaultSize), &(pvc->_vaultSize), sizeof(uint64_t));
    safeCall(cudaMemcpy(&(pvc_d->_vaultSize), &(pvc->_vaultSize), sizeof(uint64_t), cudaMemcpyHostToDevice));
    //sycl_device_queue.memcpy(&(pvc_d->_numExtraVaults), &(pvc->_numExtraVaults), sizeof(uint64_t));
    safeCall(cudaMemcpy(&(pvc_d->_numExtraVaults), &(pvc->_numExtraVaults), sizeof(uint64_t), cudaMemcpyHostToDevice));
    //sycl_device_queue.memcpy(&(pvc_d->_extraVaultIndex), &(pvc->_extraVaultIndex), sizeof(uint64_t));
    safeCall(cudaMemcpy(&(pvc_d->_extraVaultIndex), &(pvc->_extraVaultIndex), sizeof(uint64_t), cudaMemcpyHostToDevice));
    //SendQueue_d* sq_d = sycl::malloc_device<SendQueue_d>(1, sycl_device_queue);
    SendQueue_d* sq_d;
    safeCall(cudaMalloc((void **)&sq_d, sizeof(SendQueue_d)));
    copySendQueueDevice(pvc->getSendQueue(), sq_d);
    //sycl_device_queue.memcpy(&(pvc_d->_sendQueue), &(sq_d), sizeof(uint64_t));
    safeCall(cudaMemcpy(&(pvc_d->_sendQueue), &(sq_d), sizeof(uint64_t), cudaMemcpyHostToDevice));
    int _extraVaultSize = pvc->_extraVault.size();
    //sycl_device_queue.memcpy(&(pvc_d->_extraVaultSize), &(pvc->_extraVault._size), sizeof(int));
    safeCall(cudaMemcpy(&(pvc_d->_extraVaultSize), &(pvc->_extraVault._size), sizeof(int), cudaMemcpyHostToDevice));
    //ParticleVault_d **_extraVault_d = sycl::malloc_device<ParticleVault_d*>(_extraVaultSize, sycl_device_queue);
    ParticleVault_d **_extraVault_d;
    safeCall(cudaMalloc((void ***)&_extraVault_d, sizeof(ParticleVault_d*)*_extraVaultSize));
    for(int i=0; i<_extraVaultSize; i++)
    {
        //ParticleVault_d* tmp = sycl::malloc_device<ParticleVault_d>(1, sycl_device_queue);;
        ParticleVault_d* tmp;
        safeCall(cudaMalloc((void **)&tmp, sizeof(ParticleVault_d)));
        copyParticleVault_h2d(tmp, pvc->_extraVault[i]);
        //sycl_device_queue.memcpy(&(_extraVault_d[i]), &tmp, sizeof(ParticleVault_d*));
        safeCall(cudaMemcpy(&(_extraVault_d[i]), &(tmp), sizeof(ParticleVault_d*), cudaMemcpyHostToDevice));
    }
    //sycl_device_queue.memcpy(&(pvc_d->_extraVault), &_extraVault_d, sizeof(ParticleVault_d**));
    safeCall(cudaMemcpy(&(pvc_d->_extraVault), &_extraVault_d, sizeof(ParticleVault_d**), cudaMemcpyHostToDevice));
}

void copyParticleVaultContainerD2H(ParticleVaultContainer_d* pvc_d, ParticleVaultContainer* pvc)
{
    ParticleVaultContainer_d* pvc_h = (ParticleVaultContainer_d*)malloc(sizeof(ParticleVaultContainer_d));
    //sycl_device_queue.memcpy(pvc_h, pvc_d, sizeof(ParticleVaultContainer_d)).wait();
    safeCall(cudaMemcpy(pvc_h, pvc_d, sizeof(ParticleVaultContainer_d), cudaMemcpyDeviceToHost));
    safeCall(cudaDeviceSynchronize());

    pvc->_vaultSize = pvc_h->_vaultSize;
    pvc->_numExtraVaults = pvc_h->_numExtraVaults;
    pvc->_extraVaultIndex = pvc_h->_extraVaultIndex;

    //sycl::free(pvc_h->getSendQueue(), sycl_device_queue);
    cudaFree(pvc_h->getSendQueue());
    int _extraVaultSize = pvc_h->_extraVaultSize;
    ParticleVault_d** _extraVault_h = (ParticleVault_d **)malloc(_extraVaultSize*sizeof(ParticleVault_d*));
    //sycl_device_queue.memcpy(_extraVault_h, pvc_h->_extraVault, _extraVaultSize*sizeof(ParticleVault_d*)).wait();
    safeCall(cudaMemcpy(_extraVault_h, pvc_h->_extraVault, _extraVaultSize*sizeof(ParticleVault_d*), cudaMemcpyDeviceToHost));
    safeCall(cudaDeviceSynchronize());

    //sycl::free(pvc_h->_extraVault, sycl_device_queue);
    cudaFree(pvc_h->_extraVault);
    for(int i=0; i<_extraVaultSize; i++)
    {
        copyParticleVault_d2h(pvc->_extraVault[i], _extraVault_h[i]);
        //sycl::free(_extraVault_h[i],sycl_device_queue);
        cudaFree(_extraVault_h[i]);
    }
    free(_extraVault_h);
    free(pvc_h);
 }

void copyMonteCarloDevice_first(MonteCarlo* mc, MonteCarlo_d* mc_d)
{
    //Tallies_d *tallies_d = sycl::malloc_device<Tallies_d>(1, sycl_device_queue);
    Tallies_d *tallies_d;
    safeCall(cudaMalloc((void **)&tallies_d, sizeof(Tallies_d)));
    //sycl_device_queue.memcpy(&(mc_d->_tallies_d), &(tallies_d), sizeof(Tallies_d*));
    safeCall(cudaMemcpy(&(mc_d->_tallies_d), &(tallies_d), sizeof(Tallies_d*), cudaMemcpyHostToDevice));

    //sycl_device_queue.memcpy(&(tallies_d->_num_balance_replications), &(mc->_tallies->_num_balance_replications), sizeof(int));
    safeCall(cudaMemcpy(&(tallies_d->_num_balance_replications), &(mc->_tallies->_num_balance_replications), sizeof(int), cudaMemcpyHostToDevice));
    //sycl_device_queue.memcpy(&(tallies_d->_num_flux_replications), &(mc->_tallies->_num_flux_replications), sizeof(int));
    safeCall(cudaMemcpy(&(tallies_d->_num_flux_replications), &(mc->_tallies->_num_flux_replications), sizeof(int), cudaMemcpyHostToDevice));
    //sycl_device_queue.memcpy(&(tallies_d->_num_cellTally_replications), &(mc->_tallies->_num_cellTally_replications), sizeof(int));
    safeCall(cudaMemcpy(&(tallies_d->_num_cellTally_replications), &(mc->_tallies->_num_cellTally_replications), sizeof(int), cudaMemcpyHostToDevice));

    int _scalarFluxDomainSize = mc->_tallies->_scalarFluxDomain.size();
    //ScalarFluxDomain_d *_scalarFluxDomain_d = sycl::malloc_device<ScalarFluxDomain_d>(_scalarFluxDomainSize, sycl_device_queue);
    ScalarFluxDomain_d *_scalarFluxDomain_d;
    safeCall(cudaMalloc((void **)&_scalarFluxDomain_d, sizeof(ScalarFluxDomain_d)*_scalarFluxDomainSize));
    //sycl_device_queue.memcpy(&(tallies_d->_scalarFluxDomainSize), &(_scalarFluxDomainSize), sizeof(int));
    safeCall(cudaMemcpy(&(tallies_d->_scalarFluxDomainSize), &(_scalarFluxDomainSize), sizeof(int), cudaMemcpyHostToDevice));
    //sycl_device_queue.memcpy(&(tallies_d->_scalarFluxDomain), &(_scalarFluxDomain_d), sizeof(ScalarFluxDomain_d*));
    safeCall(cudaMemcpy(&(tallies_d->_scalarFluxDomain), &(_scalarFluxDomain_d), sizeof(ScalarFluxDomain_d*), cudaMemcpyHostToDevice));

    for(int i=0; i<_scalarFluxDomainSize; i++)
    {
        int _taskSize = mc->_tallies->_scalarFluxDomain[i]._task.size();
        //ScalarFluxTask_d *_task = sycl::malloc_device<ScalarFluxTask_d>(_taskSize,sycl_device_queue);
        ScalarFluxTask_d *_task;
        safeCall(cudaMalloc((void **)&_task, sizeof(ScalarFluxTask_d)*_taskSize));
        //sycl_device_queue.memcpy(&(_scalarFluxDomain_d[i]._taskSize), &(_taskSize), sizeof(int));
        safeCall(cudaMemcpy(&(_scalarFluxDomain_d[i]._taskSize), &(_taskSize), sizeof(int), cudaMemcpyHostToDevice));
        //sycl_device_queue.memcpy(&(_scalarFluxDomain_d[i]._task), &(_task), sizeof(ScalarFluxTask_d*));
        safeCall(cudaMemcpy(&(_scalarFluxDomain_d[i]._task), &(_task), sizeof(ScalarFluxTask_d*), cudaMemcpyHostToDevice));
        for(int j=0; j<_taskSize; j++)
        {
            int _cellSize = mc->_tallies->_scalarFluxDomain[i]._task[j]._cell.size();
            //ScalarFluxCell *_cell = sycl::malloc_device<ScalarFluxCell>(_cellSize, sycl_device_queue);
            ScalarFluxCell *_cell;
            safeCall(cudaMalloc((void **)&_cell, sizeof(ScalarFluxCell)*_cellSize));
            //sycl_device_queue.memcpy(&(_task[j]._cellSize), &(_cellSize), sizeof(int));
            safeCall(cudaMemcpy(&(_task[j]._cellSize), &(_cellSize), sizeof(int), cudaMemcpyHostToDevice));
            //sycl_device_queue.memcpy(&(_task[j]._cell), &(_cell), sizeof(ScalarFluxCell*));
            safeCall(cudaMemcpy(&(_task[j]._cell), &(_cell), sizeof(ScalarFluxCell*), cudaMemcpyHostToDevice));
            for(int k=0; k<_cellSize; k++)
            {
                int _size = mc->_tallies->_scalarFluxDomain[i]._task[j]._cell[k].size();
                //double *_group = sycl::malloc_device<double>(_size, sycl_device_queue);
                double *_group;
                safeCall(cudaMalloc((void **)&_group, sizeof(double)*_size));
                //sycl_device_queue.memcpy(_group, mc->_tallies->_scalarFluxDomain[i]._task[j]._cell[k]._group, _size*sizeof(double));
                safeCall(cudaMemcpy(_group, mc->_tallies->_scalarFluxDomain[i]._task[j]._cell[k]._group, _size*sizeof(double), cudaMemcpyHostToDevice));
                //sycl_device_queue.memcpy(&(_cell[k]._size), &(_size), sizeof(int));
                safeCall(cudaMemcpy(&(_cell[k]._size), &(_size), sizeof(int), cudaMemcpyHostToDevice));
                //sycl_device_queue.memcpy(&(_cell[k]._group), &(_group), sizeof(double *));
                safeCall(cudaMemcpy(&(_cell[k]._group), &(_group), sizeof(double *), cudaMemcpyHostToDevice));
            }
        }
    }
}

void copyMonteCarloDevice_part(MonteCarlo* mc, MonteCarlo_d* mc_d)
{
    //sycl_device_queue.memcpy(&(mc_d->domain_d), &(mc->domain_d), sizeof(MC_Domain_d*));
    safeCall(cudaMemcpy(&(mc_d->domain_d), &(mc->domain_d), sizeof(MC_Domain_d*), cudaMemcpyHostToDevice));
    //sycl_device_queue.memcpy(&(mc_d->_material_d), &(mc->_material_d), sizeof(Material_d*));
    safeCall(cudaMemcpy(&(mc_d->_material_d), &(mc->_material_d), sizeof(Material_d*), cudaMemcpyHostToDevice));
    //sycl_device_queue.memcpy(&(mc_d->_nuclearData_d), &(mc->_nuclearData_d), sizeof(NuclearData_d*));
    safeCall(cudaMemcpy(&(mc_d->_nuclearData_d), &(mc->_nuclearData_d), sizeof(NuclearData_d*), cudaMemcpyHostToDevice));

    //MC_Time_Info* time_info_d = sycl::malloc_device<MC_Time_Info>(1, sycl_device_queue);
    MC_Time_Info* time_info_d;
    safeCall(cudaMalloc((void **)&time_info_d, sizeof(MC_Time_Info)));
    //sycl_device_queue.memcpy(time_info_d, mc->time_info, sizeof(MC_Time_Info));
    safeCall(cudaMemcpy(time_info_d, mc->time_info, sizeof(MC_Time_Info), cudaMemcpyHostToDevice));
    //sycl_device_queue.memcpy(&(mc_d->time_info_d), &(time_info_d), sizeof(MC_Time_Info*));
    safeCall(cudaMemcpy(&(mc_d->time_info_d), &(time_info_d), sizeof(MC_Time_Info*), cudaMemcpyHostToDevice));

    //ParticleVaultContainer_d* _particleVaultContainer_d = sycl::malloc_device<ParticleVaultContainer_d>(1, sycl_device_queue);
    ParticleVaultContainer_d* _particleVaultContainer_d;
    safeCall(cudaMalloc((void **)&_particleVaultContainer_d, sizeof(ParticleVaultContainer_d)));
    //sycl_device_queue.memcpy(&(mc_d->_particleVaultContainer_d), &(_particleVaultContainer_d), sizeof(ParticleVaultContainer_d*));
    safeCall(cudaMemcpy(&(mc_d->_particleVaultContainer_d), &(_particleVaultContainer_d), sizeof(ParticleVaultContainer_d*), cudaMemcpyHostToDevice));
    //sycl_device_queue.memcpy(&(_particleVaultContainer_d->_vaultSize), &(mc->_particleVaultContainer->_vaultSize), sizeof(uint64_t));
    safeCall(cudaMemcpy(&(_particleVaultContainer_d->_vaultSize), &(mc->_particleVaultContainer->_vaultSize), sizeof(uint64_t), cudaMemcpyHostToDevice));
    //sycl_device_queue.memcpy(&(_particleVaultContainer_d->_numExtraVaults), &(mc->_particleVaultContainer->_numExtraVaults), sizeof(uint64_t));
    safeCall(cudaMemcpy(&(_particleVaultContainer_d->_numExtraVaults), &(mc->_particleVaultContainer->_numExtraVaults), sizeof(uint64_t), cudaMemcpyHostToDevice));
    //sycl_device_queue.memcpy(&(_particleVaultContainer_d->_extraVaultIndex), &(mc->_particleVaultContainer->_extraVaultIndex), sizeof(uint64_cu));
    safeCall(cudaMemcpy(&(_particleVaultContainer_d->_extraVaultIndex), &(mc->_particleVaultContainer->_extraVaultIndex), sizeof(uint64_cu), cudaMemcpyHostToDevice));
    int _extraVaultSize = mc->_particleVaultContainer->_extraVault.size();
    //sycl_device_queue.memcpy(&(_particleVaultContainer_d->_extraVaultSize), &(_extraVaultSize), sizeof(int));
    safeCall(cudaMemcpy(&(_particleVaultContainer_d->_extraVaultSize), &(_extraVaultSize), sizeof(int), cudaMemcpyHostToDevice));
    //ParticleVault_d ** _extraVault=sycl::malloc_device<ParticleVault_d*>(_extraVaultSize, sycl_device_queue);
    ParticleVault_d ** _extraVault;
    safeCall(cudaMalloc((void ***)&_extraVault, sizeof(ParticleVault_d*)*_extraVaultSize));
    //sycl_device_queue.memcpy(&(_particleVaultContainer_d->_extraVault), &(_extraVault), sizeof(ParticleVault**));
    safeCall(cudaMemcpy(&(_particleVaultContainer_d->_extraVault), &(_extraVault), sizeof(ParticleVault**), cudaMemcpyHostToDevice));
    for(int i=0;i<_extraVaultSize;i++)
    {
	    //ParticleVault_d *tmp = sycl::malloc_device<ParticleVault_d>(1, sycl_device_queue);
        ParticleVault_d *tmp;
        safeCall(cudaMalloc((void **)&tmp, sizeof(ParticleVault_d)));
	    copyParticleVault_h2d(tmp, mc->_particleVaultContainer->_extraVault[i]);
	    //sycl_device_queue.memcpy(&(_extraVault[i]), &(tmp), sizeof(ParticleVault*));
        safeCall(cudaMemcpy(&(_extraVault[i]), &(tmp), sizeof(ParticleVault*), cudaMemcpyHostToDevice));
    }
}

void copyMonteCarloHost_part(MonteCarlo_d* mc_d, MonteCarlo* mc)
{
    MC_Time_Info *tmp;
    //sycl_device_queue.memcpy(&tmp, &(mc_d->time_info_d), sizeof(MC_Time_Info*)).wait();
    safeCall(cudaMemcpy(&tmp, &(mc_d->time_info_d), sizeof(MC_Time_Info*), cudaMemcpyDeviceToHost));
    safeCall(cudaDeviceSynchronize());
    //sycl_device_queue.memcpy(mc->time_info, tmp, sizeof(MC_Time_Info)).wait();
    safeCall(cudaMemcpy(mc->time_info, tmp, sizeof(MC_Time_Info), cudaMemcpyDeviceToHost));
    safeCall(cudaDeviceSynchronize());
    //sycl::free(tmp,sycl_device_queue);
    cudaFree(tmp);

    ParticleVaultContainer_d* tmp_p;
    //sycl_device_queue.memcpy(&tmp_p, &(mc_d->_particleVaultContainer_d), sizeof(ParticleVaultContainer_d*)).wait();
    safeCall(cudaMemcpy(&tmp_p, &(mc_d->_particleVaultContainer_d), sizeof(ParticleVaultContainer_d*), cudaMemcpyDeviceToHost));
    safeCall(cudaDeviceSynchronize());
    //sycl_device_queue.memcpy(&(mc->_particleVaultContainer->_vaultSize), &(tmp_p->_vaultSize), sizeof(uint64_t)).wait();
    safeCall(cudaMemcpy(&(mc->_particleVaultContainer->_vaultSize), &(tmp_p->_vaultSize), sizeof(uint64_t), cudaMemcpyDeviceToHost));
    safeCall(cudaDeviceSynchronize());
    //sycl_device_queue.memcpy(&(mc->_particleVaultContainer->_numExtraVaults), &(tmp_p->_numExtraVaults), sizeof(uint64_t)).wait(); 
    safeCall(cudaMemcpy(&(mc->_particleVaultContainer->_numExtraVaults), &(tmp_p->_numExtraVaults), sizeof(uint64_t), cudaMemcpyDeviceToHost));
    safeCall(cudaDeviceSynchronize());
    //sycl_device_queue.memcpy(&(mc->_particleVaultContainer->_extraVaultIndex), &(tmp_p->_extraVaultIndex), sizeof(uint64_cu)).wait();
    safeCall(cudaMemcpy(&(mc->_particleVaultContainer->_extraVaultIndex), &(tmp_p->_extraVaultIndex), sizeof(uint64_cu), cudaMemcpyDeviceToHost));
    safeCall(cudaDeviceSynchronize());

    ParticleVault_d **tmp_ev;
    //sycl_device_queue.memcpy(&(mc->_particleVaultContainer->_extraVault._size), &(tmp_p->_extraVaultSize), sizeof(int)).wait();
    safeCall(cudaMemcpy(&(mc->_particleVaultContainer->_extraVault._size), &(tmp_p->_extraVaultSize), sizeof(int), cudaMemcpyDeviceToHost));
    safeCall(cudaDeviceSynchronize());
    //sycl_device_queue.memcpy(&tmp_ev, &(tmp_p->_extraVault), sizeof(ParticleVault_d **)).wait();
    safeCall(cudaMemcpy(&tmp_ev, &(tmp_p->_extraVault), sizeof(ParticleVault_d **), cudaMemcpyDeviceToHost));
    safeCall(cudaDeviceSynchronize());
    for(int i=0;i<mc->_particleVaultContainer->_extraVault.size();i++)
    {
	    ParticleVault_d *tmp_0;
	    //sycl_device_queue.memcpy(&tmp_0, &(tmp_ev[i]), sizeof(ParticleVault_d *)).wait();
        safeCall(cudaMemcpy(&tmp_0, &(tmp_ev[i]), sizeof(ParticleVault_d *), cudaMemcpyDeviceToHost));
	safeCall(cudaDeviceSynchronize());
        copyParticleVault_d2h(mc->_particleVaultContainer->_extraVault[i], tmp_0);
	    //sycl::free(tmp_0, sycl_device_queue);
        cudaFree(tmp_0);
    }
    //sycl::free(tmp_ev, sycl_device_queue);
    cudaFree(tmp_ev);
    //sycl::free(tmp_p, sycl_device_queue);
    cudaFree(tmp_p);
}

void copyMonteCarloHost_last(MonteCarlo_d* mc_d, MonteCarlo* mc)
{
    Tallies_d *tmp_t;
    //sycl_device_queue.memcpy(&tmp_t, &(mc_d->_tallies_d), sizeof(Tallies_d*)).wait();
    safeCall(cudaMemcpy(&tmp_t, &(mc_d->_tallies_d), sizeof(Tallies_d*), cudaMemcpyDeviceToHost));
    safeCall(cudaDeviceSynchronize());
    ScalarFluxDomain_d* tmp_s;
    //sycl_device_queue.memcpy(&(tmp_s), &(tmp_t->_scalarFluxDomain), sizeof(ScalarFluxDomain_d*)).wait();
    safeCall(cudaMemcpy(&(tmp_s), &(tmp_t->_scalarFluxDomain), sizeof(ScalarFluxDomain_d*), cudaMemcpyDeviceToHost));
    safeCall(cudaDeviceSynchronize());
    int _scalarFluxDomainSize = mc->_tallies->_scalarFluxDomain.size();
    for(int i=0;i<_scalarFluxDomainSize;i++)
    {
        ScalarFluxTask_d* tmp_task;
        //sycl_device_queue.memcpy(&(tmp_task), &(tmp_s[i]._task), sizeof(ScalarFluxTask_d*)).wait();
        safeCall(cudaMemcpy(&(tmp_task), &(tmp_s[i]._task), sizeof(ScalarFluxTask_d*), cudaMemcpyDeviceToHost));
	safeCall(cudaDeviceSynchronize());
        int _taskSize = mc->_tallies->_scalarFluxDomain[i]._task.size();
        for(int j=0; j<_taskSize; j++)
        {
            ScalarFluxCell* tmp_cell;
            //sycl_device_queue.memcpy(&(tmp_cell), &(tmp_task[j]._cell), sizeof(ScalarFluxCell*)).wait();
            safeCall(cudaMemcpy(&(tmp_cell), &(tmp_task[j]._cell), sizeof(ScalarFluxCell*), cudaMemcpyDeviceToHost));
	    safeCall(cudaDeviceSynchronize());
            int _cellSize = mc->_tallies->_scalarFluxDomain[i]._task[j]._cell.size();
            for(int k=0; k<_cellSize; k++)
            {
                double *tmp_d;
                //sycl_device_queue.memcpy(&(tmp_d), &(tmp_cell[k]._group), sizeof(double*)).wait();
                safeCall(cudaMemcpy(&(tmp_d), &(tmp_cell[k]._group), sizeof(double*), cudaMemcpyDeviceToHost));
		safeCall(cudaDeviceSynchronize());
                int _size = mc->_tallies->_scalarFluxDomain[i]._task[j]._cell[k].size();
                //sycl_device_queue.memcpy(mc->_tallies->_scalarFluxDomain[i]._task[j]._cell[k]._group, tmp_d, _size*sizeof(double)).wait();
                safeCall(cudaMemcpy(mc->_tallies->_scalarFluxDomain[i]._task[j]._cell[k]._group, tmp_d, _size*sizeof(double), cudaMemcpyDeviceToHost));
		safeCall(cudaDeviceSynchronize());
                //sycl::free(tmp_d, sycl_device_queue);
                cudaFree(tmp_d);
            }
            //sycl::free(tmp_cell, sycl_device_queue);
            cudaFree(tmp_cell);
        }
        //sycl::free(tmp_task, sycl_device_queue);
        cudaFree(tmp_task);
    }
    //sycl::free(tmp_s, sycl_device_queue);
    cudaFree(tmp_s);
    //sycl::free(tmp_t, sycl_device_queue);
    cudaFree(tmp_t);
}

void setGPU()
{

    int rank;
    MPI_Comm comm_mc_world(MPI_COMM_WORLD);

    int Ngpus;
    safeCall(cudaGetDeviceCount(&Ngpus));

    mpiComm_rank(comm_mc_world, &rank);
    int GPUID = rank % Ngpus;
    safeCall(cudaSetDevice(GPUID));
}

using namespace std;

MonteCarlo *mcco = NULL;

int main(int argc, char **argv)
{
    mpiInit(&argc, &argv);
    printBanner(GIT_VERS, GIT_HASH);

    Parameters params = getParameters(argc, argv);
    const string &filename = params.simulationParams.inputFile;
    ifstream inp_file(filename.c_str());
    if (!inp_file.good())
    {
        return -1;
    }
    printParameters(params, cout);

    setGPU();
//printf("point0\n");
    // mcco stores just about everything.
    mcco = initMC(params);
//printf("point0.5\n");
    int myRank, nRanks;
    mpiComm_rank(MPI_COMM_WORLD, &myRank);
//printf("point1\n");
    copyMaterialDatabase_device(mcco);
    copyNuclearData_device(mcco->_nuclearData, mcco->_nuclearData_d);
    copyDomainDevice(mcco->_nuclearData->_numEnergyGroups, mcco->domain, mcco->domain_d, mcco->domainSize);
//     printf("point2\n");
    mpiBarrier(MPI_COMM_WORLD);
    int loadBalance = params.simulationParams.loadBalance;

    MC_FASTTIMER_START(MC_Fast_Timer::main); // this can be done once mcco exist.

    const int nSteps = params.simulationParams.nSteps;

    // allocate arrays to hold counters in pinned memory on the host and on the device.
    int replications = mcco->_tallies->GetNumBalanceReplications();
    uint64_cu *tallies;
    safeCall(cudaMallocHost((void **)&tallies, sizeof(uint64_cu) * NUM_TALLIES * replications));

    uint64_cu *tallies_d;
    safeCall(cudaMalloc((void **)&tallies_d, sizeof(uint64_cu) * NUM_TALLIES * replications));

    for (int il = 0; il < replications; il++)
    {
        for (int j1 = 0; j1 < NUM_TALLIES; j1++)
        {
            tallies[NUM_TALLIES * il + j1] = 0;
        }
    }
    safeCall(cudaMemcpy(tallies_d, tallies, sizeof(uint64_cu) * NUM_TALLIES * replications, cudaMemcpyHostToDevice));

    for (int ii = 0; ii < nSteps; ++ii)
    {
//	printf("%d\n", ii);
//	checkMsg("-111 failed\n");
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

    safeCall(cudaFreeHost(tallies));
    safeCall(cudaFree(tallies_d));

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

#if defined(HAVE_CUDA)

__launch_bounds__(256) __global__ void CycleTrackingKernel(MonteCarlo_d *monteCarlo, int num_particles, ParticleVault_d *processingVault, ParticleVault_d *processedVault, uint64_cu *tallies)
{
    int global_index = getGlobalThreadID();
    int local_index = getLocalThreadID();
    int replications = monteCarlo->_tallies_d->GetNumBalanceReplications();

    extern __shared__ int values_l[];
    if (local_index < replications * NUM_TALLIES)
    {
        values_l[local_index] = 0;
    }
    __syncthreads();

    if (global_index < num_particles)
    {
        CycleTrackingGuts(monteCarlo, global_index, processingVault, processedVault, &values_l[0]);
    }

    __syncthreads();
    if (local_index < replications * NUM_TALLIES)
    {
#if defined(HAVE_CUDA)
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
    //MonteCarlo_d *monteCarlo_d = sycl::malloc_device<MonteCarlo_d>(1, sycl_device_queue);
  // checkMsg("-4 failed\n");
    MonteCarlo_d *monteCarlo_d;
    safeCall(cudaMalloc(&monteCarlo_d, sizeof(MonteCarlo_d)));
  //  checkMsg("-3 failed\n");
    copyMonteCarloDevice_first(monteCarlo, monteCarlo_d);
  //  checkMsg("-2 failed\n");
    do
    {

        int particle_count = 0; // Initialize count of num_particles processed

        while (!done)
        {
            uint64_t fill_vault = 0;

            for (uint64_t processing_vault = 0; processing_vault < my_particle_vault.processingSize(); processing_vault++)
            {
                //MC_FASTTIMER_START(MC_Fast_Timer::cycleTracking_Kernel);
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
#if defined(HAVE_CUDA)

                        const size_t N = numParticles;
                        unsigned int wg_size = 256;
                        unsigned int num_wgs = (N + wg_size - 1) / wg_size;
	//		checkMsg("-1 failed\n");
                        //ParticleVault_d *processingVault_d = sycl::malloc_device<ParticleVault_d>(1, sycl_device_queue);
                        ParticleVault_d *processingVault_d;
                        safeCall(cudaMalloc(&processingVault_d, sizeof(ParticleVault_d)));
                        //ParticleVault_d *processedVault_d = sycl::malloc_device<ParticleVault_d>(1, sycl_device_queue);
                        ParticleVault_d *processedVault_d;
                        safeCall(cudaMalloc(&processedVault_d, sizeof(ParticleVault_d)));
                      //  checkMsg("0 failed\n");
                        copyParticleVault_h2d(processingVault_d, processingVault);
		//	checkMsg("1 failed\n");
                        copyParticleVault_h2d(processedVault_d, processedVault);
		//	checkMsg("2 failed\n");
                        copyMonteCarloDevice_part(monteCarlo, monteCarlo_d);
		//	checkMsg("3 failed\n");
                        safeCall(cudaDeviceSynchronize());
                        //sycl_device_queue.wait();
                        MC_FASTTIMER_START(MC_Fast_Timer::cycleTracking_Kernel);
                        
			CycleTrackingKernel<<<num_wgs, wg_size, NUM_TALLIES * replications * sizeof(int), 0>>>(monteCarlo_d, numParticles, processingVault_d, processedVault_d, tallies_d);
                                       
                        checkMsg("CycleTrackingKernel, execution failed\n");
                        safeCall(cudaDeviceSynchronize());
                        MC_FASTTIMER_STOP(MC_Fast_Timer::cycleTracking_Kernel);
                        safeCall(cudaMemcpy(tallies, tallies_d, NUM_TALLIES * sizeof(uint64_cu) * replications, cudaMemcpyDeviceToHost));
                        safeCall(cudaDeviceSynchronize());

                        copyParticleVault_d2h(processingVault, processingVault_d);
	                    copyParticleVault_d2h(processedVault, processedVault_d);
                        copyMonteCarloHost_part(monteCarlo_d, monteCarlo);
                        //sycl::free(processingVault_d, sycl_device_queue);
                        cudaFree(processingVault_d);
			            //sycl::free(processedVault_d, sycl_device_queue);
                        cudaFree(processedVault_d);
#endif                  
                    }
                    break;

                    case gpuWithOpenMP:
                    {

                        std::cout << " this isn't supported with hip yet " << std::endl;
                    }
                    break;

                    /*case cpu:
#include "mc_omp_parallel_for_schedule_static.hh"
                        for (int particle_index = 0; particle_index < numParticles; particle_index++)
                        {
                            CycleTrackingGuts(monteCarlo, particle_index, processingVault, processedVault, &particle_index);
                        }
                        break;*/
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
                    safeCall(cudaMemcpy(tallies_d, tallies, sizeof(uint64_cu) * NUM_TALLIES * replications, cudaMemcpyHostToDevice));
                }

                particle_count += numParticles;
                //MC_FASTTIMER_STOP(MC_Fast_Timer::cycleTracking_Kernel);

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
            //sycl::free(monteCarlo_d, sycl_device_queue);
            cudaFree(monteCarlo_d);
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
