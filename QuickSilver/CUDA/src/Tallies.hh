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

#ifndef TALLIES_HH
#define TALLIES_HH

#include "portability.hh"
#include "QS_Vector.hh"
#include <stdlib.h>
#include <string>
#include <vector>
#include <cinttypes>
#include "NuclearData.hh"
#include "MonteCarlo.hh"
#include "utils.hh"
#include "macros.hh"
#include "BulkStorage.hh"
#include "DeclareMacro.hh"
#include "EnergySpectrum.hh"

typedef unsigned long long int uint64_cu;

class Fluence;

struct MC_Tally_Event
{
    enum Enum
    {
        Collision,
        Facet_Crossing_Transit_Exit,
        Census,
        Facet_Crossing_Tracking_Error,
        Facet_Crossing_Escape,
        Facet_Crossing_Reflection,
        Facet_Crossing_Communication
    };
};

class Balance
{
public:
    uint64_cu _absorb;      // Number of particles absorbed
    uint64_cu _census;      // Number of particles that enter census
    uint64_cu _escape;      // Number of particles that escape
    uint64_cu _collision;   // Number of collosions
    uint64_cu _end;         // Number of particles at end of cycle
    uint64_cu _fission;     // Number of fission events
    uint64_cu _produce;     // Number of particles created by collisions
    uint64_cu _scatter;     // Number of scatters
    uint64_cu _start;       // Number of particles at beginning of cycle
    uint64_cu _source;      // Number of particles sourced in
    uint64_cu _rr;          // Number of particles Russian Rouletted in population control
    uint64_cu _split;       // Number of particles split in population control
    uint64_cu _numSegments; // Number of segements

    Balance() : _absorb(0), _census(0), _escape(0), _collision(0), _end(0), _fission(0), _produce(0), _scatter(0), _start(0),
                _source(0), _rr(0), _split(0), _numSegments(0) {}

    ~Balance() {}

    void PrintHeader()
    {
        Print0("%6s %7s %6s %6s %10s %10s %7s %10s %10s %8s %10s %10s",
               "start", "source", "rr", "split",
               "absorb", "scatter", "fission", "produce",
               "collisn", "escape", "census",
               "num_seg");
    }

    void Print()
    {
        Print0("%6" PRIu64 " %7" PRIu64 " %6" PRIu64 " %6" PRIu64 " %10" PRIu64 " %10" PRIu64 " %7" PRIu64 " %10" PRIu64 " %10" PRIu64 " %8" PRIu64 " %10" PRIu64 " %10" PRIu64 "",
               _start, _source, _rr, _split,
               _absorb, _scatter, _fission, _produce,
               _collision, _escape, _census,
               _numSegments);
    }

    void Reset()
    {
        _absorb = _census = _escape = _collision = _end = _fission = _produce = _scatter = _start = _source =
            _rr = _split = _numSegments = 0;
    }

    HOST_DEVICE
    void Add(Balance &bal)
    {
        _absorb += bal._absorb;
        _census += bal._census;
        _escape += bal._escape;
        _collision += bal._collision;
        _end += bal._end;
        _fission += bal._fission;
        _produce += bal._produce;
        _scatter += bal._scatter;
        _start += bal._start;
        _source += bal._source;
        _rr += bal._rr;
        _split += bal._split;
        _numSegments += bal._numSegments;
    }
};

class ScalarFluxCell
{
public:
    double *_group;
    int _size;

    ScalarFluxCell() : _group(0), _size(0) {}

    ScalarFluxCell(double *storage, int size)
        : _group(storage),
          _size(size)
    {
        for (int i = 0; i < _size; i++)
        {
            _group[i] = 0.0;
        }
    }

    ~ScalarFluxCell() {}

    int size() const { return _size; }
};

class CellTallyTask
{
public:
    qs_vector<double> _cell;

    CellTallyTask() : _cell() {}

    CellTallyTask(MC_Domain *domain)
    {
        if (_cell.capacity() == 0)
        {
            _cell.reserve(domain->cell_state.size(), VAR_MEM);
        }

        _cell.Open();
        for (int cellIndex = 0; cellIndex < domain->cell_state.size(); cellIndex++)
        {
            _cell.push_back(0.0);
        }
        _cell.Close();
    }

    void Add(CellTallyTask &cellTallyTask)
    {
        for (int cellIndex = 0; cellIndex < _cell.size(); cellIndex++)
        {
            _cell[cellIndex] += cellTallyTask._cell[cellIndex];
        }
    }

    void Reset()
    {
        for (int cellIndex = 0; cellIndex < _cell.size(); cellIndex++)
        {
            _cell[cellIndex] = 0.0;
        }
    }

    ~CellTallyTask() {}
};
//----------------------------------------------------
class CellTallyTask_d
{
public:
    //qs_vector<double> _cell;
    double *_cell;
    int _cellSize;

    CellTallyTask_d() {}
    ~CellTallyTask_d() {}
};

class ScalarFluxTask
{
public:
    qs_vector<ScalarFluxCell> _cell;
    BulkStorage<double> _scalarFluxCellStorage;

    ScalarFluxTask() : _cell() {}

    ScalarFluxTask(MC_Domain *domain, int numGroups)
    {
        if (_cell.capacity() == 0)
        {
            _cell.reserve(domain->cell_state.size(), VAR_MEM);
            _scalarFluxCellStorage.setCapacity(domain->cell_state.size() * numGroups, VAR_MEM);
        }

        _cell.Open();
        for (int cellIndex = 0; cellIndex < domain->cell_state.size(); cellIndex++)
        {
            double *tmp = _scalarFluxCellStorage.getBlock(numGroups);
            _cell.push_back(ScalarFluxCell(tmp, numGroups));
        }
        _cell.Close();
    }

    void Add(ScalarFluxTask &scalarFluxTask)
    {
        unsigned int numGroups = _cell[0].size();
        for (int cellIndex = 0; cellIndex < _cell.size(); cellIndex++)
        {
            for (int groupIndex = 0; groupIndex < numGroups; groupIndex++)
            {
                _cell[cellIndex]._group[groupIndex] += scalarFluxTask._cell[cellIndex]._group[groupIndex];
            }
        }
    }

    void Reset()
    {
        unsigned int numGroups = _cell[0].size();
        for (int cellIndex = 0; cellIndex < _cell.size(); cellIndex++)
        {
            for (int groupIndex = 0; groupIndex < numGroups; groupIndex++)
            {
                _cell[cellIndex]._group[groupIndex] = 0.0;
            }
        }
    }

    ~ScalarFluxTask() {}
};
//---------------------------------------------------------------------------------------------------

class ScalarFluxTask_d
{
public:
    //qs_vector<ScalarFluxCell> _cell;
    ScalarFluxCell* _cell;
    int _cellSize;

    ScalarFluxTask_d() : _cell() {}
    ~ScalarFluxTask_d() {}
};
class CellTallyDomain
{
public:
    qs_vector<CellTallyTask> _task;

    CellTallyDomain() : _task() {}

    CellTallyDomain(MC_Domain *domain, int cellTally_replications)
    {
        // Assume OMP_NUM_THREADS tasks
        if (_task.capacity() == 0)
        {
            _task.reserve(cellTally_replications, VAR_MEM);
        }
        _task.Open();
        for (int task_index = 0; task_index < cellTally_replications; task_index++)
        {
            _task.push_back(CellTallyTask(domain));
        }
        _task.Close();
    }

    ~CellTallyDomain() {}
};
//-------------------------------------------------------------------------------------
class CellTallyDomain_d
{
public:
    //qs_vector<CellTallyTask> _task;
    CellTallyTask_d* _task;
    int _taskSize;

    CellTallyDomain_d() {}
    ~CellTallyDomain_d() {}
};
class ScalarFluxDomain
{
public:
    qs_vector<ScalarFluxTask> _task;

    ScalarFluxDomain() : _task() {}

    ScalarFluxDomain(MC_Domain *domain, int numGroups, int flux_replications)
    {
        // Assume OMP_NUM_THREADS tasks
        if (_task.capacity() == 0)
        {
            _task.reserve(flux_replications, VAR_MEM);
        }
        _task.Open();
        for (int task_index = 0; task_index < flux_replications; task_index++)
        {
            _task.push_back(ScalarFluxTask(domain, numGroups));
        }
        _task.Close();
    }

    ~ScalarFluxDomain() {}
};
//------------------------------------------------------------------------------------------------------
class ScalarFluxDomain_d
{
public:
    //qs_vector<ScalarFluxTask> _task;
    ScalarFluxTask_d* _task;
    int _taskSize;

    ScalarFluxDomain_d() {}//: _task() {}
    ~ScalarFluxDomain_d() {}
};
class FluenceDomain
{
public:
    FluenceDomain(int numCells) : _cell(numCells, 0.0)
    {
    }

    void addCell(int index, double value) { _cell[index] += value; }
    double getCell(int index) { return _cell[index]; }
    int size() { return _cell.size(); }

private:
    std::vector<double> _cell;
};
//-------------------------------------------------------------------------------------
class FluenceDomain_d
{
public:
    FluenceDomain_d(int numCells) 
    {
        _cell = new double[numCells];
    }

    void addCell(int index, double value) { _cell[index] += value; }
    double getCell(int index) { return _cell[index]; }
    int size() { return _cellSize; }

private:
   //std::vector<double> _cell;
   double *_cell;
   int _cellSize;
};

class Fluence
{
public:
    Fluence(){};
    ~Fluence()
    {
        for (int i = 0; i < _domain.size(); i++)
        {
            if (_domain[i] != NULL)
                delete _domain[i];
        }
    }

    void compute(int domain, ScalarFluxDomain &scalarFluxDomain);

    std::vector<FluenceDomain *> _domain;
};
//-----------------------------------------------------------------------------------------
class Fluence_d
{
public:
    Fluence_d(){};
    ~Fluence_d()
    {
        //for (int i = 0; i < _domain.size(); i++)
        for (int i = 0; i < _domainSize; i++)
        {
            if (_domain[i] != NULL)
                delete _domain[i];
        }
    }

    //void compute(int domain, ScalarFluxDomain_d &scalarFluxDomain);

    //std::vector<FluenceDomain *> _domain;
    FluenceDomain_d** _domain;
    int _domainSize;
};
//--------------------------------------------------------------------
class Tallies
{
public:
    Balance _balanceCumulative;
    qs_vector<Balance> _balanceTask;
    qs_vector<ScalarFluxDomain> _scalarFluxDomain;
    qs_vector<CellTallyDomain> _cellTallyDomain;
    Fluence _fluence;
    EnergySpectrum _spectrum;

    Tallies(int balRep, int fluxRep, int cellRep, std::string spectrumName, int spectrumSize) : _balanceCumulative(), _balanceTask(),
                                                                                                _scalarFluxDomain(), _num_balance_replications(balRep),
                                                                                                _num_flux_replications(fluxRep), _num_cellTally_replications(cellRep),
                                                                                                _spectrum(std::move(spectrumName), spectrumSize)
    {
    }

    HOST_DEVICE_CUDA
    int GetNumBalanceReplications()
    {
        return _num_balance_replications;
    }

    HOST_DEVICE_CUDA
    int GetNumFluxReplications()
    {
        return _num_flux_replications;
    }

    HOST_DEVICE_CUDA
    int GetNumCellTallyReplications()
    {
        return _num_cellTally_replications;
    }

    ~Tallies() {}

    void InitializeTallies(MonteCarlo *monteCarlo,
                           int balance_replications,
                           int flux_replications,
                           int cell_replications);

    void CycleInitialize(MonteCarlo *monteCarlo);

    void SumTasks();
    void CycleFinalize(MonteCarlo *mcco);
    void PrintSummary(MonteCarlo *mcco);

    // These atomic operations seem to be working.
    HOST_DEVICE_CUDA
    void TallyScalarFlux(double value, int domain, int task, int cell, int group)
    {
        ATOMIC_ADD(_scalarFluxDomain[domain]._task[task]._cell[cell]._group[group], value);
    }

    HOST_DEVICE_CUDA
    void TallyCellValue(double value, int domain, int task, int cell)
    {
        ATOMIC_ADD(_cellTallyDomain[domain]._task[task]._cell[cell], value);
    }

    double ScalarFluxSum(MonteCarlo *mcco);

//private:
    int _num_balance_replications;
    int _num_flux_replications;
    int _num_cellTally_replications;
};

//-----------------------------------------------------------------------------------------
class Tallies_d
{
public:
    //Balance _balanceCumulative;
    //qs_vector<Balance> _balanceTask;
    //qs_vector<ScalarFluxDomain> _scalarFluxDomain;
    //qs_vector<CellTallyDomain>  _cellTallyDomain;
    Balance* _balanceTask;
    int _balanceTaskSize;
    ScalarFluxDomain_d* _scalarFluxDomain;
    int _scalarFluxDomainSize;
    CellTallyDomain_d* _cellTallyDomain;
    int _cellTallyDomainSize;
    //Fluence_d                     _fluence;
   //EnergySpectrum_d              _spectrum;

    HOST_DEVICE_CUDA
    int GetNumBalanceReplications()
    {
        return _num_balance_replications;
    }

    HOST_DEVICE_CUDA
    int GetNumFluxReplications()
    {
        return _num_flux_replications;
    }

    HOST_DEVICE_CUDA
    int GetNumCellTallyReplications()
    {
        return _num_cellTally_replications;
    }

    ~Tallies_d() {}

    /*void InitializeTallies(MonteCarlo *monteCarlo,
                           int balance_replications,
                           int flux_replications,
                           int cell_replications);*/

    //void CycleInitialize(MonteCarlo *monteCarlo);

    //void SumTasks();
    //void CycleFinalize(MonteCarlo *mcco);
    //void PrintSummary(MonteCarlo *mcco);

    // These atomic operations seem to be working.
    HOST_DEVICE_CUDA
    void TallyScalarFlux(double value, int domain, int task, int cell, int group)
    {
        ATOMIC_ADD( _scalarFluxDomain[domain]._task[task]._cell[cell]._group[group], value );
    }

    HOST_DEVICE_CUDA
    void TallyCellValue(double value, int domain, int task, int cell)
    {
        ATOMIC_ADD(_cellTallyDomain[domain]._task[task]._cell[cell], value);
    }

    //double ScalarFluxSum(MonteCarlo *mcco);


    int _num_balance_replications;
    int _num_flux_replications;
    int _num_cellTally_replications;
};

#endif
