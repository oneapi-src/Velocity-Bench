/*
 * Modifications Copyright (C) 2023 Intel Corporation
 *
 * This Program is subject to the terms of the GNU Lesser General Public License v3.0 or later
 * 
 * If a copy of the license was not distributed with this file, you can obtain one at 
 * https://www.gnu.org/licenses/lgpl-3.0-standalone.html
 * 
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */


//
// Created by amr-nasr on 20/10/2019.
//

#include <operations/engines/concrete/RTMEngine.hpp>

#include <memory-manager/MemoryManager.h>

#include <iostream>
#include <chrono>

#define PB_STR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PB_WIDTH 50

using namespace operations::engines;
using namespace operations::common;
using namespace operations::dataunits;
using namespace operations::configuration;
using namespace operations::helpers::callbacks;
using namespace operations::exceptions;


void print_progress(double percentage, const char *str = nullptr) {
    int val = (int) (percentage * 100);
    int left_pad = (int) (percentage * PB_WIDTH);
    int right_pad = PB_WIDTH - left_pad;
    printf("\r%s\t%3d%% [%.*s%*s]", str, val, left_pad, PB_STR, right_pad, "");
    fflush(stdout);
}

using namespace std;

RTMEngine::RTMEngine(RTMEngineConfigurations *apConfiguration,
                     ComputationParameters *apParameters) {
    this->mpConfiguration = apConfiguration;
    this->mpParameters = apParameters;
    this->mpCallbacks = new CallbackCollection();
    this->mpTimer = Timer::GetInstance();
    m_dReadIOTime = 0.0;
}

RTMEngine::RTMEngine(RTMEngineConfigurations *apConfiguration,
                     ComputationParameters *apParameters,
                     helpers::callbacks::CallbackCollection *apCallbackCollection) {
    this->mpConfiguration = apConfiguration;
    this->mpParameters = apParameters;
    this->mpCallbacks = apCallbackCollection;
    this->mpTimer = Timer::GetInstance();
    m_dReadIOTime = 0.0;
}

RTMEngine::~RTMEngine() = default;

vector<uint> RTMEngine::GetValidShots() {
    cout << "Detecting available shots for processing..." << std::endl;
#ifdef ENABLE_GPU_TIMINGS
    Timer *timer = Timer::GetInstance();
    timer->StartTimer("Engine::GetValidShots");
#endif
    vector<uint> possible_shots =
            this->mpConfiguration->GetTraceManager()->GetWorkingShots(
                    this->mpConfiguration->GetTraceFiles(),
                    this->mpConfiguration->GetSortMin(),
                    this->mpConfiguration->GetSortMax(),
                    this->mpConfiguration->GetSortKey());
#ifdef ENABLE_GPU_TIMINGS
    timer->StopTimer("Engine::GetValidShots");
#endif
    if (possible_shots.empty()) {
        cout << "No valid shots detected... terminating." << std::endl;
        exit(EXIT_FAILURE);
    }
    cout << "Valid shots detected to process\t: "
         << possible_shots.size() << std::endl;
    return possible_shots;
}

GridBox *RTMEngine::Initialize() {
#ifdef ENABLE_GPU_TIMINGS
    this->mpTimer->StartTimer("Engine::Initialization");
#endif

#ifndef NDEBUG
    this->mpCallbacks->BeforeInitialization(this->mpParameters);
#endif

    /// Set computation parameters to all components with
    /// parameters given to the constructor for all needed functions.
    for (auto const &component : this->mpConfiguration->GetComponents()->ExtractValues()) {
        component->SetComputationParameters(this->mpParameters);
    }

    /// Set computation parameters to all dependent components with
    /// parameters given to the constructor for all needed functions.
    for (auto const &dependent_component : this->mpConfiguration->GetDependentComponents()->ExtractValues()) {
        dependent_component->SetComputationParameters(this->mpParameters);
    }

    /// Set dependent components to all components with for all
    /// needed functions.
    for (auto const &component : this->mpConfiguration->GetComponents()->ExtractValues()) {
        component->SetDependentComponents(this->mpConfiguration->GetDependentComponents());
    }

    /// Set Components Map to all components.
    for (auto const &component : this->mpConfiguration->GetComponents()->ExtractValues()) {
        component->SetComponentsMap(this->mpConfiguration->GetComponents());
    }

    /// Acquire Configuration to all components with
    /// parameters given to the constructor for all needed functions.
    for (auto const &component : this->mpConfiguration->GetComponents()->ExtractValues()) {
        component->AcquireConfiguration();
    }

#ifdef ENABLE_GPU_TIMINGS
    this->mpTimer->StartTimer("ModelHandler::ReadModel");
#endif
    GridBox *grid_box = this->mpConfiguration->GetModelHandler()->ReadModel(this->mpConfiguration->GetModelFiles());
#ifdef ENABLE_GPU_TIMINGS
    this->mpTimer->StopTimer("ModelHandler::ReadModel");
#endif

    /// Set the GridBox with the parameters given to the constructor for
    /// all needed functions.
    for (auto const &component : this->mpConfiguration->GetComponents()->ExtractValues()) {
        component->SetGridBox(grid_box);
    }
#ifdef ENABLE_GPU_TIMINGS
    this->mpTimer->StartTimer("ModelHandler::PreprocessModel");
#endif
    this->mpConfiguration->GetModelHandler()->PreprocessModel();

#ifdef ENABLE_GPU_TIMINGS
    this->mpTimer->StopTimer("ModelHandler::PreprocessModel");
    this->mpTimer->StartTimer("BoundaryManager::ExtendModel");
#endif
    this->mpConfiguration->GetBoundaryManager()->ExtendModel();

#ifdef ENABLE_GPU_TIMINGS    
    this->mpTimer->StopTimer("BoundaryManager::ExtendModel");
#endif

#ifndef NDEBUG
    this->mpCallbacks->AfterInitialization(grid_box);
#endif
    this->mpConfiguration->GetComputationKernel()->SetBoundaryManager(this->mpConfiguration->GetBoundaryManager());
#ifdef ENABLE_GPU_TIMINGS
    this->mpTimer->StopTimer("Engine::Initialization");
#endif

    grid_box->Report(VERBOSE);
    return grid_box;
}

void RTMEngine::MigrateShots(vector<uint> shot_numbers, GridBox *apGridBox) {
#ifdef ENABLE_GPU_TIMINGS
    this->mpTimer->AddTimer("Engine::MigrateShot");
    this->mpTimer->AddRunTimeEntryToTimer("Engine::MigrateShot", this->mpParameters->GetCollectedQueueCreationTime());
#endif
    for (int shot_number:shot_numbers) {
        MigrateShots(shot_number, apGridBox);
    }
}

void RTMEngine::MigrateShots(uint shot_id, GridBox *apGridBox) 
{
#ifdef ENABLE_GPU_TIMINGS
    this->mpTimer->StartTimer("TraceManager::ReadShot");
#endif
    std::chrono::steady_clock::time_point const tpReadShotStart(std::chrono::steady_clock::now());
    this->mpConfiguration->GetTraceManager()->ReadShot(this->mpConfiguration->GetTraceFiles(), shot_id, this->mpConfiguration->GetSortKey()); //[JT>>:] Reads I/O files
    std::chrono::steady_clock::time_point const tpReadShotEnd(std::chrono::steady_clock::now());
    m_dReadIOTime += std::chrono::duration<double>(tpReadShotEnd - tpReadShotStart).count();
#ifdef ENABLE_GPU_TIMINGS
    this->mpTimer->StopTimer("TraceManager::ReadShot");
    this->mpTimer->StartTimer("Engine::MigrateShot");
#endif
    this->mpConfiguration->GetMigrationAccommodator()->ResetShotCorrelation(); //[JT>>:] GPU Device::Memset takes place

#ifndef NDEBUG
    this->mpCallbacks->BeforeShotPreprocessing(this->mpConfiguration->GetTraceManager()->GetTracesHolder());
#endif

#ifdef ENABLE_GPU_TIMINGS
    this->mpTimer->StartTimer("TraceManager::PreprocessShot");
#endif
    this->mpConfiguration->GetTraceManager()->PreprocessShot(this->mpConfiguration->GetSourceInjector()->GetCutOffTimeStep()); //[JT>>:] GPU Device memory allocation and memcpy 

#ifdef ENABLE_GPU_TIMINGS
    this->mpTimer->StopTimer("TraceManager::PreprocessShot");
#endif

#ifndef NDEBUG
    this->mpCallbacks->AfterShotPreprocessing(
            this->mpConfiguration->GetTraceManager()->GetTracesHolder());
#endif

    this->mpConfiguration->GetSourceInjector()->SetSourcePoint(this->mpConfiguration->GetTraceManager()->GetSourcePoint()); // [JT>>:] Blank...
#ifdef ENABLE_GPU_TIMINGS
    this->mpTimer->StartTimer("ModelHandler::SetupWindow");
#endif
    this->mpConfiguration->GetModelHandler()->SetupWindow(); // [JT>>:] Launches setupwindow kernel
#ifdef ENABLE_GPU_TIMINGS
    this->mpTimer->StopTimer("ModelHandler::SetupWindow");
#endif

#ifdef ENABLE_GPU_TIMINGS
    this->mpTimer->StartTimer("BoundaryManager::ReExtendModel");
#endif
    this->mpConfiguration->GetBoundaryManager()->ReExtendModel(); //[JT>>:] No GPU kernel or GPU activity
#ifdef ENABLE_GPU_TIMINGS
    this->mpTimer->StopTimer("BoundaryManager::ReExtendModel");
    this->mpTimer->StartTimer("ForwardCollector::ResetGrid(Forward)");
#endif
    this->mpConfiguration->GetForwardCollector()->ResetGrid(true); // [JT>>:] GPU device memset takes place here
#ifdef ENABLE_GPU_TIMINGS                                                                  
    this->mpTimer->StopTimer("ForwardCollector::ResetGrid(Forward)");
#endif 
#ifndef NDEBUG
    this->mpCallbacks->BeforeForwardPropagation(apGridBox);
#endif

    this->Forward(apGridBox); // [JT>>:] GPU stencil kernel launches here, Step (Compute), SaveForward (GPU Device memory), ApplySources (Compute)

#ifdef ENABLE_GPU_TIMINGS
    this->mpTimer->StartTimer("ForwardCollector::ResetGrid(Backward)");
#endif
    this->mpConfiguration->GetForwardCollector()->ResetGrid(false); // [JT>>:] GPU device memset takes place here

#ifdef ENABLE_GPU_TIMINGS
    this->mpTimer->StopTimer("ForwardCollector::ResetGrid(Backward)");
    this->mpTimer->StartTimer("BoundaryManager::AdjustModelForBackward");
#endif    
    this->mpConfiguration->GetBoundaryManager()->AdjustModelForBackward(); // [JT>>:] No GPU kernel or GPU activity

#ifdef ENABLE_GPU_TIMINGS
    this->mpTimer->StopTimer("BoundaryManager::AdjustModelForBackward");
#endif

#ifndef NDEBUG
    this->mpCallbacks->BeforeBackwardPropagation(apGridBox);
#endif

    this->Backward(apGridBox); // [JT>>:] GPU stencil kernel launches here
#ifndef NDEBUG
    this->mpCallbacks->BeforeShotStacking(apGridBox, this->mpConfiguration->GetMigrationAccommodator()->GetShotCorrelation());
#endif

#ifdef ENABLE_GPU_TIMINGS
    this->mpTimer->StartTimer("CorrelationKernel::Stack");
#endif    
    this->mpConfiguration->GetMigrationAccommodator()->Stack(); // [JT>>:] Cross correlation kernel here

#ifdef ENABLE_GPU_TIMINGS
    this->mpTimer->StopTimer("CorrelationKernel::Stack");
#endif

#ifndef NDEBUG
    this->mpCallbacks->AfterShotStacking(apGridBox, this->mpConfiguration->GetMigrationAccommodator()->GetStackedShotCorrelation());
#endif

#ifdef ENABLE_GPU_TIMINGS
    this->mpTimer->StopTimer("Engine::MigrateShot");
#endif
}

MigrationData *RTMEngine::Finalize(GridBox *apGridBox) {
#ifndef NDEBUG
    this->mpCallbacks->AfterMigration(apGridBox, this->mpConfiguration->GetMigrationAccommodator()->GetStackedShotCorrelation());
#endif

#ifdef ENABLE_GPU_TIMINGS
    this->mpTimer->StartTimer("CorrelationKernel::GetMigrationData");
#endif
    MigrationData *migration = this->mpConfiguration->GetMigrationAccommodator()->GetMigrationData();
#ifdef ENABLE_GPU_TIMINGS
    this->mpTimer->StopTimer("CorrelationKernel::GetMigrationData");
#endif

    delete apGridBox;
    return migration;
}

void RTMEngine::Forward(GridBox *apGridBox) 
{
#ifdef ENABLE_GPU_TIMINGS
    this->mpTimer->StartTimer("Engine::Forward");
#endif
    uint onePercent = apGridBox->GetNT() / 100 + 1;
    for (uint t = 1; t < apGridBox->GetNT(); t++) {
#ifdef ENABLE_GPU_TIMINGS
        this->mpTimer->StartTimer("ForwardCollector::SaveForward");
#endif
        this->mpConfiguration->GetForwardCollector()->SaveForward();
#ifdef ENABLE_GPU_TIMINGS
        this->mpTimer->StopTimer("ForwardCollector::SaveForward");
        this->mpTimer->StartTimer("SourceInjector::ApplySource");
#endif
        this->mpConfiguration->GetSourceInjector()->ApplySource(t);
#ifdef ENABLE_GPU_TIMINGS
        this->mpTimer->StopTimer("SourceInjector::ApplySource");
        this->mpTimer->StartTimer("Forward::ComputationKernel::Step");
#endif
        this->mpConfiguration->GetComputationKernel()->Step();
#ifdef ENABLE_GPU_TIMINGS
        this->mpTimer->StopTimer("Forward::ComputationKernel::Step");
#endif
#ifndef NDEBUG
        this->mpCallbacks->AfterForwardStep(apGridBox, t);
#endif
        if ((t % onePercent) == 0) {
            print_progress(((float) t) / apGridBox->GetNT(), "Forward Propagation");
        }
    }
    print_progress(1, "Forward Propagation");
    cout << " ... Done" << endl;
#ifdef ENABLE_GPU_TIMINGS
    this->mpTimer->StopTimer("Engine::Forward");
#endif
}

void RTMEngine::Backward(GridBox *apGridBox) {
#ifdef ENABLE_GPU_TIMINGS
    this->mpTimer->StartTimer("Engine::Backward");
#endif
    uint onePercent = apGridBox->GetNT() / 100 + 1;
    for (uint t = apGridBox->GetNT() - 1; t > 0; t--) {
#ifdef ENABLE_GPU_TIMINGS
        this->mpTimer->StartTimer("TraceManager::ApplyTraces");
#endif
        this->mpConfiguration->GetTraceManager()->ApplyTraces(t);
#ifdef ENABLE_GPU_TIMINGS
        this->mpTimer->StopTimer("TraceManager::ApplyTraces");
        this->mpTimer->StartTimer("Backward::ComputationKernel::Step");
#endif
        this->mpConfiguration->GetComputationKernel()->Step();

#ifdef ENABLE_GPU_TIMINGS
        this->mpTimer->StopTimer("Backward::ComputationKernel::Step");
        this->mpTimer->StartTimer("ForwardCollector::FetchForward");
#endif
        this->mpConfiguration->GetForwardCollector()->FetchForward();
#ifdef ENABLE_GPU_TIMINGS
        this->mpTimer->StopTimer("ForwardCollector::FetchForward");
#endif
#ifndef NDEBUG
        this->mpCallbacks->AfterFetchStep(this->mpConfiguration->GetForwardCollector()->GetForwardGrid(), t);
        this->mpCallbacks->AfterBackwardStep(apGridBox, t);
#endif
#ifdef ENABLE_GPU_TIMINGS
        this->mpTimer->StartTimer("CorrelationKernel::Correlate");
#endif
        this->mpConfiguration->GetMigrationAccommodator()->Correlate(this->mpConfiguration->GetForwardCollector()->GetForwardGrid());
#ifdef ENABLE_GPU_TIMINGS
        this->mpTimer->StopTimer("CorrelationKernel::Correlate");
#endif
        if ((t % onePercent) == 0) {
            print_progress(((float) (apGridBox->GetNT() - t)) / apGridBox->GetNT(), "Backward Propagation");
        }
    }
    print_progress(1, "Backward Propagation");

    cout << " ... Done" << endl;

#ifdef ENABLE_GPU_TIMINGS 
    this->mpTimer->StopTimer("Engine::Backward");
#endif
}

