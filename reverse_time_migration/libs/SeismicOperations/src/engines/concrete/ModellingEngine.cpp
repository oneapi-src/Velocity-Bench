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
// Created by amr-nasr on 12/11/2019.
//

#include "operations/engines/concrete/ModellingEngine.hpp"

#include <memory-manager/MemoryManager.h>

/// { @todo To be removed when all NotImplementedException() are resolved
#include "operations/exceptions/NotImplementedException.h"
/// }

#define PB_STR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PB_WIDTH 50

using namespace std;
using namespace operations::engines;
using namespace operations::common;
using namespace operations::dataunits;
using namespace operations::configuration;
using namespace operations::helpers::callbacks;
using namespace operations::exceptions;

void print_modelling_progress(double percentage, const char *str = nullptr) {
    int val = (int) (percentage * 100);
    int left_pad = (int) (percentage * PB_WIDTH);
    int right_pad = PB_WIDTH - left_pad;
    printf("\r%s\t%3d%% [%.*s%*s]", str, val, left_pad, PB_STR, right_pad, "");
    fflush(stdout);
}

/*!
 * GetInstance():This is the so-called Singleton design pattern.
 * Its distinguishing feature is that there can only ever be exactly one
 * instance of that class and the pattern ensures that. The class has a private
 * constructor and a statically-created instance that is returned with the
 * GetInstance method. You cannot create an instance from the outside and thus
 * get the object only through said method.Since instance is static in the
 * GetInstance method it will retain its value between multiple invocations. It
 * is allocated and constructed some where before it's first used. E.g. in this
 * answer it seems like GCC initializes the static variable at the time the
 * function is first used.
 */

/*!
 * Constructor to start the modelling engine given the appropriate engine
 * configuration.
 * @param apConfiguration
 * The configuration which will control the actual work of the engine.
 * @param apParameters
 * The computation parameters that will control the simulations settings like
 * boundary length, order of numerical solution.
 */
ModellingEngine::ModellingEngine(ModellingEngineConfigurations *apConfiguration,
                                 ComputationParameters *apParameters) {
    this->mpConfiguration = apConfiguration;
    /**
     * Callbacks are like hooks that you can register function to be called in
     * specific events(etc. after the forward propagation, after each time step,
     * and so on) this enables the user to add functionalities,take snapshots
     * or track state in different approaches. A callback is the class actually
     * implementing the functions to capture or track state like norm, images, csv
     * and so on. They are registered to a callback collection which is what links
     * them to the engine. The engine would inform the callback collection at the
     * time of each event, and the callback collection would then inform each
     * callback registered to it.
     */
    this->mpCallbacks = new CallbackCollection();
    this->mpParameters = apParameters;
    this->mpTimer = Timer::GetInstance();
}

/**
 * Constructor to start the modelling engine given the appropriate engine
 * configuration.
 * @param apConfiguration
 * The configuration which will control the actual work of the engine.
 * @param apParameters
 * The computation parameters that will control the simulations settings like
 * boundary length, order of numerical solution.
 * @param apCallbackCollection
 * The callbacks registered to be called in the right time.
 */
ModellingEngine::ModellingEngine(ModellingEngineConfigurations *apConfiguration,
                                 ComputationParameters *apParameters,
                                 helpers::callbacks::CallbackCollection *apCallbackCollection) {
    /// Modeling configuration of the class is that given as
    /// argument to the constructor
    this->mpConfiguration = apConfiguration;
    /// Callback of the class is that given as argument to the constructor
    this->mpCallbacks = apCallbackCollection;
    /// Computation parameters of this class is that given as
    /// argument to the constructor.
    this->mpParameters = apParameters;

    this->mpTimer = Timer::GetInstance();
}

ModellingEngine::~ModellingEngine()
{
    delete mpConfiguration;
    delete mpCallbacks;
    delete mpParameters;

}

/**
 * Run the initialization steps for the modelling engine.
 */
GridBox *ModellingEngine::Initialize() {
    this->mpTimer->StartTimer("Engine::Initialization");
#ifndef NDEBUG
    this->mpCallbacks->BeforeInitialization(this->mpParameters);
#endif

    /// Set computation parameters to all components with
    /// parameters given to the constructor for all needed functions.
    for (auto component :
            this->mpConfiguration->GetComponents()->ExtractValues()) {
        component->SetComputationParameters(this->mpParameters);
    }

    /// Set computation parameters to all dependent components with
    /// parameters given to the constructor for all needed functions.
    for (auto const &dependent_component :
            this->mpConfiguration->GetDependentComponents()->ExtractValues()) {
        dependent_component->SetComputationParameters(this->mpParameters);
    }

    /// Set dependent components to all components with for all
    /// needed functions.
    for (auto const &component :
            this->mpConfiguration->GetComponents()->ExtractValues()) {
        component->SetDependentComponents(
                this->mpConfiguration->GetDependentComponents());
    }

    /// Set Components Map to all components.
    for (auto const &component :
            this->mpConfiguration->GetComponents()->ExtractValues()) {
        component->SetComponentsMap(this->mpConfiguration->GetComponents());
    }

    /// Acquire Configuration to all components with
    /// parameters given to the constructor for all needed functions.
    for (auto const &component :
            this->mpConfiguration->GetComponents()->ExtractValues()) {
        component->AcquireConfiguration();
    }

    this->mpTimer->StartTimer("ModelHandler::ReadModel");
    GridBox *grid_box =
            this->mpConfiguration->GetModelHandler()->ReadModel(
                    this->mpConfiguration->GetModelFiles());
    this->mpTimer->StopTimer("ModelHandler::ReadModel");

    /// Set the GridBox with the parameters given to the constructor for
    /// all needed functions.
    for (auto component :
            this->mpConfiguration->GetComponents()->ExtractValues()) {
        component->SetGridBox(grid_box);
    }

    /**
     * All pre-processing needed to be done on the model before the beginning of
     * the reverse time migration, should be applied in this function.
     */
    this->mpTimer->StartTimer("ModelHandler::PreprocessModel");
    this->mpConfiguration->GetModelHandler()->PreprocessModel();
    this->mpTimer->StopTimer("ModelHandler::PreprocessModel");

    /**
     * Extends the velocities/densities to the added boundary parts to the
     * velocity/density of the model appropriately. This is only called once after
     * the initialization of the model.
     */
    this->mpTimer->StartTimer("BoundaryManager::ExtendModel");
    this->mpConfiguration->GetBoundaryManager()->ExtendModel();
    this->mpTimer->StopTimer("BoundaryManager::ExtendModel");

#ifndef NDEBUG
    this->mpCallbacks->AfterInitialization(grid_box);
#endif

    /**
     * This function is used to parse the modelling_configuration_file to get the
     * parameters of ModellingConfiguration struct (parameters of modeling)
     * Parses a file with the proper format as the modelling configuration.
     */
    ModellingConfiguration model_conf =
            this->mpConfiguration->GetModellingConfigurationParser()->ParseConfiguration(
                    this->mpConfiguration->GetModellingConfigurationFile(),
                    grid_box->GetActualGridSize(Y_AXIS) == 1);

    /// Getting the number of time steps for the grid from the model_conf struct
    grid_box->SetNT(int(model_conf.TotalTime / grid_box->GetDT()));

    /**
     * Initializes the trace writer with all needed data for it
     * to be able to start recording traces according the the given configuration.
     */
    this->mpConfiguration->GetTraceWriter()->InitializeWriter(
            &model_conf, this->mpConfiguration->GetTraceFiles());

    /// After updating the model_conf struct by the data exist in the
    /// modelling_configuration_file put its value in the modelling_configuration
    /// struct of this class
    this->mpModellingConfiguration = model_conf;

    this->mpConfiguration->GetComputationKernel()->SetBoundaryManager(
            this->mpConfiguration->GetBoundaryManager());
    this->mpTimer->StopTimer("Engine::Initialization");

    grid_box->Report(VERBOSE);
    return grid_box;
}

vector<uint> ModellingEngine::GetValidShots() {
    /// @todo To be enhanced to handle more than one shot per run.
    vector<uint> shots = {0};
    return shots;
}

void ModellingEngine::MigrateShots(vector<uint> aShotNumbers, GridBox *apGridBox) {
    this->mpTimer->StartTimer("Engine::Model");
    /// If not in the debug mode(release mode) call the ReExtendModel() of
    /// this class
    /*!
     * Extends the velocities/densities to the added boundary parts to the
     * velocity/density of the model appropriately. This is called repeatedly with
     * before the forward propagation of each shot.
     */
    this->mpConfiguration->GetBoundaryManager()->ReExtendModel();
#ifndef NDEBUG
    /// If in the debug mode use the call back of BeforeForwardPropagation and
    /// give it our updated GridBox
    this->mpCallbacks->BeforeForwardPropagation(apGridBox);
#endif
    /// If not in the debug mode(release mode) call the Forward function of
    /// this class and give it our updated GridBox
    /*!
     * Begin the forward propagation and recording of the traces.
     */
    this->Forward(apGridBox, aShotNumbers);
    mem_free(apGridBox);
    this->mpTimer->StopTimer("Engine::Model");
}

MigrationData *ModellingEngine::Finalize(GridBox *apGridBox) {
    return nullptr;
}

void ModellingEngine::Forward(GridBox *apGridBox, vector<uint> const &aShotNumbers) {
    this->mpTimer->StartTimer("Engine::Forward");

    /**
     * This function is used to set the source point for the source injector
     * function with the source_point that exist in the modelling_configuration of
     * this class Sets the source point to apply the injection to it.
     */
    this->mpConfiguration->GetSourceInjector()->SetSourcePoint(
            &this->mpModellingConfiguration.SourcePoint);
    this->mpTimer->StartTimer("ModelHandler::SetupWindow");
    this->mpConfiguration->GetModelHandler()->SetupWindow();
    this->mpTimer->StopTimer("ModelHandler::SetupWindow");
    this->mpTimer->StartTimer("BoundaryManager::ReExtendModel");
    this->mpConfiguration->GetBoundaryManager()->ReExtendModel();
    this->mpTimer->StopTimer("BoundaryManager::ReExtendModel");
    cout << "Forward Propagation" << endl;
    int onePercent = apGridBox->GetNT() / 100 + 1;
    for (uint t = 1; t < apGridBox->GetNT(); t++) {
        /**
         * Function to apply source injection to the wave field(s). It should inject
         * the current frame in our grid with the appropriate value. It should be
         * responsible of controlling when to stop the injection.
         */
        this->mpConfiguration->GetSourceInjector()->ApplySource(t);

        /**
         * This function should solve the wave equation. It calculates the next
         * time-step from previous and current frames. It should also update the
         * GridBox structure so that after the function call, the GridBox structure
         * current frame should point to the newly calculated result, the previous
         * frame should point to the current frame at the time of the function call.
         * The next frame should point to the previous frame at the time of the
         * function call.
         */
        this->mpConfiguration->GetComputationKernel()->Step();

#ifndef NDEBUG
        /// If in the debug mode use the call back of AfterForwardStep and give
        /// it the updated gridBox
        this->mpCallbacks->AfterForwardStep(apGridBox, t);
#endif
        /// If not in the debug mode (release mode) we use call the RecordTrace()
        /**
         * Records the traces from the domain according to the configuration given
         * in the initialize function.
         */
        this->mpConfiguration->GetTraceWriter()->RecordTrace();
        if ((t % onePercent) == 0) {
            print_modelling_progress(((float) t) / apGridBox->GetNT(), "Forward Propagation");
        }
    }
    print_modelling_progress(1, "Forward Propagation");
    cout << " ... Done" << endl;
    this->mpTimer->StopTimer("Engine::Forward");
}
