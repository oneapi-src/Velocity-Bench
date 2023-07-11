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
// Created by zeyad-osama on 20/09/2020.
//

#ifndef OPERATIONS_LIB_MODELLING_ENGINE_CONFIGURATION_HPP
#define OPERATIONS_LIB_MODELLING_ENGINE_CONFIGURATION_HPP

#include <operations/engine-configurations/interface/EngineConfigurations.hpp>

#include <operations/components/independents/primitive/BoundaryManager.hpp>
#include <operations/components/independents/primitive/ComputationKernel.hpp>
#include <operations/components/independents/primitive/ModelHandler.hpp>
#include <operations/components/independents/primitive/ModellingConfigurationParser.hpp>
#include <operations/components/independents/primitive/TraceWriter.hpp>
#include <operations/components/independents/primitive/SourceInjector.hpp>
#include <operations/components/independents/primitive/TraceManager.hpp>

#include <map>

namespace operations {
    namespace configuration {

/**
 * @example Example of modeling(one shot)
 * in the Modeling by giving it a velocity file in (model_files) and by using
 * the parameters come from the parsing of (modelling_configuration_file) ,we
 * do only the forward propagation using the concrete components of RTM
 * defined here and generate the pressure also Record at each time step the
 * pressure at the surface in the places we want to have receivers on using
 * (trace_writer) and put the output in (trace_file)
 */

/**
 * @example Example of modeling(different shot)
 * the same for one shot except for the generation of the the different traces
 * files,they can be generated in two ways 1-run the acoustic_modeller
 * different times each with different shot location and use different trace
 * file name for each shot/run 2-run the script in data/run_modeller.sh and
 * give it the range you want to start(inclusive),end(exclusive) and increment
 * the shots by in the (i) iterator and it will automatically runs the
 * acoustic modeler different times for each shot location and generates
 * output trace files named (shot_i.trace) where (i)(iterator) represents the
 * shot location in x
 */

        /**
         * @brief Class that contains pointers to concrete
         * implementations of some of the components of RTM framework
         * to be used in modelling engine
         *
         * @note ModellingEngineConfiguration is only used for modelling
         * and it doesn't have a forward collector
         * Because we don't need to store the forward propagation.
         * If we are modeling then we only do forward and store the traces while
         * propagating using the (TraceWriter) component we store the traces for each
         * shot in the trace file.(each shot's traces  in a different file)
         */
        class ModellingEngineConfigurations : public EngineConfigurations {
        public:
            /**
             * @brief Constructor
             **/
            ModellingEngineConfigurations() = default;

            /**
             * @brief Destructor for correct destroying of the pointers
             **/
            ~ModellingEngineConfigurations() override {
                delete mpModelHandler;
                delete mpBoundaryManager;
                delete mpSourceInjector;
                delete mpComputationKernel;
                delete mpTraceWriter;
                delete mpModellingConfigurationParser;

                delete mpMemoryHandler;
            }

            inline components::ModelHandler *GetModelHandler() const {
                return this->mpModelHandler;
            }

            void SetModelHandler(components::ModelHandler *apModelHandler) {
                this->mpModelHandler = apModelHandler;
                this->mpComponentsMap->Set(MODEL_HANDLER, this->mpModelHandler);
            }

            inline components::BoundaryManager *GetBoundaryManager() const {
                return this->mpBoundaryManager;
            }

            void SetBoundaryManager(components::BoundaryManager *apBoundaryManager) {
                this->mpBoundaryManager = apBoundaryManager;
                this->mpComponentsMap->Set(BOUNDARY_MANAGER, this->mpBoundaryManager);
            }

            inline components::SourceInjector *GetSourceInjector() const {
                return this->mpSourceInjector;
            }

            void SetSourceInjector(components::SourceInjector *apSourceInjector) {
                this->mpSourceInjector = apSourceInjector;
                this->mpComponentsMap->Set(SOURCE_INJECTOR, this->mpSourceInjector);
            }

            inline components::ComputationKernel *GetComputationKernel() const {
                return this->mpComputationKernel;
            }

            void SetComputationKernel(components::ComputationKernel *apComputationKernel) {
                this->mpComputationKernel = apComputationKernel;
                this->mpComponentsMap->Set(COMPUTATION_KERNEL, this->mpComputationKernel);

                /// Set Memory Handler accordingly.
                this->SetMemoryHandler(this->mpComputationKernel->GetMemoryHandler());
            }

            inline components::ModellingConfigurationParser *GetModellingConfigurationParser() const {
                return this->mpModellingConfigurationParser;
            }

            void
            SetModellingConfigurationParser(components::ModellingConfigurationParser *apModellingConfigurationParser) {
                this->mpModellingConfigurationParser = apModellingConfigurationParser;
                this->mpComponentsMap->Set(MODELLING_CONFIG_PARSER, this->mpModellingConfigurationParser);
            }

            inline components::TraceWriter *GetTraceWriter() const {
                return this->mpTraceWriter;
            }

            inline void SetTraceWriter(components::TraceWriter *apTraceWriter) {
                this->mpTraceWriter = apTraceWriter;
                this->mpComponentsMap->Set(TRACE_WRITER, this->mpTraceWriter);
            }

            inline const std::map<std::string, std::string> &GetModelFiles() const {
                return this->mModelFiles;
            }

            inline void SetModelFiles(const std::map<std::string, std::string> &aModelFiles) {
                this->mModelFiles = aModelFiles;
            }

            inline const std::string &GetTraceFiles() const {
                return this->mTraceFiles;
            }

            inline void SetTraceFiles(const std::string &aTraceFiles) {
                this->mTraceFiles = aTraceFiles;
            }

            inline const std::string &GetModellingConfigurationFile() const {
                return this->mModellingConfigurationFile;
            }

            inline void SetModellingConfigurationFile(const std::string &aModellingConfigurationFile) {
                this->mModellingConfigurationFile = aModellingConfigurationFile;
            }

        private:
            void SetMemoryHandler(components::MemoryHandler *apMemoryHandler) {
                this->mpMemoryHandler = apMemoryHandler;
                this->mpDependentComponentsMap->Set(MEMORY_HANDLER, this->mpMemoryHandler);
            }

        private:
            /* Independent Components */

            components::ModelHandler *mpModelHandler;
            components::BoundaryManager *mpBoundaryManager;
            components::SourceInjector *mpSourceInjector;
            components::ComputationKernel *mpComputationKernel;
            components::TraceWriter *mpTraceWriter;
            components::ModellingConfigurationParser *mpModellingConfigurationParser;

            /* Dependent Components */

            components::MemoryHandler *mpMemoryHandler;

            /// All model (i.e. parameters) files.
            /// @example Velocity, density, epsilon and delta phi and theta
            std::map<std::string, std::string> mModelFiles;
            /// Traces files are different files each file contains
            /// the traces for one shot that may be output from the
            /// modeling engine
            std::string mTraceFiles;
            /// File used to parse the modeling configuration parameters from
            std::string mModellingConfigurationFile;
        };
    } //namespace configuration
} //namespace operations

#endif //OPERATIONS_LIB_MODELLING_ENGINE_CONFIGURATION_HPP
