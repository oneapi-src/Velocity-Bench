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
// Created by mirna-moawad on 22/10/2019.
//

#ifndef OPERATIONS_LIB_COMPONENTS_MODEL_HANDLERS_HOMOGENOUS_MODEL_HANDLER_HPP
#define OPERATIONS_LIB_COMPONENTS_MODEL_HANDLERS_HOMOGENOUS_MODEL_HANDLER_HPP

#include <operations/components/dependents/concrete/memory-handlers/WaveFieldsMemoryHandler.hpp>
#include <operations/components/independents/primitive/ModelHandler.hpp>
#include <operations/components/dependency/concrete/HasDependents.hpp>

#include <memory-manager/MemoryManager.h>

namespace operations {
    namespace components {

        class SyntheticModelHandler : public ModelHandler,
                                      public dependency::HasDependents {
        public:
            explicit SyntheticModelHandler(operations::configuration::ConfigurationMap *apConfigurationMap);

            ~SyntheticModelHandler() override;

            dataunits::GridBox *ReadModel(std::map<std::string, std::string> const &file_names) override;

            void PreprocessModel() override;

            void SetComputationParameters(common::ComputationParameters *apParameters) override;

            void SetDependentComponents(
                    operations::helpers::ComponentsMap<DependentComponent> *apDependentComponentsMap) override;

            void SetGridBox(dataunits::GridBox *apGridBox) override;

            void SetupWindow() override;

            void AcquireConfiguration() override;

        private:
            float GetSuitableDT(float *coefficients, std::map<std::string, float> maximums,
                                int half_length, float dt_relax);

            void Initialize(std::map<std::string, std::string> const &file_names);

            void RegisterWaveFields(uint nx, uint ny, uint nz);

            void RegisterParameters(uint nx, uint ny, uint nz);

            void SetupPadding();

            void AllocateWaveFields();

            void AllocateParameters();

            float SetModelField(float *field, std::vector<float> &model_file,
                                int nx, int nz, int ny,
                                int logical_nx, int logical_nz, int logical_ny);


        private:
            common::ComputationParameters *mpParameters = nullptr;

            dataunits::GridBox *mpGridBox = nullptr;

            WaveFieldsMemoryHandler *mpWaveFieldsMemoryHandler = nullptr;

            std::vector<std::vector<float>> mModelFile;

            std::vector<std::pair<dataunits::GridBox::Key, std::string>> PARAMS_NAMES;

            std::vector<dataunits::GridBox::Key> WAVE_FIELDS_NAMES;
            
            SyntheticModelHandler           (SyntheticModelHandler const &RHS) = delete;
            SyntheticModelHandler &operator=(SyntheticModelHandler const &RHS) = delete;
        };
    }//namespace components
}//namespace operations

#endif // OPERATIONS_LIB_COMPONENTS_MODEL_HANDLERS_HOMOGENOUS_MODEL_HANDLER_HPP
