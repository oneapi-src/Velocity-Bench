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

#ifndef OPERATIONS_LIB_DATA_UNITS_MIGRATION_MIGRATION_DATA_HPP
#define OPERATIONS_LIB_DATA_UNITS_MIGRATION_MIGRATION_DATA_HPP

#include "operations/data-units/interface/DataUnit.hpp"
#include "Result.hpp"
#include "operations/common/DataTypes.h"

#include "operations/exceptions/Exceptions.h"

#include <utility>
#include <vector>

namespace operations {
    namespace dataunits {

        class MigrationData : public DataUnit {
        public:
            /**
             * @brief Constructor
             * @param[in] mig_type
             * @param[in] nx
             * @param[in] nz
             * @param[in] ny
             * @param[in] nt
             * @param[in] dx
             * @param[in] dz
             * @param[in] dy
             * @param[in] dt
             * @param[in] avResults
             */
            MigrationData(uint nx, uint ny, uint nz, uint nt,
                          float dx, float dy, float dz, float dt,
                          std::vector<Result *> avResults) :
                    MigrationData(nx, ny, nz, nt, 1, dx, dy, dz, dt, std::move(avResults)) {
            }

            MigrationData(uint nx, uint ny, uint nz, uint nt,
                          uint gather_dimension,
                          float dx, float dy, float dz, float dt,
                          std::vector<Result *> avResults) {
                this->mNX = nx;
                this->mNY = ny;
                this->mNZ = nz;
                this->mNT = nt;

                this->mDX = dx;
                this->mDY = dy;
                this->mDZ = dz;
                this->mDT = dt;

                this->mvResults = std::move(avResults);

                this->mGatherDimension = gather_dimension;
            }

            /**
             * @brief Destructor.
             */
            ~MigrationData() override {
                for (auto const &result : this->mvResults) {
                    delete result;
                }
            }

            /**
             * @brief NT getter.
             */
            inline uint GetNT() const {
                return this->mNT;
            }

            /**
             * @brief DT getter.
             */
            inline float GetDT() const {
                return this->mDT;
            }

            /**
             * @brief Grid size per axe getter.
             * @param[in] axis      Axe direction
             * @return[out] value   Value
             */
            uint GetGridSize(uint axis) const {
                if (exceptions::is_out_of_range(axis)) {
                    throw exceptions::AxisException();
                }

                uint val;
                if (axis == Y_AXIS) {
                    val = this->mNY;
                } else if (axis == Z_AXIS) {
                    val = this->mNZ;
                } else if (axis == X_AXIS) {
                    val = this->mNX;
                }
                return val;
            }

            /**
             * @brief Cell dimensions per axe getter.
             * @param[in] axis      Axe direction
             * @return[out] value   Value
             */
            float GetCellDimensions(uint axis) const {
                if (exceptions::is_out_of_range(axis)) {
                    throw exceptions::AxisException();
                }

                float val;
                if (axis == Y_AXIS) {
                    val = this->mDY;
                } else if (axis == Z_AXIS) {
                    val = this->mDZ;
                } else if (axis == X_AXIS) {
                    val = this->mDX;
                }
                return val;
            }

            void SetResults(uint index, Result *apResult) {
                this->mvResults[index] = apResult;
            }

            std::vector<Result *> GetResults() const {
                return this->mvResults;
            }

            Result *GetResultAt(uint index) const {
                return this->mvResults[index];
            }

            uint GetGatherDimension() const {
                return this->mGatherDimension;
            }

        private:
            uint mNX;
            uint mNY;
            uint mNZ;
            uint mNT;

            float mDX;
            float mDY;
            float mDZ;
            float mDT;

            std::vector<Result *> mvResults;

            /**
            * @brief The extra dimension is supposed to carry the number of angles in angle
            * domain common image gathers or number of offsets in offset domain common
            * image gathers, etc
            */
            uint mGatherDimension;
        };
    } //namespace dataunits
} //namespace operations

#endif // OPERATIONS_LIB_DATA_UNITS_MIGRATION_MIGRATION_DATA_HPP
