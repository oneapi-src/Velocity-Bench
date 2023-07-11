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
// Created by zeyad-osama on 02/09/2020.
//

#ifndef PIPELINE_WRITERS_WRITER_HPP
#define PIPELINE_WRITERS_WRITER_HPP

#include <operations/engines/concrete/RTMEngine.hpp>
#include <operations/common/DataTypes.h>
#include <operations/utils/io/write_utils.h>
#include <operations/utils/filters/noise_filtering.h>


namespace stbx {
    namespace writers {
        /**
         * @brief Writer class is used to write various types
         * of formats for a given migration data
         *
         * @note Should be inherited whenever a new approach is used
         */
        class Writer {
        public:
            /**
             * @brief Constructor could be overridden to
             * initialize needed  member variables.
             */
            Writer() = default;

            /**
             * @brief Destructor should be overridden to
             * ensure correct memory management.
             */
            virtual ~Writer() {
                delete[] mRawMigration;
            };

            /**
             * @brief Assign initialized engine to agent to use in
             * all functions.
             * @param aEngine : RTMEngine
             */
            inline void AssignMigrationData(
                    operations::dataunits::MigrationData *apMigrationData) {
                mpMigrationData = apMigrationData;
            }


            virtual void Write(const std::string &write_path, bool is_traces = false) {
                SpecifyRawMigration();
                PostProcess();
                Filter();
                WriteSegy(mRawMigration, write_path + "/raw_migration");
                WriteSegy(mFilteredMigration, write_path + "/filtered_migration");
                WriteBinary(mRawMigration, write_path + "/raw_migration");
                WriteBinary(mFilteredMigration, write_path + "/filtered_migration");
                WriteTimeResults(write_path);
            }

        protected:
            /**
             * @brief Initialize member variables assigned to
             * each derived class.
             */
            virtual void Initialize() = 0;

            /**
             * @brief Preprocess migration data before starting
             * to write.
             */
            virtual void PostProcess() = 0;

            /**
             * @brief Extracts migration results from provided
             * Migration Data.
             * @return Migration results
             */
            virtual void SpecifyRawMigration() {};

            /**
             * @brief  Filter migration data into
             * @note *mpMigrationData will be internally used
             * @param frame : float *       Frame to be filtered
             */
            virtual void Filter() {
#ifdef ENABLE_GPU_TIMINGS
                Timer *timer = Timer::GetInstance();
                timer->StartTimer("Writer::FilterMigration");
#endif
                uint nx = this->mpMigrationData->GetGridSize(X_AXIS);
                uint ny = this->mpMigrationData->GetGridSize(Y_AXIS);
                uint nz = this->mpMigrationData->GetGridSize(Z_AXIS);

                mFilteredMigration = new float[nx * ny * nz];

                operations::utils::filters::filter_stacked_correlation(this->mRawMigration,
                                                                       this->mFilteredMigration,
                                                                       nx, ny, nz,
                                                                       this->mpMigrationData->GetCellDimensions(X_AXIS),
                                                                       this->mpMigrationData->GetCellDimensions(Y_AXIS),
                                                                       this->mpMigrationData->GetCellDimensions(
                                                                               Z_AXIS));
#ifdef ENABLE_GPU_TIMINGS
                timer->StopTimer("Writer::FilterMigration");
#endif
            };

            /**
             * @brief  Writes migration data into .segy format
             * @note *mpMigrationData will be internally used
             * @param file_name : string     File name to be written.
             * @param is_traces : bool       Check
             */
            virtual void WriteSegy(float *frame, const std::string &file_name, bool is_traces = false) {
                operations::dataunits::MigrationData *md = mpMigrationData;
                std::string file_name_extension = file_name + ".segy";

                operations::utils::io::write_segy(this->mpMigrationData->GetGridSize(X_AXIS),
                                                  this->mpMigrationData->GetGridSize(Y_AXIS),
                                                  this->mpMigrationData->GetGridSize(Z_AXIS),
                                                  this->mpMigrationData->GetNT(),
                                                  this->mpMigrationData->GetCellDimensions(X_AXIS),
                                                  this->mpMigrationData->GetCellDimensions(Y_AXIS),
                                                  this->mpMigrationData->GetCellDimensions(Z_AXIS),
                                                  this->mpMigrationData->GetDT(),
                                                  frame, file_name_extension, is_traces);
            };

            /**
             * @brief  Writes migration data into .bin format
             * @note *mpMigrationData will be internally used
             * @param file_name : string     File name to be written.
             * @param is_traces : bool       Check
             */
            virtual void WriteBinary(float *frame, const std::string &file_name, bool is_traces = false) {
                std::string file_name_extension = file_name + ".bin";
                operations::utils::io::write_binary(frame,
                                                    this->mpMigrationData->GetGridSize(X_AXIS),
                                                    this->mpMigrationData->GetGridSize(Z_AXIS),
                                                    file_name_extension.c_str());
            };

            /**
             * @brief  Writes migration data into .su format
             * @note *mpMigrationData will be internally used
             * @param file_name : string     File name to be written.
             * @param is_traces : bool       Check
             */
            virtual void WriteSU(float *frame, const std::string &file_name, bool is_traces = false) {
                std::string file_name_extension = file_name + ".su";
            };

            /**
             * @brief  Writes migration data into .csv format
             * @note *mpMigrationData will be internally used
             * @param file_name : string     File name to be written.
             * @param is_traces : bool       Check
             */
            virtual void WriteCSV(float *frame, const std::string &file_name, bool is_traces = false) {
                std::string file_name_extension = file_name + ".csv";
            };

            /**
             * @brief  Writes migration data into .png format
             * @note *mpMigrationData will be internally used
             * @param file_name : string     File name to be written.
             * @param is_traces : bool       Check
             */
            virtual void WriteImage(float *frame, const std::string &file_name, bool is_traces = false) {
                std::string file_name_extension = file_name + ".png";
            };

            /**
             * @brief  Writes time results performed by Timer.
             * @param file_name : string     File name to be written.
             */
            void WriteTimeResults(const std::string &file_name) {
                Timer *timer = Timer::GetInstance();
                std::cout << std::endl << "Timings of the application are: " << std::endl;
                std::cout << "==============================" << std::endl;
                timer->ExportToFile(file_name + "/timing_results.txt", 1);
            };

        protected:
            /// Engine instance needed by agent to preform task upon
            operations::dataunits::MigrationData *mpMigrationData;

            float *mFilteredMigration;

            float *mRawMigration;

        private:
            Writer           (Writer const &RHS) = delete;
            Writer &operator=(Writer const &RHS) = delete;
        };
    }//namespace writers
}//namespace stbx

#endif //PIPELINE_WRITERS_WRITER_HPP
