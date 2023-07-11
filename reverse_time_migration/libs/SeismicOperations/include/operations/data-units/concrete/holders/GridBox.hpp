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
// Created by zeyad-osama on 23/08/2020.
//

#ifndef OPERATIONS_LIB_DATA_UNITS_GRID_BOX_HPP
#define OPERATIONS_LIB_DATA_UNITS_GRID_BOX_HPP

#include "operations/data-units/interface/DataUnit.hpp"
#include "operations/common//DataTypes.h"
#include "FrameBuffer.hpp"

#include "operations/exceptions/Exceptions.h"

#include <iostream>
#include <map>
#include <vector>
#include <cstring>

namespace operations {
    namespace dataunits {
/**
 * @tableofcontents
 *      - Brief
 *      - Keys Description
 *      - Keys
 *
 * @brief
 * Grid box holds all the meta data (i.e. dt, nt, grid size, window
 * size, window start, reference point and cell dimensions) needed
 * by all components.
 *
 * It also holds all wave fields, parameters and parameters window
 * in member maps included in this class.
 *
 * Keys to all maps are of type u_int16_t thus 16 bits are needed for
 * the key to be constructed. These 16 bits are constructed as same as
 * Unix commands
 *
 * @example: (PARM | CURR | DIR_X) meaning that this element is a parameter
 * which have current time step and in the x direction.
 *
 * Keys are discussed more below.
 *
 * @note Using this class should not mess up with the bit mask
 *
 * @keysdescription
 *      - Bit 0:
 *          Identifies an element is a window or not.
 *      - Bit 1,2:
 *          Identifies an element is a wave field or a parameter.
 *              * WAVE      01
 *              * PARM      10
 *      - Bit 3,4,5:
 *          Identifies an element's time, whether it's current, next or previous.
 *              * CURR      001
 *              * PREV      010
 *              * NEXT      011
 *      - Bit 6,7:
 *          Identifies an element's direction. X,Y or Z.
 *              * DIR_Z     00
 *              * DIR_X     01
 *              * DIR_Y     10
 *      - Bit 8,9,10,11,12,13:
 *          Identifies an element's direction. X,Y or Z.
 *              * GB_VEL    000001
 *              * GB_DLT    000010
 *              * GB_EPS    000011
 *              * GB_THT    000100
 *              * GB_PHI    000101
 *              * GB_DEN    000111
 *      - Bit 14,15:
 *          Parity bits.
 *
 *
 * @keys
 *
 * WINDOW               1  00  000  00  000000  00
 *
 * WAVE FIELD           0  01  000  00  000000  00
 * PARAMETER            0  10  000  00  000000  00
 *
 * CURRENT              0  00  001  00  000000  00
 * PREVIOUS             0  00  010  00  000000  00
 * NEXT                 0  00  011  00  000000  00
 *
 * DIRECTION Z          0  00  000  00  000000  00
 * DIRECTION X          0  00  000  01  000000  00
 * DIRECTION Y          0  00  000  10  000000  00
 *
 * VELOCITY             0  00  000  00  000001  00
 * DELTA                0  00  000  00  000010  00
 * EPSILON              0  00  000  00  000011  00
 * THETA                0  00  000  00  000100  00
 * PHI                  0  00  000  00  000101  00
 * DENSITY              0  00  000  00  000111  00
 * PARTICLE VELOCITY    0  00  000  00  001000  00
 * PRESSURE             0  00  000  00  001001  00
*/

/**
 * @note
 * Never add definitions to your concrete implementations
 * with the same names.
 */
#define WIND       0b1000000000000000
#define WAVE       0b0010000000000000
#define PARM       0b0100000000000000
#define CURR       0b0000010000000000
#define PREV       0b0000100000000000
#define NEXT       0b0000110000000000
#define DIR_Z      0b0000000000000000
#define DIR_X      0b0000000100000000
#define DIR_Y      0b0000001000000000
#define GB_VEL     0b0000000000000100
#define GB_DLT     0b0000000000001000
#define GB_EPS     0b0000000000001100
#define GB_THT     0b0000000000010000
#define GB_PHI     0b0000000000010100
#define GB_DEN     0b0000000000011100
#define GB_PRTC    0b0000000000100000
#define GB_PRSS    0b0000000000100100

        /**
         * @brief
         * Grid box holds all the meta data (i.e. dt, nt, grid size, window
         * size, window start, reference point and cell dimensions) needed
         * by all components.
         *
         * It also holds all wave fields, parameters and parameters window
         * in member maps included in this class.
         */
        class GridBox : public DataUnit {
        public:
            typedef u_int16_t Key;

        public:
            /**
             * @brief Constructor
             */
            GridBox();

            /**
             * @brief Define the destructor as virtual which is a member function
             * that is declared within a base class and re-defined (Overridden)
             * by a derived class default = {}
             */
            ~GridBox() override;

            /**
             * @brief DT setter.
             * @throw IllogicalException()
             */
            void SetDT(float _dt);

            /**
             * @brief DT getter.
             */
            inline float GetDT() const {
                return this->mDT;
            }

            /**
             * @brief NT setter.
             * @throw IllogicalException()
             */
            void SetNT(float _nt);

            /**
             * @brief NT getter.
             */
            inline uint GetNT() const {
                return this->mNT;
            }

            /**
             * @brief Reference point per axe getter.
             * @param[in] axis      Axe direction
             * @param[in] val       Value to be set
             * @throw IllogicalException()
             */
            void SetReferencePoint(uint axis, float val);

            /**
             * @brief Reference point per axe setter.
             * @param[in] axis      Axe direction
             * @return[out] value   Value
             * @throw IllogicalException()
             */
            float GetReferencePoint(uint axis);

            /**
             * @brief ReferencePoint struct getter.
             * @return[out] FPoint3D   Value
             */
            inline FPoint3D *GetReferencePoint() {
                return this->mpReferencePoint;
            }

            /**
             * @brief Cell dimension per axe setter.
             * @param[in] axis      Axe direction
             * @param[in] val       Value to be set
             * @throw IllogicalException()
             */
            void SetCellDimensions(uint axis, float val);

            /**
             * @brief Cell dimension per axe getter.
             * @param[in] axis      Axe direction
             * @return[out] value   Value
             */
            float GetCellDimensions(uint axis);

            /**
             * @brief Initial Cell dimension per axe setter.
             * @param[in] axis      Axe direction
             * @param[in] val       Value to be set
             * @throw IllogicalException()
             */
            void SetInitialCellDimensions(uint axis, float val);

            /**
             * @brief Initial Cell dimension per axe getter.
             * @param[in] axis      Axe direction
             * @return[out] value   Value
             */
            float GetInitialCellDimensions(uint axis);

            /**
             * @brief CellDimensions struct getter.
             * @return[out] CellDimensions   Value
             */
            inline CellDimensions *GetCellDimensions() {
                return this->mpCellDimensions;
            }

            /**
             * @brief CellDimensions struct getter for initial dimensions.
             * @return[out] CellDimensions   Value
             */
            inline CellDimensions *GetInitialCellDimensions() {
                return this->mpInitialCellDimensions;
            }

            /**
             * @brief WindowProperties -> size per axe setter.
             * @param[in] axis      Axe direction
             * @param[in] val       Value to be set
             * @throw IllogicalException()
             */
            void SetActualWindowSize(uint axis, uint val);

            /**
             * @brief WindowProperties -> size per axe getter.
             * @param[in] axis      Axe direction
             * @return[out] value   Value
             */
            uint GetActualWindowSize(uint axis);

            /**
             * @brief WindowProperties struct getter.
             * @return[out] WindowProperties   Value
             */
            inline WindowProperties *GetWindowProperties() {
                return this->mpWindowProperties;
            }

            /**
             * @brief Window start per axe setter.
             * @param[in] axis      Axe direction
             * @param[in] val       Value to be set
             * @throw IllogicalException()
             */
            void SetWindowStart(uint axis, uint val);

            /**
             * @brief Window start per axe getter.
             * @param[in] axis      Axe direction
             * @return[out] value   Value
             */
            uint GetWindowStart(uint axis);

            /**
             * @brief Window start struct getter.
             */
            Point3D *GetWindowStart() {
                return &this->mpWindowProperties->window_start;
            }

            /**
             * @brief Grid size per axe setter.
             * @param[in] axis      Axe direction
             * @param[in] val       Value to be set
             * @throw IllogicalException()
             */
            void SetActualGridSize(uint axis, uint val);

            /**
             * @brief Grid size per axe getter.
             * @param[in] axis      Axe direction
             * @return[out] value   Value
             */
            uint GetActualGridSize(uint axis);

            /**
             * @brief Grid size struct getter.
             * @return[out] GridSize   Value
             */
            inline GridSize *GetActualGridSize() {
                return this->mpActualGridSize;
            }

            /**
             * @brief Grid size per axe setter.
             * @param[in] axis      Axe direction
             * @param[in] val       Value to be set
             * @throw IllogicalException()
             */
            void SetLogicalGridSize(uint axis, uint val);

            /**
             * @brief Grid size per axe getter.
             * @param[in] axis      Axe direction
             * @return[out] value   Value
             */
            uint GetLogicalGridSize(uint axis);

            /**
             * @brief Grid size struct getter.
             * @return[out] GridSize   Value
             */
            inline GridSize *GetLogicalGridSize() {
                return this->mpLogicalGridSize;
            }

            /**
             * @brief Grid size per axe setter.
             * @param[in] axis      Axe direction
             * @param[in] val       Value to be set
             * @throw IllogicalException()
             */
            void SetComputationGridSize(uint axis, uint val);

            /**
             * @brief Grid size per axe getter.
             * @param[in] axis      Axe direction
             * @return[out] value   Value
             */
            uint GetComputationGridSize(uint axis);

            /**
             * @brief Grid size struct getter.
             * @return[out] GridSize   Value
             */
            inline GridSize *GetComputationGridSize() {
                return this->mpComputationGridSize;
            }

            /**
             * @brief Logical Window size per axe setter.
             * @param[in] axis      Axe direction
             * @param[in] val       Value to be set
             * @throw IllogicalException()
             */
            void SetLogicalWindowSize(uint axis, uint val);

            /**
             * @brief Logical Window size per axe getter.
             * @param[in] axis      Axe direction
             * @return[out] value   Value
             */
            uint GetLogicalWindowSize(uint axis);

            /**
             * @brief Logical Window size struct getter.
             * @return[out] GridSize   Value
             */
            inline GridSize *GetLogicalWindowSize() {
                return this->mpLogicalWindowSize;
            }

            /**
             * @brief Initial Grid size per axe getter.
             * @param[in] axis      Axe direction
             * @return[out] value   Value
             */
            void SetInitialGridSize(uint axis, uint val);

            /**
             * @brief Initial Grid size per axe getter.
             * @param[in] axis      Axe direction
             * @return[out] value   Value
             */
            uint GetInitialGridSize(uint axis);

            /**
             * @brief Initial Grid size struct getter.
             * @return[out] GridSize   Value
             */
            inline GridSize *GetInitialGridSize() {
                return this->mpInitialGridSize;
            }

            /**
            * @brief Registers an allocated wave field pointer and it's
            * window pointer accompanied with it's value.
            * @param[in] key                        Wave field name
            * @param[in] ptr_wave_field             Allocated wave field pointer
            */
            void RegisterWaveField(u_int16_t key, FrameBuffer<float> *ptr_wave_field);

            /**
             * @brief Registers an allocated parameter pointer
             * accompanied with it's value.
             * @param[in] key                       Parameter name
             * @param[in] ptr_parameter             Allocated parameter pointer
             * @param[in] ptr_parameter_window      Allocated parameter window pointer
             */
            void RegisterParameter(u_int16_t key,
                                   FrameBuffer<float> *ptr_parameter,
                                   FrameBuffer<float> *ptr_parameter_window = nullptr);

            /**
             * @brief Master wave field setter.
             * @return[out] float *
             */
            void RegisterMasterWaveField();

            /**
             * @brief Wave Fields map getter.
             * @return[out] this->mWaveField
             */
            inline std::map<u_int16_t, FrameBuffer<float> *> GetWaveFields() {
                return this->mWaveFields;
            }

            /**
             * @brief Parameters map getter.
             * @return[out] this->mParameters
             */
            inline std::map<u_int16_t, FrameBuffer<float> *> GetParameters() {
                return this->mParameters;
            }

            /**
             * @brief Window Parameters map getter.
             * @return[out] this->mWindowParameters
             */
            inline std::map<u_int16_t, FrameBuffer<float> *> GetWindowParameters() {
                return this->mWindowParameters;
            }

            /**
             * @brief Master wave field getter.
             * @return[out] float *
             */
            float *GetMasterWaveField();

            /**
             * @brief WaveField/Parameter/WindowParameter getter.
             * @param[in] key           Key
             * @return[out] float *      (Parameter | Wave field | Window Parameter) pointer
             * @throw NotFoundException()
             */
            FrameBuffer<float> *Get(u_int16_t key);

            /**
             * @brief WaveField/Parameter/WindowParameter setter.
             * @param[in] key       Key
             * @param[in] val       FrameBuffer
             * @throw NotFoundException()
             */
            void Set(u_int16_t key, FrameBuffer<float> *val);

            /**
             * @brief WaveField/Parameter/WindowParameter setter.
             * @param[in] key       Key
             * @param[in] val       float pointer
             * @throw NotFoundException()
             */
            void Set(u_int16_t key, float *val);

            /**
             * @brief WaveField/Parameter/WindowParameter value re-setter.
             * @param[in] key       Key
             * @note Uses normal setter with a nullptr set value.
             */
            void Reset(u_int16_t key);

            /**
             * @brief Swaps two pointers with respect to the provided keys.
             * @param[in] _src
             * @param[in] _dst
             * @throw NotFoundException()
             */
            void Swap(u_int16_t _src, u_int16_t _dst);

            /**
             * @brief Clones current grid box into the sent grid box.
             *
             * @param[in] apGridBox
             * GridBox to clone in.
             */
            void Clone(GridBox *apGridBox);

            /**
             * @brief Clones current grid box's meta data into
             * the sent grid box.
             *
             * @param[in] apGridBox
             * GridBox to clone in.
             */
            void CloneMetaData(GridBox *apGridBox);

            /**
             * @brief Clones current grid box's wave fields into
             * the sent grid box.
             *
             * @param[in] apGridBox
             * GridBox to clone in.
             *
             * @note Same pointers for registry are used.
             */
            void CloneWaveFields(GridBox *apGridBox);

            /**
             * @brief Clones current grid box's parameters into
             * the sent grid box.
             *
             * @param[in] apGridBox
             * GridBox to clone in.
             *
             * @note Same pointers for registry are used.
             */
            void CloneParameters(GridBox *apGridBox);

            /**
             * @brief Report all current grid box inner values.
             * @param[in] aReportLevel
             */
            void Report(REPORT_LEVEL aReportLevel = SIMPLE);

            /**
             * @brief WaveField/Parameter/WindowParameter checker.
             * @param[in] key           Key
             * @return[out] boolean
             *
             */
            bool Has(u_int16_t key);

        public:
            /**
             * @param[in] key
             * @param[in] mask
             * @return[out] bool is the mask included in the key.
             */
            static bool inline Includes(u_int16_t key, u_int16_t mask) {
                return (key & mask) == mask;
            }

            /**
             * @brief Masks source bits by provided mask bits.
             * @param[in,out] key
             * @param[in] _src
             * @param[in] _dest
             */
            static void Replace(u_int16_t *key, u_int16_t _src, u_int16_t _dest);

            /**
             * @brief Converts a given key (i.e. u_int16_t) to string.
             * @param[in] key
             * @return[out] String value
             */
            static std::string Stringify(u_int16_t key);

            std::string Beautify(std::string const &str);

        private:
            /**
             * @brief Bit setter
             * @param[in] key
             * @param[in] type
             */
            static void inline SetBits(u_int16_t *key, u_int16_t type) {
                *key |= type;
            }

            /**
             * @brief Convert first letter in string to uppercase
             * @paramp[in] str to be capitalized
             * @return[out] capitalized string
             */
            std::string Capitalize(std::string str);

            /**
             * @brief Replace all occurrences of a character in string
             * @paramp[in] str to be operated on
             * @paramp[in] str to be changed from
             * @paramp[in] str to be changed to
             * @return[out] replaced string
             */
            std::string ReplaceAll(std::string str,
                                   const std::string &from,
                                   const std::string &to);

            GridBox           (GridBox const &RHS) = delete;
            GridBox &operator=(GridBox const &RHS) = delete;

        private:
            /// Wave field map
            std::map<u_int16_t, FrameBuffer<float> *> mParameters;
            /// Parameters map
            std::map<u_int16_t, FrameBuffer<float> *> mWindowParameters;
            /// Window Parameters map
            std::map<u_int16_t, FrameBuffer<float> *> mWaveFields;
            /// Size of the logical grid.
            GridSize *mpLogicalGridSize;
            /// Size of the actual grid.
            GridSize *mpActualGridSize;
            /// Initial size of the grid according to the SEGY model.
            GridSize *mpInitialGridSize;
            /// Actual window size and window start.
            WindowProperties *mpWindowProperties;
            /// Size of the computation grid.
            GridSize *mpComputationGridSize;
            /// Logical window size.
            GridSize *mpLogicalWindowSize;
            /// Step sizes of the grid.
            CellDimensions *mpCellDimensions;
            /// Initial step sizes of the grid according to the SEGY model.
            CellDimensions *mpInitialCellDimensions;
            /// Time-step size.
            float mDT;
            /// Number of time steps.
            uint mNT;
            /// Reference point of the model in real coordinate distance.
            FPoint3D *mpReferencePoint;
        };
    } //namespace dataunits
} //namespace operations

#endif //OPERATIONS_LIB_DATA_UNITS_GRID_BOX_HPP
