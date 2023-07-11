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
// Created by zeyad-osama on 18/09/2020.
//

#ifndef OPERATIONS_LIB_COMPUTATION_PARAMETERS_HPP
#define OPERATIONS_LIB_COMPUTATION_PARAMETERS_HPP

#include "DataTypes.h"
#include <iostream>

namespace operations {
    namespace common {

        /**
         * @brief Parameters of the simulation independent from the block.
         */
        class ComputationParameters {
        public:
            /**
             * @brief Constructor of the class, it takes as input the half_length
             */
            explicit ComputationParameters(HALF_LENGTH aHalfLength) {
                /// Assign default values.
                this->mBlockX = 512;
                this->mBlockY = 15;
                this->mBlockZ = 44;
                this->mThreadCount = 16;

                this->mSourceFrequency = 200;
                this->mIsotropicRadius = 5;
                this->mBoundaryLength = 20;
                this->mHalfLength = aHalfLength;
                this->mRelaxedDT = 0.4;
                this->mIsUsingWindow = false;

                /// Array of floats of size hl+1 only contains the zero and positive (x>0 )
                /// coefficients and not all coefficients
                this->mpSecondDerivativeFDCoefficient = new float[aHalfLength + 1];

                /// Array of floats of size hl+1 only contains the zero and positive (x>0 )
                /// coefficients and not all coefficients
                this->mpFirstDerivativeFDCoefficient = new float[aHalfLength + 1];

                /// Array of floats of size hl+1 only contains the zero and positive (x>0 )
                /// coefficients and not all coefficients
                this->mpFirstDerivativeStaggeredFDCoefficient = new float[aHalfLength + 1];

                if (aHalfLength == O_2) {
                    /**
                     * Accuracy = 2
                     * Half Length = 1
                     */

                    this->mpSecondDerivativeFDCoefficient[0] = -2;
                    this->mpSecondDerivativeFDCoefficient[1] = 1;

                    this->mpFirstDerivativeFDCoefficient[0] = 0;
                    this->mpFirstDerivativeFDCoefficient[1] = 0.5;

                    this->mpFirstDerivativeStaggeredFDCoefficient[0] = 0.0f;
                    this->mpFirstDerivativeStaggeredFDCoefficient[1] = 1.0f;
                } else if (aHalfLength == O_4) {
                    /**
                     * Accuracy = 4
                     * Half Length = 2
                     */

                    this->mpSecondDerivativeFDCoefficient[0] = -2.5;
                    this->mpSecondDerivativeFDCoefficient[1] = 1.33333333333;
                    this->mpSecondDerivativeFDCoefficient[2] = -0.08333333333;

                    this->mpFirstDerivativeFDCoefficient[0] = 0.0f;
                    this->mpFirstDerivativeFDCoefficient[1] = 2.0f / 3.0f;
                    this->mpFirstDerivativeFDCoefficient[2] = -1.0f / 12.0f;

                    this->mpFirstDerivativeStaggeredFDCoefficient[0] = 0.0f;
                    this->mpFirstDerivativeStaggeredFDCoefficient[1] = 1.125f;
                    this->mpFirstDerivativeStaggeredFDCoefficient[2] = -0.041666666666666664f;
                } else if (aHalfLength == O_8) {
                    /**
                     * Accuracy = 8
                     * Half Length = 4
                     */

                    this->mpSecondDerivativeFDCoefficient[0] = -2.847222222;
                    this->mpSecondDerivativeFDCoefficient[1] = +1.6;
                    this->mpSecondDerivativeFDCoefficient[2] = -0.2;
                    this->mpSecondDerivativeFDCoefficient[3] = +2.53968e-2;
                    this->mpSecondDerivativeFDCoefficient[4] = -1.785714e-3;

                    this->mpFirstDerivativeFDCoefficient[0] = 0;
                    this->mpFirstDerivativeFDCoefficient[1] = +0.8;
                    this->mpFirstDerivativeFDCoefficient[2] = -0.2;
                    this->mpFirstDerivativeFDCoefficient[3] = +0.03809523809;
                    this->mpFirstDerivativeFDCoefficient[4] = -0.00357142857;

                    this->mpFirstDerivativeStaggeredFDCoefficient[0] = 0.0f;
                    this->mpFirstDerivativeStaggeredFDCoefficient[1] = 1.1962890625f;
                    this->mpFirstDerivativeStaggeredFDCoefficient[2] = -0.07975260416666667f;
                    this->mpFirstDerivativeStaggeredFDCoefficient[3] = 0.0095703125f;
                    this->mpFirstDerivativeStaggeredFDCoefficient[4] = -0.0006975446428571429f;
                } else if (aHalfLength == O_12) {
                    /**
                     * Accuracy = 12
                     * Half Length = 6
                     */

                    this->mpSecondDerivativeFDCoefficient[0] = -2.98277777778;
                    this->mpSecondDerivativeFDCoefficient[1] = 1.71428571429;
                    this->mpSecondDerivativeFDCoefficient[2] = -0.26785714285;
                    this->mpSecondDerivativeFDCoefficient[3] = 0.05291005291;
                    this->mpSecondDerivativeFDCoefficient[4] = -0.00892857142;
                    this->mpSecondDerivativeFDCoefficient[5] = 0.00103896103;
                    this->mpSecondDerivativeFDCoefficient[6] = -0.00006012506;

                    this->mpFirstDerivativeFDCoefficient[0] = 0.0;
                    this->mpFirstDerivativeFDCoefficient[1] = 0.857142857143;
                    this->mpFirstDerivativeFDCoefficient[2] = -0.267857142857;
                    this->mpFirstDerivativeFDCoefficient[3] = 0.0793650793651;
                    this->mpFirstDerivativeFDCoefficient[4] = -0.0178571428571;
                    this->mpFirstDerivativeFDCoefficient[5] = 0.0025974025974;
                    this->mpFirstDerivativeFDCoefficient[6] = -0.000180375180375;

                    this->mpFirstDerivativeStaggeredFDCoefficient[0] = 0.0f;
                    this->mpFirstDerivativeStaggeredFDCoefficient[1] = 1.2213363647460938f;
                    this->mpFirstDerivativeStaggeredFDCoefficient[2] = -0.09693145751953125f;
                    this->mpFirstDerivativeStaggeredFDCoefficient[3] = 0.017447662353515626f;
                    this->mpFirstDerivativeStaggeredFDCoefficient[4] = -0.002967289515904018f;
                    this->mpFirstDerivativeStaggeredFDCoefficient[5] = 0.0003590053982204861f;
                    this->mpFirstDerivativeStaggeredFDCoefficient[6] = -2.184781161221591e-05f;
                } else if (aHalfLength == O_16) {
                    /**
                     * Accuracy = 16
                     * Half Length = 8
                     */
                    this->mpSecondDerivativeFDCoefficient[0] = -3.05484410431;
                    this->mpSecondDerivativeFDCoefficient[1] = 1.77777777778;
                    this->mpSecondDerivativeFDCoefficient[2] = -0.311111111111;
                    this->mpSecondDerivativeFDCoefficient[3] = 0.0754208754209;
                    this->mpSecondDerivativeFDCoefficient[4] = -0.0176767676768;
                    this->mpSecondDerivativeFDCoefficient[5] = 0.00348096348096;
                    this->mpSecondDerivativeFDCoefficient[6] = -0.000518000518001;
                    this->mpSecondDerivativeFDCoefficient[7] = 5.07429078858e-05;
                    this->mpSecondDerivativeFDCoefficient[8] = -2.42812742813e-06;

                    this->mpFirstDerivativeFDCoefficient[0] = -6.93889390391e-17;
                    this->mpFirstDerivativeFDCoefficient[1] = 0.888888888889;
                    this->mpFirstDerivativeFDCoefficient[2] = -0.311111111111;
                    this->mpFirstDerivativeFDCoefficient[3] = 0.113131313131;
                    this->mpFirstDerivativeFDCoefficient[4] = -0.0353535353535;
                    this->mpFirstDerivativeFDCoefficient[5] = 0.00870240870241;
                    this->mpFirstDerivativeFDCoefficient[6] = -0.001554001554;
                    this->mpFirstDerivativeFDCoefficient[7] = 0.0001776001776;
                    this->mpFirstDerivativeFDCoefficient[8] = -9.71250971251e-06;

                    this->mpFirstDerivativeStaggeredFDCoefficient[0] = 0.0f;
                    this->mpFirstDerivativeStaggeredFDCoefficient[1] = 1.2340910732746122f;
                    this->mpFirstDerivativeStaggeredFDCoefficient[2] = -0.10664984583854668f;
                    this->mpFirstDerivativeStaggeredFDCoefficient[3] = 0.023036366701126076f;
                    this->mpFirstDerivativeStaggeredFDCoefficient[4] = -0.005342385598591385f;
                    this->mpFirstDerivativeStaggeredFDCoefficient[5] = 0.0010772711700863268f;
                    this->mpFirstDerivativeStaggeredFDCoefficient[6] = -0.00016641887751492495f;
                    this->mpFirstDerivativeStaggeredFDCoefficient[7] = 1.7021711056048922e-05f;
                    this->mpFirstDerivativeStaggeredFDCoefficient[8] = -8.523464202880773e-07f;
                }
            }

            /**
             * @brief Destructor is virtual which can be overridden in
             * derived classes from this class.
             */
            virtual ~ComputationParameters() {
                /// Destruct the array of floats of coefficients of the
                /// second derivative finite difference
                delete[] this->mpSecondDerivativeFDCoefficient;

                /// Destruct the array of floats of coefficients of the
                /// first derivative finite difference
                delete[] this->mpFirstDerivativeFDCoefficient;

                /// Destruct the array of floats of coefficients of the
                /// first derivative staggered finite difference
                delete[] this->mpFirstDerivativeStaggeredFDCoefficient;
            }

            /**
             * Setter and getters.
             */
        public:
            inline const EQUATION_ORDER &GetEquationOrder() const {
                return this->mEquationOrder;
            }

            inline void SetEquationOrder(const EQUATION_ORDER &aEquationOrder) {
                this->mEquationOrder = aEquationOrder;
            }

            inline const PHYSICS &GetPhysics() const {
                return this->mPhysics;
            }

            inline void SetPhysics(const PHYSICS &aPhysics) {
                this->mPhysics = aPhysics;
            }

            inline const APPROXIMATION &GetApproximation() const {
                return mApproximation;
            }

            inline void SetApproximation(const APPROXIMATION &aApproximation) {
                this->mApproximation = aApproximation;
            }

            inline float GetSourceFrequency() const {
                return this->mSourceFrequency;
            }

            inline void SetSourceFrequency(float aSourceFrequency) {
                this->mSourceFrequency = aSourceFrequency;
            }

            inline int GetIsotropicRadius() const {
                return this->mIsotropicRadius;
            }

            inline void SetIsotropicRadius(int aIsotropicRadius) {
                this->mIsotropicRadius = aIsotropicRadius;
            }

            inline const HALF_LENGTH &GetHalfLength() const {
                return this->mHalfLength;
            }

            inline void SetHalfLength(const HALF_LENGTH &aHalfLength) {
                this->mHalfLength = aHalfLength;
            }

            inline int GetBoundaryLength() const {
                return this->mBoundaryLength;
            }

            inline void SetBoundaryLength(int aBoundaryLength) {
                this->mBoundaryLength = aBoundaryLength;
            }

            inline float *GetSecondDerivativeFDCoefficient() const {
                return this->mpSecondDerivativeFDCoefficient;
            }

            inline float *GetFirstDerivativeFDCoefficient() const {
                return this->mpFirstDerivativeFDCoefficient;
            }

            inline float *GetFirstDerivativeStaggeredFDCoefficient() const {
                return this->mpFirstDerivativeStaggeredFDCoefficient;
            }

            inline float GetRelaxedDT() const {
                return this->mRelaxedDT;
            }

            inline void SetRelaxedDT(float aRelaxedDt) {
                this->mRelaxedDT = aRelaxedDt;
            }

            inline bool IsUsingWindow() const {
                return this->mIsUsingWindow;
            }

            inline void SetIsUsingWindow(bool aIsUsingWindow) {
                this->mIsUsingWindow = aIsUsingWindow;
            }

            inline int GetLeftWindow() const {
                return this->mLeftWindow;
            }

            inline void SetLeftWindow(int aLeftWindow) {
                this->mLeftWindow = aLeftWindow;
            }

            inline int GetRightWindow() const {
                return this->mRightWindow;
            }

            inline void SetRightWindow(int aRightWindow) {
                this->mRightWindow = aRightWindow;
            }

            inline int GetDepthWindow() const {
                return mDepthWindow;
            }

            inline void SetDepthWindow(int aDepthWindow) {
                this->mDepthWindow = aDepthWindow;
            }

            inline int GetFrontWindow() const {
                return mFrontWindow;
            }

            inline void SetFrontWindow(int aFrontWindow) {
                this->mFrontWindow = aFrontWindow;
            }

            inline int GetBackWindow() const {
                return this->mBackWindow;
            }

            inline void SetBackWindow(int aBackWindow) {
                this->mBackWindow = aBackWindow;
            };

            inline int GetAlgorithm() const {
                return this->mAlgorithm;
            }

            inline void SetAlgorithm(ALGORITHM aAlgorithm) {
                this->mAlgorithm = aAlgorithm;
            };

            uint GetBlockX() const {
                return this->mBlockX;
            }

            void SetBlockX(uint block_x) {
                this->mBlockX = block_x;
            }

            uint GetBlockY() const {
                return this->mBlockY;
            }

            void SetBlockY(uint block_y) {
                this->mBlockY = block_y;
            }

            uint GetBlockZ() const {
                return this->mBlockZ;
            }

            void SetBlockZ(uint block_z) {
                this->mBlockZ = block_z;
            }

            uint GetThreadCount() const {
                return this->mThreadCount;
            }

            void SetThreadCount(uint thread_count) {
                this->mThreadCount = thread_count;
            }

            void SetCollectedQueueCreationTime(double const dQTime) {
                this->mQueueConstructionTime = dQTime;
            }

            inline double GetCollectedQueueCreationTime() {
                return this->mQueueConstructionTime;
            }

        private:
            /// Wave equation order.
            EQUATION_ORDER mEquationOrder = SECOND;

            /// Wave equation physics.
            PHYSICS mPhysics = ACOUSTIC;

            /// Wave equation approximation.
            APPROXIMATION mApproximation = ISOTROPIC;

            /// Frequency of the source (ricker wavelet).
            float mSourceFrequency;

            /// Injected isotropic radius accompanied with the ricker
            /// for shear wave suppression
            int mIsotropicRadius;

            /// Boundary length (i.e. Regardless the type of boundary).
            int mBoundaryLength;
            HALF_LENGTH mHalfLength;

            /// Pointer of floats (array of floats) which contains the
            /// second derivative finite difference coefficient.
            float *mpSecondDerivativeFDCoefficient;

            /// Pointer of floats (array of floats) which contains the
            /// first derivative finite difference coefficient.
            float *mpFirstDerivativeFDCoefficient;

            /// Pointer of floats (array of floats) which contains the
            /// first derivative staggered finite difference coefficient.
            float *mpFirstDerivativeStaggeredFDCoefficient;

            /// Stability condition safety / Relaxation factor.
            float mRelaxedDT;

            /// Use window for propagation.
            bool mIsUsingWindow;

            /// Left-side window size.
            int mLeftWindow = 0;

            /// Right-side window size.
            int mRightWindow = 0;

            /// Depth window size.
            int mDepthWindow = 0;

            /// Front window size.
            int mFrontWindow = 0;

            /// Backward window size.
            int mBackWindow = 0;

            /// Algorithm
            /// (i.e. -> RTM | PSDM | PSTM | FWI)
            ALGORITHM mAlgorithm = RTM;

            /// Cache blocking in X
            uint mBlockX;

            /// Cache blocking in Y
            uint mBlockY;

            /// Cache blocking in Z
            uint mBlockZ;

            /// Number of threads
            uint mThreadCount;

            /// Queue construction time
            double mQueueConstructionTime = 0.0; // Should be zero for CUDA but not for SYCL

        };
    }//namespace common
}//namespace operations

#endif //OPERATIONS_LIB_COMPUTATION_PARAMETERS_HPP
