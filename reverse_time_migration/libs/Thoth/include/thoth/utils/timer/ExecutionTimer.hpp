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
// Created by zeyad-osama on 10/03/2021.
//

#ifndef THOTH_UTILS_TIMER_EXECUTION_TIMER_HPP
#define THOTH_UTILS_TIMER_EXECUTION_TIMER_HPP

#include <chrono>
#include <iostream>

namespace thoth {
    namespace utils {
        namespace timer {

            /**
             * @brief Function execution timer.
             */
            class ExecutionTimer {
            public:
                /**
                 * @brief Function execution timer. Takes a block of code as parameter and evaluate
                 * it's run time. Takes a flag that determines whether to print put the execution
                 * time or not.
                 *
                 * @note Time of execution is return as microseconds.
                 */
                template<typename T>
                static long
                Evaluate(T f, bool aShowExecutionTime = false) {
                    auto start = std::chrono::high_resolution_clock::now();
                    f();
                    auto stop = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                    auto count = duration.count();
                    if (aShowExecutionTime) {
                        ExecutionTimer::ShowResult(count);
                    }
                    return count;
                }

                /**
                 * @brief Takes execution time in micro seconds and shows it in seconds,
                 */
                static int
                ShowResult(long aTime) {
                    return printf("Execution Time: %.4f SEC\n", aTime / (1.0 * 1e6));
                }
            };
        }
    }
}

#endif //THOTH_UTILS_TIMER_EXECUTION_TIMER_HPP
