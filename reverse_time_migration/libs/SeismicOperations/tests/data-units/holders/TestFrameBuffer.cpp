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
// Created by ahmed-ayyad on 12/01/2021
//

#include <operations/data-units/concrete/holders/FrameBuffer.hpp>

#include <operations/test-utils/EnvironmentHandler.hpp>

#include <libraries/catch/catch.hpp>

#include <iostream>

using namespace operations::dataunits;
using namespace operations::testutils;

TEST_CASE("FrameBuffer - Undefined Size", "[FrameBuffer]") {

    set_environment();

    auto fb_int = new FrameBuffer<int>();
    auto fb_float = new FrameBuffer<float>();

    SECTION("Negative Testing before Allocation") {
        REQUIRE(fb_int->GetNativePointer() == nullptr);
        REQUIRE(fb_float->GetNativePointer() == nullptr);
    }

    SECTION("Allocate Memory Function Test") {
        fb_int->Allocate(10, "int buffer");
        fb_float->Allocate(10, O_8, "float");

        REQUIRE(fb_int->GetNativePointer() != nullptr);
        REQUIRE(fb_float->GetNativePointer() != nullptr);


        SECTION("Memory Set Function Test") {
            char test_value = 2;
            int eval;

            eval = test_value << 24 | test_value << 16 | test_value << 8 | test_value;

            Device::MemSet(fb_int->GetNativePointer(), 2, sizeof(int) * 10);
            Device::MemSet(fb_float->GetNativePointer(), 0, sizeof(float) * 10);

            int misses = 0;
            for (int i = 0; i < 10; i++) {
                misses += (eval != fb_int->GetHostPointer()[i]);
                misses += (0 != fb_float->GetHostPointer()[i]);
            }

            REQUIRE(misses == 0);
        }

        SECTION("Memory Copy  Function Test") {
            int test_value_int[10];
            float test_value_float[10];

            for (int i = 0; i < 10; i++) {
                test_value_int[i] = i;
                test_value_float[i] = 12.5 / i;
            }

            Device::MemCpy(fb_int->GetNativePointer(), test_value_int, sizeof(int) * 10);
            Device::MemCpy(fb_float->GetNativePointer(), test_value_float, sizeof(float) * 10);

            int misses = 0;
            for (int i = 0; i < 10; i++) {
                misses += (test_value_int[i] != fb_int->GetHostPointer()[i]);
                misses += (test_value_float[i] != fb_float->GetHostPointer()[i]);
            }

            REQUIRE(misses == 0);
        }

        SECTION("Reflect Function Test") {
            fb_int->GetHostPointer()[5] = 30;
            fb_float->GetHostPointer()[5] = 0.3f;

            fb_int->ReflectOnNative();
            fb_float->ReflectOnNative();

            REQUIRE(fb_int->GetHostPointer()[5] == 30);
            REQUIRE(fb_float->GetHostPointer()[5] == 0.3f);
        }

        SECTION("Deallocate Function Test") {
            fb_int->Free();
            fb_float->Free();

            REQUIRE(fb_int->GetHostPointer() == nullptr);
            REQUIRE(fb_float->GetNativePointer() == nullptr);
        }
    }

    delete fb_int;
    delete fb_float;
}


TEST_CASE("FrameBuffer - Defined Size", "[FrameBuffer]") {
    auto fb_int = new FrameBuffer<int>(10);
    auto fb_float = new FrameBuffer<float>(10);

    SECTION("Allocate Memory Test") {
        REQUIRE(fb_int->GetHostPointer() != nullptr);
        REQUIRE(fb_float->GetNativePointer() != nullptr);
    }

    SECTION("Memory SetTest Function Test") {
        char test_value = 2;
        int eval;

        eval = test_value << 24 | test_value << 16 | test_value << 8 | test_value;

        Device::MemSet(fb_int->GetNativePointer(), test_value, sizeof(int) * 10);
        Device::MemSet(fb_float->GetNativePointer(), 0, sizeof(float) * 10);

        int misses = 0;
        for (int i = 0; i < 10; i++) {
            misses += (eval != fb_int->GetHostPointer()[i]);
            misses += (0 != fb_float->GetHostPointer()[i]);
        }

        REQUIRE(misses == 0);
    }

    SECTION("Memory Copy Function Test") {
        int test_value_int[10];
        float test_value_float[10];

        for (int i = 0; i < 10; i++) {
            test_value_int[i] = i;
            test_value_float[i] = 12.5 / i;
        }

        Device::MemCpy(fb_int->GetNativePointer(), test_value_int, sizeof(int) * 10);
        Device::MemCpy(fb_float->GetNativePointer(), test_value_float, sizeof(int) * 10);

        int misses = 0;
        for (int i = 0; i < 10; i++) {
            misses += (test_value_int[i] != fb_int->GetHostPointer()[i]);
            misses += (test_value_float[i] != fb_float->GetHostPointer()[i]);
        }

        REQUIRE(misses == 0);
    }

    SECTION("Reflect Function Test") {
        fb_int->GetHostPointer()[5] = 30;
        fb_float->GetHostPointer()[5] = 0.3;

        fb_int->ReflectOnNative();
        fb_float->ReflectOnNative();

        REQUIRE(fb_int->GetHostPointer()[5] == 30);
        REQUIRE(fb_float->GetHostPointer()[5] == 0.3f);
    }

    SECTION("Deallocate Function Test") {
        fb_int->Free();
        fb_float->Free();

        REQUIRE(fb_int->GetHostPointer() == nullptr);
        REQUIRE(fb_float->GetNativePointer() == nullptr);
    }

    delete fb_int;
    delete fb_float;
}
