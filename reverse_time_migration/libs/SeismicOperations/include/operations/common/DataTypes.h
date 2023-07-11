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
// Created by zeyad-osama on 19/09/2020.
//

#ifndef OPERATIONS_LIB_BASE_DATA_TYPES_H
#define OPERATIONS_LIB_BASE_DATA_TYPES_H

/**
 * @brief Axis definitions
 */
#define Y_AXIS 0
#define Z_AXIS 1
#define X_AXIS 2

/// Unsigned integer type to be used for strictly positive numbers.
typedef unsigned int uint;

/**
 * @brief Enums for the supported order lengths.
 *
 * @note Here we put values for each enum variable
 * if we make instance of HALF_LENGTH and assign it to O_2 for example
 * it will return 1;
 */
enum HALF_LENGTH {
    O_2 = 1, O_4 = 2, O_8 = 4, O_12 = 6, O_16 = 8
};

enum PHYSICS {
    ACOUSTIC, ELASTIC
};
enum APPROXIMATION {
    ISOTROPIC, VTI, TTI
};
enum EQUATION_ORDER {
    SECOND, FIRST
};
enum GRID_SAMPLING {
    UNIFORM, VARIABLE
};
enum INTERPOLATION {
    NONE, SPLINE, TRILINEAR
};
enum ALGORITHM {
    RTM, FWI, PSDM, PSTM
};

/**
 * @brief Struct describing the number of points in our grid.
 */
struct GridSize {
public:
    GridSize() = default;

    GridSize(uint _nx, uint _ny, uint _nz)
            : nx(_nx), ny(_ny), nz(_nz) {}

public:
    uint nx;
    uint nz;
    uint ny;
};

/**
 * @brief The step size in each direction.
 */
struct CellDimensions {
public:
    CellDimensions() : dx(0.0f), dz(0.0f), dy(0.0f){}

    CellDimensions(float _dx, float _dy, float _dz)
            : dx(_dx), dy(_dy), dz(_dz) {}

public:
    float dx;
    float dz;
    float dy;
};

/**
 * @brief Point co-ordinates in 3D.
 */
struct Point3D {
public:
    Point3D(): x(0), z(0), y(0){}

    Point3D(uint _x, uint _y, uint _z)
            : x(_x), y(_y), z(_z) {}

    // Copy constructor
    Point3D(const Point3D &P) {
        x = P.x;
        y = P.y;
        z = P.z;
    }

    void operator=(const Point3D &P) {
        x = P.x;
        y = P.y;
        z = P.z;
    }

    bool operator==(const Point3D &P) {
        bool value = false;
        value = ((x == P.x) && (y == P.y) && (z == P.z));

        return value;
    }

public:
    uint x;
    uint z;
    uint y;
};

/**
 * @brief Floating Point co-ordinates in 3D.
 */
struct FPoint3D {
public:
    FPoint3D() : x(0.0f), z(0.0f), y(0.0f){} 

    FPoint3D(float _x, float _y, float _z)
            : x(_x), y(_y), z(_z) {}

    // Copy constructor
    FPoint3D(const FPoint3D &P) {
        x = P.x;
        y = P.y;
        z = P.z;
    }

    FPoint3D &operator=(const FPoint3D &P) = delete; 

    bool operator==(const FPoint3D &P) const {
        bool value = false;
        value = ((x == P.x) && (y == P.y) && (z == P.z));

        return value;
    }

public:
    float x;
    float z;
    float y;
};

/**
 * @brief Integer point in 3D.
 */
struct IPoint3D {
public:
    IPoint3D() : x(0), z(0), y(0){} 

    IPoint3D(int _x, int _y, int _z)
            : x(_x), y(_y), z(_z) {}

public:
    int x;
    int z;
    int y;
};

/**
 * @brief An object representing our full window size,
 * will not match grid size if a window model is applied.
 */
struct WindowProperties {
public:
    WindowProperties() : wnx(0), wnz(0), wny(0){} 

    WindowProperties(uint _wnx, uint _wnz, uint _wny, Point3D _window_start)
            : wnx(_wnx), wny(_wny), wnz(_wnz), window_start(_window_start) {}

public:
    /**
     * @brief Will be set by traces to the appropriate
     * start of the velocity (including boundary).
     *
     * As the trace manager knows the shot location so it will change
     * the start of the window according to the shot location in each shot
     */
    Point3D window_start;

    /// Will be set by ModelHandler to match all parameters properties.
    uint wnx;
    uint wnz;
    uint wny;
};

/**
 * @brief Parameters needed for the modelling operation.
 */
struct ModellingConfiguration {
public:
    /// Starting point for the receivers.
    Point3D ReceiversStart;
    /// The increment step of the receivers.
    Point3D ReceiversIncrement;
    /// The end point of the receivers exclusive.
    Point3D ReceiversEnd;
    /// The source point for the modelling.
    Point3D SourcePoint;
    /// The total time of the simulation in seconds.
    float TotalTime;

    ModellingConfiguration() : TotalTime(0) {}
};

#endif // OPERATIONS_LIB_BASE_DATA_TYPES_H
