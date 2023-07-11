/* Modifications Copyright (C) 2023 Intel Corporation
 *
 * This Program is subject to the terms of the GNU General Public License v3.0 only.​
 * If a copy of the license was not distributed with this file, ​
 * you can obtain one at https://spdx.org/licenses/GPL-3.0-only.html
 *​
 *
 * SPDX-License-Identifier: GPL-3.0-only
 */

#include <mpi.h>
#include <iostream>
#include <cmath>
#include "modelling.h"

using namespace std;

void check_stability()
{
    float vel_min, vel_max, fm, dx, dz, dt;

    vel_min  = mod_sp->vmin;
    vel_max  = mod_sp->vmax;
    dx       = mod_sp->dx;
    dz       = mod_sp->dz;
    dt       = wave_sp->fd_dt_sec;
    fm       = 2.0f * wave_sp->dom_freq;

    // int order  = job_sp->fdop;
    float hmin = min2(dx, dz);
    float hmax = max2(dx, dz);

    //Stability check
    // dt < (0.606*hmin/vmax)  - 0.606 for 4th order, 1/sqrt(2) for 2nd order
    float criteria = (0.606f * hmin) / vel_max;

    if(dt > criteria)
    {
        cout << "\n****************************************";
        cout << "\n*                                      *";
        cout << "\n*     STABILITY CONDITION VIOLATED     *";
        cout << "\n*                                      *";
        cout << "\n****************************************";

        cout << "\n Current Time Step(dt)     : " << dt       << " sec ";
        cout << "\n Required Time Step(dt)    : " << criteria << " sec ";
        MPI::COMM_WORLD.Abort(-5);        
    }
    else
    {
        cout << "\n Current Time Step(dt)     : " << dt       << " sec ";
        cout << "\n Required Time Step(dt)    : " << criteria << " sec ";
        cout << "\n     STABILITY CONDITION FULFILLED      \n";
    }

    // Dispersion check  -  maxfreq < vmin/(hmax*5)
    criteria = vel_min / (hmax * 5.0f);

    if(fm > criteria)
    {
        cout << "\n*************************************************";
        cout << "\n*                                               *";
        cout << "\n*   Non-dispersion relation not satisfied!      *";
        cout << "\n*                                               *";
        cout << "\n*************************************************";
        cout << "\n Current peak frequency     : " << wave_sp->dom_freq;
        cout << "\n Required peak frequency    : " << criteria / 2.0f;
        MPI::COMM_WORLD.Abort(-5);
    }
    else
    {
        cout << "\n Current peak frequency     : " << wave_sp->dom_freq;
        cout << "\n Required peak frequency    : " << criteria / 2.0f;
        cout << "\n     GRID SPACING CRITERIA (DISPERSION) FULFILLED \n";
    }
}// End of stability_check function
