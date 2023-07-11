/* Modifications Copyright (C) 2023 Intel Corporation
 *
 * This Program is subject to the terms of the GNU General Public License v3.0 only.​
 * If a copy of the license was not distributed with this file, ​
 * you can obtain one at https://spdx.org/licenses/GPL-3.0-only.html
 *​
 *
 * SPDX-License-Identifier: GPL-3.0-only
 */

#include "mpi.h"
#include <iostream>
#include <cmath>

#include "modelling.h"

using namespace std;

double Modelling_worker(const char* job)
{
    std::chrono::steady_clock::time_point time_start_;
    std::chrono::steady_clock::time_point time_end_;
    double time_total_ = 0.0;

    TIMER_START_()

	map<string, string> map_json;

	int rank = MPI::COMM_WORLD.Get_rank();
	int size = MPI::COMM_WORLD.Get_size();

    set_job_name(job);

    // Read input job card
	int nobject = read_json_objects(job, map_json);

    // Set input parameters
    set_json_object(map_json);

    // Receive workload count
	wrkld_t* wrkld_sp   = new wrkld_t[size];
	geo2d_t* mygeo2d_sp = new geo2d_t[1];

    calculate_workload(geo2d_sp->nsrc, wrkld_sp);

	int my_nsrc = wrkld_sp[rank].myNsrc;
	cout << "\n Shot to be handled for rank " << rank << " is : " << my_nsrc;

	mygeo2d_sp->nrec     = new int[my_nsrc];
	mygeo2d_sp->src2d_sp = new crd2d_t[my_nsrc];
	mygeo2d_sp->rec2d_sp = new crd2d_t*[my_nsrc];

	receive_workload(wrkld_sp, mygeo2d_sp);

	// Read velocity and density model
	mod_sp->vel2d = new float*[mod_sp->nx];
    mod_sp->rho2d = new float*[mod_sp->nx];
	for(int ix = 0; ix < mod_sp->nx; ix++)
    {
        mod_sp->vel2d[ix] = new float[mod_sp->nz];
        mod_sp->rho2d[ix] = new float[mod_sp->nz];
    }

	read2d(mod_sp->velfile,  mod_sp->vel2d, mod_sp->nx, mod_sp->nz);
    read2d(mod_sp->densfile, mod_sp->rho2d, mod_sp->nx, mod_sp->nz);

    minmax2d(                mod_sp->vel2d, mod_sp->nx, mod_sp->nz, mod_sp->vmin, mod_sp->vmax);
    cout << "\n Min. Velocity : " << mod_sp->vmin << "\t Max. Velocity : " << mod_sp->vmax;

    // Check stability and dispersion
    check_stability();
    TIMER_END_()

    // Call wave propagation
    modelling_module(my_nsrc, mygeo2d_sp);

    delete[] mygeo2d_sp->nrec;
    delete[] mygeo2d_sp->src2d_sp;
    delete[] mygeo2d_sp->rec2d_sp;
    delete[] mygeo2d_sp;
    delete[] wrkld_sp;

    TIMER_PRINT_("time to subtract from total")
    return time_total_;
}//End of Modelling_worker
