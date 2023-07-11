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

double Modelling_master(const char* job)
{
    std::chrono::steady_clock::time_point time_start_;
    std::chrono::steady_clock::time_point time_end_;
    double time_total_ = 0.0;

    TIMER_START_()
    // double starttime = MPI::Wtime();

	map<string, string> map_json;

	int rank = MPI::COMM_WORLD.Get_rank();
	int size = MPI::COMM_WORLD.Get_size();

    set_job_name(job);                              // sets job_sp->jobname based on input file name

    // Read input job card                          // creates key: value map from input file content
	int nobject = read_json_objects(job, map_json);

    // Set input parameters
    set_json_object(map_json);                      // populates mod_sp, wave_sp and job_sp from map_json

    // Set geometry
	if(job_sp->geomflag == 0)			            // Read source-recievers
	{
	    read_recv2d(job_sp->geomfile);
	}
	else if(job_sp->geomflag == 1)                  // Read source and set recievers - create geometry
	{
		read_source2d(job_sp->shotfile);            // Takes this route [fills up geo2d_sp using info in shotfile]
	}

	// Shot workload distribution calculation and send
	wrkld_t* wrkld_sp   = new wrkld_t[size];
	geo2d_t* mygeo2d_sp = new geo2d_t[1];

    calculate_workload(geo2d_sp->nsrc, wrkld_sp);   // Divides the sources among the available MPI nodes

	int my_nsrc = wrkld_sp[rank].myNsrc;
	cout << "\n Shot to be handled for rank " << rank << " is : " << my_nsrc;

	mygeo2d_sp->nrec     = new int[my_nsrc];
	mygeo2d_sp->src2d_sp = new crd2d_t[my_nsrc];
	mygeo2d_sp->rec2d_sp = new crd2d_t*[my_nsrc];

	send_workload(wrkld_sp);                        // Sends to each worker portion of geo2d_sp alloted to it

	master_workload(wrkld_sp, mygeo2d_sp);          // Saves master's portion of geo2d_sp in mygeo2d_sp

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

    // double endtime = MPI::Wtime();

    // cout<<"\n Total time of execution : "<< ((endtime-starttime)/3600) <<"  .Hrs";
    // cout << "\n Total time of execution : " << (endtime - starttime) << "  seconds\n";

    delete[] mygeo2d_sp->nrec;
    delete[] mygeo2d_sp->src2d_sp;
    delete[] mygeo2d_sp->rec2d_sp;
    delete[] mygeo2d_sp;
    delete[] wrkld_sp;

    TIMER_PRINT_("time to subtract from total")
    return time_total_;
}//End of Modelling_master
