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
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "modelling.h"

using namespace std;

void read2d(const char* file, float** arr, int nx, int nz)
{
	FILE* fp_mod;

	fp_mod = fopen(file, "r");
	if(fp_mod == NULL)
	{
		cerr << "\n Error!!! Unable to open input mode : " << file;
        MPI::COMM_WORLD.Abort(-2);
        return;
	}

    fseek(fp_mod, 0, SEEK_END);
    float file_size = (float) ftell(fp_mod);
    rewind(fp_mod);

    // Checking whether file is having nx*nz elements or not
    float rem = file_size/(float)(nx*nz);
    if(rem != 4.0f)     // 4.0f = size of float
    {
        cerr << "\n Error!!! Either input file " << file << " is empty or it doesn't contain ";
        cerr << "NX*NZ elements.....\n";
        MPI::COMM_WORLD.Abort(-2);
        fclose(fp_mod);
        return;
    }

	for(int i = 0; i < nx; i++)
	{
		size_t nread = fread(arr[i], sizeof(float), nz, fp_mod);
        if (nread != nz) {
            cerr << "Reading error\n";
            MPI::COMM_WORLD.Abort(-2);
            fclose(fp_mod);
            return;
        }
	}

	fclose(fp_mod);

}//End of read2d function

//get the min and max value ....
void minmax2d(float** arr, int nx, int nz, float& vmin, float& vmax)
{
    int ix, iz;

    vmax = arr[0][0];   //we can assign int val or float directly rather than copying value
    vmin = arr[0][0];   //we can assign int val or float directly rather than copying value

    for(ix = 0; ix < nx; ix++)
    {
        for(iz = 0; iz < nz; iz++)
        {
            if(arr[ix][iz] < vmin)
                vmin = arr[ix][iz];
            if(arr[ix][iz] > vmax)
                vmax = arr[ix][iz];
        }
    }

}//End of minmax2d function
