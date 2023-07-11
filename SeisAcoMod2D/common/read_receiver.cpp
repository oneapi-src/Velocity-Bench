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
#include <cmath>
#include <string>
#include <fstream>
#include <sstream>
#include "modelling.h"

using namespace std;

void read_recv2d(const char* recv_file)
{
	// Create map and store no. of receiver of each shot gather
	// to avoid receiver file reading mulitiple times
	map<int, int> map_srccount;
	map<int, int>::iterator it_srccount;
	int i, j, it;
	fstream fp_geom;
	float sx, sz, gx, gz;	
    string s_buffer, s_token[4];
    char sep = ',';

	geo2d_sp->nsrc = count_src(recv_file, map_srccount);
	cout<<"\n Total number of shot gathers are : "<<geo2d_sp->nsrc;

	geo2d_sp->nrec     = new int[     geo2d_sp->nsrc];
	geo2d_sp->src2d_sp = new crd2d_t[ geo2d_sp->nsrc];
	geo2d_sp->rec2d_sp = new crd2d_t*[geo2d_sp->nsrc];

	for(i = 0; i < geo2d_sp->nsrc; i++)
	{
		it_srccount = map_srccount.find(i+1);
        if (it_srccount != map_srccount.end()) {
            geo2d_sp->nrec[i] = it_srccount->second;
            geo2d_sp->rec2d_sp[i] = new crd2d_t[geo2d_sp->nrec[i]];
        }
	}

	fp_geom.open(recv_file, fstream::in);

	if( !fp_geom.is_open() )
	{
		cerr << "\n Error!!! Unable to open receiver geometry file : " << recv_file << " for reading";
        cerr << "\n receiver geometry file should contain following information at each line";
        cerr << "\n sx-coord,sz-coord,gx-coord,gz-coord            (comma is mandatory)";
        cerr << "\n and this file should contain the list sorted to sx-coord value";
		MPI::COMM_WORLD.Abort(-3);
	}	

	for(i = 0; i < geo2d_sp->nsrc; i++)
	{
		getline(fp_geom, s_buffer);
        // No need to check for blank line as it will be handled in count_src function for same file

        istringstream stm(s_buffer);

        it = 0;
        while(getline(stm, s_token[it], sep))
            it++;

        set_float(s_token[0], sx,      i, "Source X-coordinate");
        set_float(s_token[1], sz,      i, "Source Z-coordinate");
        set_float(s_token[2], gx,      i, "Receiver X-coordinate");
        set_float(s_token[3], gz,      i, "Receiver Z-coordinate");

        check_geom_coord(sx, sz, gx, gz);

        geo2d_sp->src2d_sp[i].x = sx;			geo2d_sp->src2d_sp[i].z = sz;	
		geo2d_sp->rec2d_sp[i][0].x = gx;		geo2d_sp->rec2d_sp[i][0].z = gz;
	
        for(j = 1; j < geo2d_sp->nrec[i]; j++)
		{		
			getline(fp_geom, s_buffer);
            istringstream stm(s_buffer);

            it = 0;
            while(getline(stm, s_token[it], sep))
                it++;

            set_float(s_token[2], gx,      i, "Receiver X-coordinate");
            set_float(s_token[3], gz,      i, "Receiver Z-coordinate");

            check_geom_coord(sx, sz, gx, gz);            

			geo2d_sp->rec2d_sp[i][j].x = gx;	geo2d_sp->rec2d_sp[i][j].z = gz;	
		}
	}
	fp_geom.close();
}// End of read_recv2d function

void check_geom_coord(float sx, float sz, float gx, float gz)
{
    int nx = mod_sp->nx, nz = mod_sp->nz;
    float dx = mod_sp->dx, dz = mod_sp->dz;
    int rem, quo;

    // Check source coordinate and receiver spacing is in terms of grid size or not 
    rem = (int) remquof(sx, dx, &quo);
    if( rem != 0 )
    {
        cerr<<"\n Error!!!   Source coordinate should be in multiples of grid size in that direction";
        cerr<<"\n Please correct x-coordinate for source(sx,sz): ["<<sx<<","<<sz<<"]  dx: "<<dx<<"\n";
        MPI::COMM_WORLD.Abort(-13);
    }
    rem = (int) remquof(sz, dz, &quo);
    if( rem != 0 )
    {
        cerr<<"\n Error!!!   Source coordinate should be in multiples of grid size in that direction";
        cerr<<"\n Please correct z-coordinate for source(sx,sz): ["<<sx<<","<<sz<<"]   dz: "<<dx<<"\n";
        MPI::COMM_WORLD.Abort(-14);
    }
    rem = (int) remquof(gx, dx, &quo);
    if( rem != 0 )
    {
        cerr<<"\n Error!!!   Receiver coordinate should be in multiples of grid size in that direction";
        cerr<<"\n Please correct x-coordinate of receiver(gx,gz): ["<<gx<<","<<gz<<"]";
        cerr<<"   for source(sx,sz): ["<<sx<<","<<sz<<"]\n";
        MPI::COMM_WORLD.Abort(-13);
    }
    rem = (int) remquof(gz, dz, &quo);
    if( rem != 0 )
    {
        cerr<<"\n Error!!!   Receiver coordinate should be in multiples of grid size in that direction";
        cerr<<"\n Please correct z-coordinate of receiver(gx,gz): ["<<gx<<","<<gz<<"]";
        cerr<<"   for source(sx,sz): ["<<sx<<","<<sz<<"]\n"; 
        MPI::COMM_WORLD.Abort(-14);
    }
    
    // Check source coordinate is within model or not 
    if( sx < 0 || sx > ((nx-1)*dx) )
    {
        cerr<<"\n Error!!!   Source coordinate should be within model bound";
        cerr<<"\n Please correct x-coordinate for source(sx,sz): ["<<sx<<","<<sz<<"]\n";
        MPI::COMM_WORLD.Abort(-16);
    }
    if( sz < 0 || sz > ((nz-1)*dz) )
    {
        cerr<<"\n Error!!!   Source coordinate should be within model bound";
        cerr<<"\n Please correct z-coordinate for source(sx,sz): ["<<sx<<","<<sz<<"]\n";
        MPI::COMM_WORLD.Abort(-17);
    } 
    if( gx < 0 || gx > ((nx-1)*dx) )
    {
        cerr<<"\n Error!!!   Receiver coordinate should be within model bound";
        cerr<<"\n Please correct x-coordinate of receiver(gx,gz): ["<<gx<<","<<gz<<"]";
        cerr<<"   for source(sx,sz): ["<<sx<<","<<sz<<"]\n";
        MPI::COMM_WORLD.Abort(-16);
    }
    if( gz < 0 || gz > ((nz-1)*dz) )
    {
        cerr<<"\n Error!!!   Receiver coordinate should be within model bound";
        cerr<<"\n Please correct z-coordinate of receiver(gx,gz): ["<<gx<<","<<gz<<"]";
        cerr<<"   for source(sx,sz): ["<<sx<<","<<sz<<"]\n";
        MPI::COMM_WORLD.Abort(-17);
    }
}// End of check_geom_coord function

int count_src(const char* recv_file, map<int, int>& map_srccount)
{
	int nsrc, nrec, it, i=0;
	fstream fp_geom;
	float sx_cur, sz_cur, sx_old, sz_old, tmp_gx, tmp_gz;	
    string s_buffer, s_token[5];
    char sep=',';

	map<int, int>::iterator it_srccount;

	fp_geom.open(recv_file, fstream::in);
	if( !fp_geom.is_open() )
	{
		cerr << "\n Error!!! Unable to open receiver geometry file : " << recv_file << " for reading";
        cerr << "\n receiver geometry file should contain following information at each line";
        cerr << "\n sx-coord,sz-coord,gx-coord,gz-coord            (comma is mandatory)";
        cerr << "\n and this file should contain the list sorted to sx-coord value";
		MPI::COMM_WORLD.Abort(-3);
	}	

    // Check file is empty or not
    if( fp_geom.peek() == ifstream::traits_type::eof() )
    {
        cerr << "\n Error!!! Provided receiver file: " << recv_file << " is empty\n";
        MPI::COMM_WORLD.Abort(-4);
    }

    // Read first coordinate
	getline(fp_geom, s_buffer);

    // Check for blank line
	if(s_buffer.empty())
    {
        cerr << "\n Error!!! Blank line is detected in receiver geometry file";
        cerr << "\n Please do not enter blank line in between\n";
        MPI::COMM_WORLD.Abort(-21);
    }

    istringstream stm(s_buffer);

    it = 0;
    while(getline(stm, s_token[it], sep))
    {
        it++;
        if(it > 4)
        {
            cerr << "\n Error!!!   Wrong number of arguments for source number: " << i+1;
            cerr << "\n Each line in file should contain exactly following parameters";
            cerr << " with comma seperated: ";
            cerr << "\n sx_mtr,sz_mtr,gx_mtr_gz_mtr";
            cerr << "\n Please correct the inputs and submit job again.....\n";
            MPI::COMM_WORLD.Abort(-26);
        }
    }

    if(it != 4)
    {
        cerr << "\n Error!!!   Wrong number of arguments for source number: " << i+1;
        cerr << "\n Each line in file should contain exactly following parameters";
        cerr << " with comma seperated: ";
        cerr << "\n sx_mtr,sz_mtr,gx_mtr_gz_mtr";
        cerr << "\n Please correct the inputs and submit job again.....\n";
        MPI::COMM_WORLD.Abort(-26);
    }

    nsrc = 1;
    nrec = 1;
    i    = 1;

    set_float(s_token[0], sx_old,      i, "Source X-coordinate");
    set_float(s_token[1], sz_old,      i, "Source Z-coordinate");
    set_float(s_token[2], tmp_gx,      i, "Receiver X-coordinate");
    set_float(s_token[3], tmp_gz,      i, "Receiver Z-coordinate");

	// Read rest of the file
	while( getline(fp_geom, s_buffer) )
	{
		if(s_buffer.empty())
        {
            cerr << "\n Error!!! Blank line is detected in receiver geometry file";
            cerr << "\n Please do not enter blank line in between\n";
            MPI::COMM_WORLD.Abort(-21);
        }

        istringstream stm(s_buffer);

        it = 0;
        while(getline(stm, s_token[it], sep))
        {
            it++;
            if(it > 4)
            {
                cerr << "\n Error!!!   Wrong number of arguments for source number: " << i+1;
                cerr << "\n Each line in file should contain exactly following parameters";
                cerr << " with comma seperated: ";
                cerr << "\n sx_mtr,sz_mtr,gx_mtr_gz_mtr";
                cerr << "\n Please correct the inputs and submit job again.....\n";
                MPI::COMM_WORLD.Abort(-26);
            }
        }

        if(it != 4)
        {
            cerr << "\n Error!!!   Wrong number of arguments for source number: " << i+1;
            cerr << "\n Each line in file should contain exactly following parameters";
            cerr << " with comma seperated: ";
            cerr << "\n sx_mtr,sz_mtr,gx_mtr_gz_mtr";
            cerr << "\n Please correct the inputs and submit job again.....\n";
            MPI::COMM_WORLD.Abort(-26);
        }

        set_float(s_token[0], sx_cur,      i, "Source X-coordinate");
        set_float(s_token[1], sz_cur,      i, "Source Z-coordinate");
        set_float(s_token[2], tmp_gx,      i, "Receiver X-coordinate");
        set_float(s_token[3], tmp_gz,      i, "Receiver Z-coordinate");

		if(sx_cur == sx_old && sz_cur == sz_old)
		{
			nrec++;		
		}
		else if(sx_cur > sx_old)
		{
			// Insert source and no. of receiver for that source to map
			it_srccount = map_srccount.end();
			map_srccount.insert(it_srccount, pair<int, int> (nsrc, nrec));
			nsrc++;
            nrec = 1;
            i++;
			sx_old = sx_cur;
			sz_old = sz_cur;		
		}
        else
        {
            cerr << "\n Error!!! Geometry file should be sorted on sx-coord value";
            cerr << "\n source X: " << sx_cur << " with all receiver pairs should occure before Source X:";
            cerr << sx_old << "\n";
            cerr << "\n Please correct the geometry file and resubmit job\n";
            MPI::COMM_WORLD.Abort(-22);
        }
	}
	fp_geom.close();

	// Insert for last shot
    it_srccount = map_srccount.end();
	map_srccount.insert(it_srccount, pair<int, int> (nsrc, nrec));	

	// Return total shot gather count
	return nsrc;
}// End of count_src function
