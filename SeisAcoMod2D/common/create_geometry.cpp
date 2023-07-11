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
#include <cmath>
#include <cstring>
#include <string>
#include <fstream>
#include <sstream>

#include "modelling.h"

using namespace std;

void read_source2d(const char* shot_file)
{
	int i, nsrc=0, count, it, result;
	FILE *fp_src;

    string s_buffer, s_token[7], s_missing("");
    char sep = ',';
    fstream fp_in;

	fp_src = fopen(shot_file, "r");

	char buffer[256], bufferstring[256];

	if(fp_src == NULL)
	{
		cerr << "\nError!!! Unable to open shot file : " << shot_file;
		MPI::COMM_WORLD.Abort(-2);
        return;
	}

	while(fgets(buffer, 256, fp_src))
	{
		//if string contain "comment" word then it is comment line, and also check whether line is blank or not
		if( (!strstr(buffer,"comment")) && (sscanf(buffer, "%s", bufferstring)==1))
			nsrc++;
	}
	rewind(fp_src);

	//Set number of source and allocate memory
	geo2d_sp->nsrc = nsrc;

    cout << "\n Total number of shot gathers are : " << geo2d_sp->nsrc;

	geo2d_sp->nrec     = (int*)      calloc(geo2d_sp->nsrc, sizeof(int));
	geo2d_sp->src2d_sp = (crd2d_t* ) calloc(geo2d_sp->nsrc, sizeof(crd2d_t));
	geo2d_sp->rec2d_sp = (crd2d_t**) calloc(geo2d_sp->nsrc, sizeof(crd2d_t*));

	int nrec, geotype, default_geotype;
	float sx, sz, ngeoph, nearoff;
    size_t buffSize = 256;

    fp_in.open(shot_file, fstream::in);

    // Check file opened successfully or not
    if( !fp_in.is_open() )
    {
        cerr << "\n Error!!! Unable to open shot file: " << shot_file;
        cerr << "\n Please check file name in input job card\n";
        MPI::COMM_WORLD.Abort(-23);
    }

    // Check for empty file
    if( fp_in.peek() == ifstream::traits_type::eof() )
    {
        cerr << "\n Error!!! Provided shot file: " << shot_file << " is empty\n";
        MPI::COMM_WORLD.Abort(-24);
    }

	for(i = 0; i < geo2d_sp->nsrc; i++)
	{
        getline(fp_in, s_buffer);
        if(s_buffer.empty())
        {
            cerr << "\n Error!!!   Blank line detected instead of information";
            cerr << "\n Please enter information at each line one by one, dont keep line ";
            cerr << "blank in between\n";
            MPI::COMM_WORLD.Abort(-20);
        }

        istringstream stm(s_buffer);

        it = 0;
        while(getline(stm, s_token[it], sep))
        {
            it++;
            if(it > 6)
            {
                cerr << "\n Error!!!   Wrong number of arguments for source number: " << i+1;
                cerr << "\n Each line in file should contain exactly following parameters";
                cerr << " with comma seperated: ";
                cerr << "\n sx_mtr,sz_mtr,no_of_receivers,near_offset,receiver_spacing_mtr,geom_type";
                cerr << "\n Please correct the inputs and submit job again.....\n";
                MPI::COMM_WORLD.Abort(-25);
            }
        }

        if(it != 6)
        {
            cerr << "\n Error!!!   Wrong number of arguments for source number: " << i+1;
            cerr << "\n Each line in file should contain exactly following parameters";
            cerr << " with comma seperated: ";
            cerr << "\n sx_mtr,sz_mtr,no_of_receivers,near_offset,receiver_spacing_mtr,geom_type";
            cerr << "\n Please correct the inputs and submit job again.....\n";
            MPI::COMM_WORLD.Abort(-25);
        }

        set_float(s_token[0], sx,      i, "Source X-coordinate");
        set_float(s_token[1], sz,      i, "Source Z-coordinate");
        set_int  (s_token[2], nrec,    i, "Number of receivers");
        set_float(s_token[3], nearoff, i, "Near offset");
        set_float(s_token[4], ngeoph,  i, "Receiver spacing");
        set_int  (s_token[5], geotype, i, "Geometry type");
        // Setting defualt geophone geometry type, it should be same for all sources
        if(i == 0)
            default_geotype = geotype;
 
        if(default_geotype != geotype)
        {
            cerr << "\n Error!!!   Geophone geometry type should be same for all shot gahter";
            cerr << "\n Please correct geophone geometry type for source(sx,sz): ["<<sx<<","<<sz<<"]\n";
            MPI::COMM_WORLD.Abort(-11);
        }

        if(nrec <= 0)
        {
            cerr << "\n Error!!! Number of receivers cannot be zero or negative";
            cerr << "\n Please correct Number of receivers or source(sx,sz): ["<<sx<<","<<sz<<"]\n";
            MPI::COMM_WORLD.Abort(-18);
        }

		geo2d_sp->nrec[i]       = nrec;
		geo2d_sp->src2d_sp[i].x = sx;
        geo2d_sp->src2d_sp[i].z = sz;
		geo2d_sp->rec2d_sp[i]   = (crd2d_t*) calloc(nrec, sizeof(crd2d_t));

		//Set receiver for each shot depending on information read
		set_receiver2d(sx, sz, nrec, nearoff, geotype, ngeoph, i);
	}
	fp_in.close();

}//End of read_source file

void set_int(  string token, int&   var, int src_indx, const char* err_msg)
{

    try
    {
        if( !token.empty() ) 
            var = stoi(token);
        else
        {
            cerr<<"\n Error!!! missing "<<err_msg<<" information in geometry file ";
            cerr<<"for source number: "<<src_indx+1<<"\n";
            MPI::COMM_WORLD.Abort(-19);
        }
    }
    catch(const std::invalid_argument& err)
    {
        cerr<<"\n Error!!! missing "<<err_msg<<" information in geometry file ";
        cerr<<"for source number: "<<src_indx+1<<"\n";
        MPI::COMM_WORLD.Abort(-19);
    }
}// End of set_int function

void set_float(string token, float& var, int src_indx, const char* err_msg)
{

    try
    {
        if( !token.empty() ) 
            var = stof(token);
        else
        {
            cerr<<"\n Error!!! missing "<<err_msg<<" information in geometry file ";
            cerr<<"for source number: "<<src_indx+1<<"\n";
            MPI::COMM_WORLD.Abort(-19);
        }   
    }
    catch(const std::invalid_argument& err)
    {
        cerr<<"\n Error!!! missing "<<err_msg<<" information in geometry file ";
        cerr<<"for source number: "<<src_indx+1<<"\n";
        MPI::COMM_WORLD.Abort(-19);
    }
}// End of set_float function

void check_geom_para(float sx, float sz, float ngeoph)
{
    int   nx = mod_sp->nx, nz = mod_sp->nz;   
    float dx = mod_sp->dx, dz = mod_sp->dz;
    int rem, quo;

    // Check source coordinate and receiver spacing is in terms of grid size or not
    rem = (int) remquof(sx, dx, &quo);
    if( rem != 0 )
    {
        cerr<<"\n Error!!!   Source coordinate should be in multiples of grid size in that direction";
        cerr<<"\n Please correct x-coordinate for source(sx,sz): ["<<sx<<","<<sz<<"]   dx: "<<dx<<"\n";
        MPI::COMM_WORLD.Abort(-13);
    }
    rem = (int) remquof(sz, dz, &quo);
    if( rem != 0 )
    {
        cerr<<"\n Error!!!   Source coordinate should be in multiples of grid size in that direction";
        cerr<<"\n Please correct z-coordinate for source(sx,sz): ["<<sx<<","<<sz<<"]   dz: "<<dz<<"\n";
        MPI::COMM_WORLD.Abort(-14);
    }
    rem = (int) remquof(ngeoph, dx, &quo);
    if( rem != 0 )
    {
        cerr<<"\n Error!!!   Geophone spacing should be in multiples of grid size in that direction";
        cerr<<"\n Please correct geophone spacing for source(sx,sz): ["<<sx<<","<<sz<<"]\n";
        MPI::COMM_WORLD.Abort(-15);
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

}// End of check_geom_para function

void set_receiver2d(float sx, float sz, int nrec, float nearoff, int geotype, float ngeoph, int srcIndx)
{
    	
    check_geom_para(sx, sz, ngeoph);

	if(geotype == 1)				//If Geometry is split
	{	
		set_geom2d_split(sx, sz, nrec, nearoff, ngeoph, srcIndx);	
	}//End of split
	
	else if(geotype == 2)			//If Geometry is endon-left
	{
		set_geom2d_endonleft(sx, sz, nrec, nearoff, ngeoph, srcIndx);
	}//End of endon-left
	
	else if(geotype == 3)			//If Geometry is endon-right
	{
		set_geom2d_endonright(sx, sz, nrec, nearoff, ngeoph, srcIndx);
	}//End of endon-right
    else
    {
        cerr<<"\n Error!!!   Wrong geometry type parameter: "<<geotype;
        cerr<<"\n Please specify correct geometry type 1:split     2:endon-left     3:endon-right";
        MPI::COMM_WORLD.Abort(-12);
    }

}//End of set_receiver2d

void set_geom2d_split(float sx, float sz, int nrec, float nearoff, float ngeoph, int srcIndx)
{
	int nx = mod_sp->nx;
	float recv_d = ngeoph;
	
	float start, end;	
    float i;	
    int j, tmp1, tmp2;	
	
	if(nearoff == 0.0f)
    {
	  	tmp1 = (sx - 0.0f)/recv_d;						//recv points in left direction
	   	tmp2 = ( ((nx-1)*mod_sp->dx) - sx )/recv_d;		//recv points in right direction

	   	if(tmp1 < (nrec/2))								//less recv points in left
	   	{	
			start = sx - (tmp1*recv_d);
			end = start + ((nrec-1)*recv_d);
	   	}
	   	else if (tmp2 < (nrec/2))						//less recv points in right
	   	{
			start = sx - ((nrec-tmp2-1)*recv_d);
			end = start + ((nrec-1)*recv_d);
	   	}	
	   	else											//sufficient point in left&right
	   	{
			start = sx - ((nrec/2) * recv_d);
			end = start + ((nrec-1) * recv_d);
	   	}			
	}
    else //split spread and nearoff not 0  
    {
	   	tmp1 = (sx - 0.0f)/recv_d;     					//recv points in left direction
        tmp2 = ( ((nx-1)*mod_sp->dx) - sx )/recv_d;    	//recv points in right direction

        if(tmp1 < (nrec/2))         					//less recv points in left
        {
            start = sx - (tmp1*recv_d);
            end = start + ((nrec-1)*recv_d) + recv_d;
        }
        else if (tmp2 < (nrec/2))   					//less recv points in right
        {
            start = sx - ((nrec-tmp2)*recv_d);
            end = start + (nrec*recv_d);
        }
        else                        					//sufficient point in left&right
        {
            start = sx - ((nrec/2) * recv_d);
	      	end = start + ((nrec-1) * recv_d);
	
			if((start-recv_d) >= 0)
	   			start -= recv_d; 	
			else
	   			end += recv_d;
        }	
	}

	//check whether first reciever is in model or not
    if(start < 0.0f)
    {
      	cerr<<"\n Error!!! for source: "<<sx<<","<<sz<<"  	reciver x-coord: "<<start<<" goes out of model";
		MPI::COMM_WORLD.Abort(-3);
    }

	//check whether last reciever is in bound or not
    if(end > ((nx-1)*mod_sp->dx)) 
    {
       	cerr<<"\n Error!!! for source: "<<sx<<","<<sz<<"  	reciver x-coord: "<<end<<" goes out of model";
      	MPI::COMM_WORLD.Abort(-3);
    }

	i = start;
	j = 0;
    while( i <= end )
    {
      	if(nearoff == 0.0f)
		{
           	geo2d_sp->rec2d_sp[srcIndx][j].x = i;
            geo2d_sp->rec2d_sp[srcIndx][j].z = sz;	
	   	}
	   	else
	   	{
			if(i != sx)
			{
               	geo2d_sp->rec2d_sp[srcIndx][j].x = i;
                geo2d_sp->rec2d_sp[srcIndx][j].z = sz;	  
			}       	
	   	}
	   	i += recv_d;	
		j++;	
	}
}//End of set_geom2d_split function

void set_geom2d_endonleft(float sx, float sz, int nrec, float nearoff, float ngeoph, int srcIndx)
{
	int nx = mod_sp->nx;
	float recv_d = ngeoph;

	float start, end;	
    float i;	
    int j, tmp1, tmp2;	

	if(nearoff == 0.0f)
    {
    	end = sx;
        start = end - ((nrec-1)*recv_d);
    }
    else
    {
        end = sx - recv_d;
        start = end - ((nrec-1)*recv_d);
    }

    //check whether first reciever is in model or not
    if(start < 0.0f )
    {
    	cerr<<"\n Error!!! for source: "<<sx<<","<<sz<<"  	reciver x-coord: "<<start<<" goes out of model";
	   	MPI::COMM_WORLD.Abort(-3);
    }
    //check whether last reciever is in bound or not
    if( end > ((nx-1)*mod_sp->dx) )
    {
    	cerr<<"\n Error!!! for source: "<<sx<<","<<sz<<"  	reciver x-coord: "<<end<<" goes out of model";
        MPI::COMM_WORLD.Abort(-3);
    }

    i = start;
	j = 0;
    while(i <= end)
    {

        geo2d_sp->rec2d_sp[srcIndx][j].x = i;		geo2d_sp->rec2d_sp[srcIndx][j].z = sz;	
        i+=recv_d;
		j++;
    }
	
}//End of set_geom2d_endonleft function

void set_geom2d_endonright(float sx, float sz, int nrec, float nearoff, float ngeoph, int srcIndx)
{
	int nx = mod_sp->nx;
	float recv_d = ngeoph;
	
	float start, end;	
    float i;	
    int j, tmp1, tmp2;	

	if( nearoff == 0.0f )
    {
       start = sx;
       end = start + ((nrec-1)*recv_d);
    }
    else
    {
        start = sx + recv_d;
        end = start + ((nrec-1)*recv_d);
    }

    //check whether first reciever is in model or not
    if(start < 0.0f)
    {
    	cerr<<"\n Error!!! for source: "<<sx<<","<<sz<<"  	reciver x-coord: "<<start<<" goes out of model";
	   	MPI::COMM_WORLD.Abort(-3);
    }

    //check whether last reciever is in bound or not
    if(end > ((nx-1)*mod_sp->dx))
    {
    	cerr<<"\n Error!!! for source: "<<sx<<","<<sz<<"  	reciver x-coord: "<<end<<" goes out of model";
        MPI::COMM_WORLD.Abort(-3);
    }

    i = start;
	j = 0;
    while(i <= end)
    {

       	geo2d_sp->rec2d_sp[srcIndx][j].x = i;		geo2d_sp->rec2d_sp[srcIndx][j].z = sz;	
        i+=recv_d;
		j++;
    }

}//End of set_geom2d_endonright function

void print_single_source_geometry()
{
    FILE *fp = fopen("test_geometry.txt\0", "w");
   
    int nrec = geo2d_sp->nrec[0], j;

    fprintf(fp, "%f %f\n", geo2d_sp->src2d_sp[0].z, geo2d_sp->src2d_sp[0].x);

    for(j = 0; j < nrec; j++)
        fprintf(fp, "%f %f\n", geo2d_sp->rec2d_sp[0][j].z, geo2d_sp->rec2d_sp[0][j].x);

    fclose(fp);

}// End of print_single_source_geometry function
