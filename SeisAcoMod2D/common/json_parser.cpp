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

#include "modelling.h"

using namespace std;

void set_job_name(const char* job)  // sets job_sp->jobname based on input file name
{
    char tmp[256];
    char *loc, *loc1;

    strncat(tmp, job, 255);
    // strcpy(tmp, job);

    loc = strrchr(tmp, '/');
    if(loc == NULL)
    {
        loc = tmp;
        loc1 = strrchr(loc, '.');
        strncpy(loc1, "\0", 1);
        strncat(job_sp->jobname, loc, 255);
        // strcpy(job_sp->jobname, loc);
    }
    else
    {
        loc1 = strrchr(loc, '.');
        strncpy(loc1, "\0", 1);
        strncat(job_sp->jobname, (loc+1), 255);
        // strcpy(job_sp->jobname, (loc+1));
    }

    strcat(job_sp->jobname, "_");

}// End of set_job_name function

int read_json_objects(const char* job, map<string, string>& map_json)   // creates key: value map from input file content
{
    int nobject = 0;
	char cline[256];
	char varname[32], value[224];  	
	FILE *fp_in;  

	pair<map<string, string>::iterator, bool> status;

 	fp_in = fopen(job, "r");
    if(fp_in == NULL)
    {
        cerr << "\n Error!!! Unable to open input job card : " << job << endl;
        MPI::COMM_WORLD.Abort(-2);
        return -1;
    }

	while(fgets(cline, 256, fp_in))
	{
		// if the current line is not blank or comment
		if( ((strstr(cline, "\"")) && (strstr(cline, ":"))) && (!strstr(cline, "comment")) && (!strstr(cline, "Comment")) )
		{
			strcpy(varname, "\0");
			strcpy(value, "\0");

			sscanf(cline, " \"%[^\"]\" : \"%[^\"]\" ", varname, value);

			// Insert varname and value into map
			status = map_json.insert(pair<string, string>(varname, value));
			if(status.second == false)
			{
				cout << "\n Element " << status.first->first << " already exists with value of " << status.first->second;
			}

			nobject++;
		}// End of if

	}// End of while

	fclose(fp_in);

	return nobject;

}// End of read_json_objects function

void mpi_error(const char* msg)
{
    cout<<"\n Error!!! missing "<<msg<<" information in json file....\n";
    MPI::COMM_WORLD.Abort(-6);

}// End of mpi_error function

void str_to_int(string token, int& var, const char* err_msg)
{
    try
    {   
        if(!token.empty())
            var = stoi(token);
        else
            mpi_error(err_msg);
    }
    catch(const std::invalid_argument& err)
    {
        mpi_error(err_msg);
    }
}// End of str_to_int function

void str_to_float(string token, float& var, const char* err_msg)
{
    try
    {
        if(!token.empty())
            var = stof(token);
        else
            mpi_error(err_msg);
    }
    catch(const std::invalid_argument& err)
    {
        mpi_error(err_msg);
    }
}// End of str_to_float function

void set_json_object(map<string, string> map_json) 
{
    map<string, string>::iterator it_json;
   
    // Model parameter
    // Velocity file
    it_json = map_json.find("Velocity");
    if(it_json != map_json.end())
        strncat(mod_sp->velfile,       it_json->second.c_str(), 223);
        // strcpy(mod_sp->velfile,       it_json->second.c_str());
    else
        mpi_error("velocity file name");

    // Density file    
    it_json = map_json.find("Density");
    if(it_json != map_json.end())
        strncat(mod_sp->densfile,      it_json->second.c_str(), 223);
        // strcpy(mod_sp->densfile,      it_json->second.c_str());
    else
        mpi_error("density file name");
    
    // NX value
    it_json = map_json.find("NX");
    if(it_json != map_json.end())
        str_to_int(it_json->second,   mod_sp->nx, "NX (number of grid points in X direction)");
    else
        mpi_error("NX (number of grid points in X direction)");
    
    // NZ value
    it_json = map_json.find("NZ");
    if(it_json != map_json.end())
        str_to_int(it_json->second,   mod_sp->nz, "NZ (number of grid points in Z direction)");
    else
        mpi_error("NZ (number of grid points in Z direction)");

    // DX value
    it_json = map_json.find("DX");
    if(it_json != map_json.end())
        str_to_float(it_json->second, mod_sp->dx, "grid spacing in X direction");
    else
        mpi_error("grid spacing in X direction");
    
    // DZ value
    it_json = map_json.find("DZ");
    if(it_json != map_json.end())
        str_to_float(it_json->second, mod_sp->dz, "grid spacing in Z direction");
    else
        mpi_error("grid spacing in Z direction");

    // Record time
    it_json = map_json.find("Time");
    if(it_json != map_json.end())
        str_to_float(it_json->second, wave_sp->rec_len, "record time");
    else
        mpi_error("record time");

    // Sampling interval
    it_json = map_json.find("FD DT");
    if(it_json != map_json.end())
        str_to_float(it_json->second, wave_sp->fd_dt_sec, "FD sampling interval (DT)");
    else
        mpi_error("FD sampling interval (DT)");

    it_json = map_json.find("DATA DT");
    if(it_json != map_json.end())
        str_to_float(it_json->second, wave_sp->real_dt_sec, "Seismogram sampling interval (DT)");
    else
        mpi_error("Seismogram sampling interval (DT)");

    if(fmod(wave_sp->real_dt_sec, wave_sp->fd_dt_sec) != 0)
        mpi_error("DATA DT must be multiple of FD DT (DATA_DT>=FD_DT & DATA_DT%FD_DT=0)");

    // Dominant frequency
    it_json = map_json.find("Frequency");
    if(it_json != map_json.end())
        str_to_float(it_json->second, wave_sp->dom_freq, "dominant frequency");
    else
        mpi_error("dominant frequency");

    // Source-Receiver geometry
	it_json = map_json.find("Read source receiver flag");
    if(it_json != map_json.end())
        str_to_int(it_json->second,   job_sp->geomflag, "read source-receiver flag");
    else
        mpi_error("read source-receiver flag");

    if(job_sp->geomflag < 0 || job_sp->geomflag > 1)
        mpi_error("wrong flag value for read source-receiver flag");

    if(job_sp->geomflag == 0)
    {
        it_json = map_json.find("if choice=0, Geometry file");
        if(it_json != map_json.end())
            strncat(job_sp->geomfile,  it_json->second.c_str(), 223);
            // strcpy(job_sp->geomfile,  it_json->second.c_str());
        else
            mpi_error("source-receiver geometry file name");
    }
    else if(job_sp->geomflag == 1)
    {
        it_json = map_json.find("if choice=1, Shot file");
	    if(it_json != map_json.end())
            strncat(job_sp->shotfile,  it_json->second.c_str(), 223);
            // strcpy(job_sp->shotfile,  it_json->second.c_str());
        else
            mpi_error("shot file name");
    }

    // Data directory path
    it_json = map_json.find("Output Directory");
    if(it_json != map_json.end())
        strncat(job_sp->tmppath,       it_json->second.c_str(), 511);
        // strcpy(job_sp->tmppath,       it_json->second.c_str());
    else
        mpi_error("Output directory path");

    if( job_sp->tmppath[strlen(job_sp->tmppath)-1] != '/')
        strcat(job_sp->tmppath,       "/");

    // Free surface
	it_json = map_json.find("Free surface");    
    if(it_json != map_json.end())
        str_to_int(it_json->second,   job_sp->frsf, "free surface flag");
    else
        mpi_error("free surface flag");
    
    if(job_sp->frsf < 0 || job_sp->frsf > 1)
        mpi_error("wrong flag value for free surface choice");

    // Job type New/Restart
    it_json = map_json.find("Job type New or Restart");
    if(it_json != map_json.end())
        strncat(job_sp->jbtype,        it_json->second.c_str(), 7);
        // strcpy(job_sp->jbtype,        it_json->second.c_str());
    else
        mpi_error("job type New or Restart");
    
    if(strcmp(job_sp->jbtype, "New") != 0 && strcmp(job_sp->jbtype, "Restart") != 0)
    {
        cerr << "\n Error!!! wrong value is specified for job status";
        cerr << "\n Please correct job status in json file and submit job again...\n";
        MPI::COMM_WORLD.Abort(-6);
    }

    // FD order
    it_json = map_json.find("Order of FD Operator");
    if(it_json != map_json.end())
        str_to_int(it_json->second, job_sp->fdop, "order of FD Operator");
    else
        mpi_error("order of FD Operator");

    // Printing all the parameter extracted from json file
    cout << "\n\n Model :";
    cout << "\n   Velocity : "                              << mod_sp->velfile;
    cout << "\n   Density : "                               << mod_sp->densfile;

    cout << "\n\n Grid points :";
    cout << "\n   NX : "                                    << mod_sp->nx;
    cout << "\n   NZ : "                                    << mod_sp->nz;

    cout << "\n\n Grid size :";
    cout << "\n   DX : "                                    << mod_sp->dx;
    cout << "\n   DZ : "                                    << mod_sp->dz;

    cout << "\n\n Time Stepping :";
    cout << "\n   Time : "                                  << wave_sp->rec_len;
    cout << "\n   FD DT : "                                 << wave_sp->fd_dt_sec;
    cout << "\n   DATA DT : "                               << wave_sp->real_dt_sec;

    cout << "\n\n Source wavelet :";
    cout << "\n   Frequency : "                             << wave_sp->dom_freq;

    cout << "\n\n Source and Receiver :";
	cout << "\n   Read Source-Receiver 0=flase  1=true : "  << job_sp->geomflag;
    cout << "\n   Geometry file : "                         << job_sp->geomfile;
	cout << "\n   Shot file : "                             << job_sp->shotfile;

    cout << "\n\n Other job parameter: ";
    cout << "\n   Output directory path: "                  << job_sp->tmppath;
	cout << "\n   Free surface(0=false, 1=true) : "         << job_sp->frsf;
    cout << "\n   Job type New or Restart : "               << job_sp->jbtype;
    cout << "\n   Order of FD Operator : "                  << job_sp->fdop;   
    cout << "\n\n\n";

}// End of set_json_object function
