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


#include "segy_io_manager.h"
#include <unordered_set>
#include <vector>
#include <algorithm>

using namespace std;

SEGYIOManager::SEGYIOManager() {}

SEGYIOManager::~SEGYIOManager() {}

vector<uint>
SEGYIOManager::GetUniqueOccurences(std::string const &file_name, std::string const &key_name, uint min_threshold, uint max_threshold) {
    SUSegy *seg = new SUSegy();
    vector<SEGYelement> conditions;
    if (key_name == "CSR") {
        conditions.push_back(SEGYelement(&segy::fldr));
    } else if (key_name == "CDP") {
        conditions.push_back(SEGYelement(&segy::ensemble_number));
    } else {
        cout << "Invalid key for ID parsing : " << key_name << std::endl;
        cout << "Only CSR or CDP are supported as sort type" << std::endl;
        exit(EXIT_FAILURE);
    }
    vector<uint> unique_occurences(seg->GetUniqueOccurences(file_name, &conditions, min_threshold, max_threshold));
    delete seg;
    return unique_occurences;
}

// data ypr to diffrentaiate between trace and velocitie/density file
// this is important to correctly fill the corresponding filelds in the SIO
// object
void SEGYIOManager::ReadTracesDataFromFile(std::string const &file_name, std::string const &sort_type,
                                           SeIO *sio) {

    SUSegy *seg = new SUSegy();

    //    SeisIO * sei = new SeisIO();

    vector<SEGYelement> conditions;
    //    conditions.push_back(SEGYelement(&segy::fldr,50));
    seg->ReadHeadersAndTraces(file_name, &conditions, suselect::all);

    // cout << "segy values" << seg->bh.hns << endl;

    sio->DM.nt = seg->bh.hns;

    sio->DM.dt = seg->bh.hdt / (float) 1000000;

    int ix = 0;

    //cout << "SIO dt = " << sio->DM.dt << endl;

    for (auto it = seg->traces.begin(); it != seg->traces.end(); it++, ix++) {

        GeneralTraces GT;
        sio->Atraces.push_back(GT);
        sio->Atraces[ix].TraceMetaData.shot_id = it->fldr;
        sio->Atraces[ix].TraceMetaData.ensembleType =
                CDP; // for now it shoulld  be modified to translate the numbers of
        // ensemble number
        sio->Atraces[ix].TraceMetaData.source_location_x = it->sx;
        sio->Atraces[ix].TraceMetaData.source_location_y = it->sy;
        sio->Atraces[ix].TraceMetaData.source_location_z = 0;
        sio->Atraces[ix].TraceMetaData.receiver_location_x = it->gx;
        sio->Atraces[ix].TraceMetaData.receiver_location_y = it->gy;
        sio->Atraces[ix].TraceMetaData.receiver_location_z = 0;
        sio->Atraces[ix].TraceMetaData.trace_id_within_line = it->tracl;
        sio->Atraces[ix].TraceMetaData.trace_id_within_file = it->tracr;
        sio->Atraces[ix].TraceMetaData.trace_id_for_shot = it->tracf;
        sio->Atraces[ix].TraceMetaData.trace_id_for_ensemble = it->cdpt;
        sio->Atraces[ix].TraceMetaData.trace_identification_code = it->trid;
        sio->Atraces[ix].TraceMetaData.scalar = it->scalco + 1 * (it->scalco == 0);

        //  cout << "SIO scalr " << sio->Atraces[ix].shot << endl;

        memcpy(sio->Atraces[ix].TraceData, it->data, 32767 * sizeof(float));
    }

    delete (seg);
}

void SEGYIOManager::ReadVelocityDataFromFile(std::string const &file_name, std::string const &sort_type, SeIO *sio) 
{

    SUSegy *seg = new SUSegy();

    vector<SEGYelement> conditions;
    seg->ReadHeadersAndTraces(file_name, &conditions, suselect::all);

    // cout << "segy values" << seg->bh.hns << endl;

    unordered_set<float> x_dim;
    unordered_set<float> y_dim;

    sio->DM.nz = seg->bh.hns;
    sio->DM.dz = seg->bh.hdt / (float) 1000.0;

    int ix = 0;

    for (auto it = seg->traces.begin(); it != seg->traces.end(); it++, ix++) {

        GeneralTraces GT;
        sio->Velocity.push_back(GT);
        sio->Velocity[ix].TraceMetaData.shot_id = it->fldr;
        sio->Velocity[ix].TraceMetaData.ensembleType =
                CDP; // for now it shoulld  be modified to translate the numbers of
        // ensemble number
        sio->Velocity[ix].TraceMetaData.source_location_x = it->sx;
        sio->Velocity[ix].TraceMetaData.source_location_y = it->sy;
        sio->Velocity[ix].TraceMetaData.source_location_z = 0;
        sio->Velocity[ix].TraceMetaData.receiver_location_x = it->gx;
        sio->Velocity[ix].TraceMetaData.receiver_location_y = it->gy;
        sio->Velocity[ix].TraceMetaData.receiver_location_z = 0;
        sio->Velocity[ix].TraceMetaData.trace_id_within_line = it->tracl;
        sio->Velocity[ix].TraceMetaData.trace_id_within_file = it->tracr;
        sio->Velocity[ix].TraceMetaData.trace_id_for_shot = it->tracf;
        sio->Velocity[ix].TraceMetaData.trace_id_for_ensemble = it->cdpt;
        sio->Velocity[ix].TraceMetaData.trace_identification_code = it->trid;
        sio->Velocity[ix].TraceMetaData.scalar = it->scalco + 1 * (it->scalco == 0);

        x_dim.insert(it->sx);
        y_dim.insert(it->sy);

        memcpy(sio->Velocity[ix].TraceData, it->data, 32767 * sizeof(float));
    }

    sio->DM.nx = x_dim.size();
    sio->DM.ny = y_dim.size();

    std::vector<float> x_positions(x_dim.begin(), x_dim.end());

    std::sort(x_positions.begin(), x_positions.end());

    float dBscalar = seg->traces.at(0).scalco + 1 * (seg->traces.at(0).scalco == 0);

    sio->DM.dx = (x_positions[1] * dBtoscale(dBscalar)) -
                 (x_positions[0] * dBtoscale(dBscalar));

    if(sio->DM.ny != 1){
        std::vector<float> y_positions(y_dim.begin(), y_dim.end());
        std::sort(y_positions.begin(), y_positions.end());

        sio->DM.dy = (y_positions[1] * dBtoscale(dBscalar)) -
                     (y_positions[0] * dBtoscale(dBscalar));
    } else {
    	sio->DM.dy = 1.0;
    }

    delete (seg);
}

void SEGYIOManager::ReadDensityDataFromFile(std::string const &file_name, std::string const &sort_type,
                                            SeIO *sio) {

    SUSegy *seg = new SUSegy();

    vector<SEGYelement> conditions;
    seg->ReadHeadersAndTraces(file_name, &conditions, suselect::all);

    int ix = 0;
    for (auto it = seg->traces.begin(); it != seg->traces.end(); it++, ix++) {

        GeneralTraces GT;
        sio->Density.push_back(GT);
        sio->Density[ix].TraceMetaData.shot_id = it->fldr;
        sio->Density[ix].TraceMetaData.ensembleType =
                CDP; // for now it shoulld  be modified to translate the numbers of
        // ensemble number
        sio->Density[ix].TraceMetaData.source_location_x = it->sx;
        sio->Density[ix].TraceMetaData.source_location_y = it->sy;
        sio->Density[ix].TraceMetaData.source_location_z = 0;
        sio->Density[ix].TraceMetaData.receiver_location_x = it->gx;
        sio->Density[ix].TraceMetaData.receiver_location_y = it->gy;
        sio->Density[ix].TraceMetaData.receiver_location_z = 0;
        sio->Density[ix].TraceMetaData.trace_id_within_line = it->tracl;
        sio->Density[ix].TraceMetaData.trace_id_within_file = it->tracr;
        sio->Density[ix].TraceMetaData.trace_id_for_shot = it->tracf;
        sio->Density[ix].TraceMetaData.trace_id_for_ensemble = it->cdpt;
        sio->Density[ix].TraceMetaData.trace_identification_code = it->trid;
        sio->Density[ix].TraceMetaData.scalar = it->scalco + 1 * (it->scalco == 0);

        //        cout << "SIO scalr " << sio->Atraces[ix].TraceMetaData.scalar <<
        //        endl;

        memcpy(sio->Density[ix].TraceData, it->data, 32767 * sizeof(float));
    }

    delete (seg);
}

//  passign the condtions , how would like to pass the parameters ???
void SEGYIOManager::ReadSelectiveDataFromFile(std::string const &file_name, std::string const &sort_type, SeIO *sio, int cond) {


    SUSegy *seg = new SUSegy();

//    SeisIO * sei = new SeisIO();

    vector<SEGYelement> conditions;
    if (sort_type == "CSR") {
        conditions.push_back(SEGYelement(&segy::fldr, cond));
    } else if (sort_type == "CDP") {
        conditions.push_back(SEGYelement(&segy::ensemble_number, cond));
    } else {
        cout << "Invalid key for ID parsing : " << sort_type << std::endl;
        cout << "Only CSR or CDP are supported as sort type" << std::endl;
        exit(EXIT_FAILURE);
    }
    seg->ReadHeadersAndTraces(file_name, &conditions, suselect::ifequal);


    // cout << "segy values" << seg->bh.hns << endl;

    unordered_set<float> x_dim;
    unordered_set<float> y_dim;

    sio->DM.nt = seg->bh.hns;

    sio->DM.dt = seg->bh.hdt / (float) 1000000;

    int ix = 0;

    for (auto it = seg->traces.begin(); it != seg->traces.end(); it++, ix++) {

        GeneralTraces GT;
        sio->Atraces.push_back(GT);
        sio->Atraces[ix].TraceMetaData.shot_id = it->fldr;
        sio->Atraces[ix].TraceMetaData.ensembleType = CDP; // for now it shoulld  be modified to translate the numbers of ensemble number
        sio->Atraces[ix].TraceMetaData.source_location_x = it->sx;
        sio->Atraces[ix].TraceMetaData.source_location_y = it->sy;
        sio->Atraces[ix].TraceMetaData.source_location_z = 0;
        sio->Atraces[ix].TraceMetaData.receiver_location_x = it->gx;
        sio->Atraces[ix].TraceMetaData.receiver_location_y = it->gy;
        sio->Atraces[ix].TraceMetaData.receiver_location_z = 0;
        sio->Atraces[ix].TraceMetaData.trace_id_within_line = it->tracl;
        sio->Atraces[ix].TraceMetaData.trace_id_within_file = it->tracr;
        sio->Atraces[ix].TraceMetaData.trace_id_for_shot = it->tracf;
        sio->Atraces[ix].TraceMetaData.trace_id_for_ensemble = it->cdpt;
        sio->Atraces[ix].TraceMetaData.trace_identification_code = it->trid;
        sio->Atraces[ix].TraceMetaData.scalar = it->scalco + 1 * (it->scalco == 0);

        // cout << "value of rec location x at " << ix <<"  " << sio->Atraces[ix].TraceMetaData.receiver_location_x  << endl;
        x_dim.insert(it->gx);
        y_dim.insert(it->gy);

        memcpy(sio->Atraces[ix].TraceData, it->data, 32767 * sizeof(float));

    }

    // cout << "value of rec location x after loop " << (int )sio->Atraces[1000].TraceMetaData.receiver_location_x  << endl;
    sio->DM.nx = x_dim.size();
    sio->DM.ny = y_dim.size();

    std::vector<float> x_positions(x_dim.begin(), x_dim.end());

    std::sort(x_positions.begin(), x_positions.end());

    float dBscalar = seg->traces.at(0).scalco + 1 * (seg->traces.at(0).scalco == 0);

    sio->DM.dx = (x_positions[1] * dBtoscale(dBscalar)) -
                 (x_positions[0] * dBtoscale(dBscalar));

    if(sio->DM.ny != 1){
        std::vector<float> y_positions(y_dim.begin(), y_dim.end());
        std::sort(y_positions.begin(), y_positions.end());

        sio->DM.dy = (y_positions[1] * dBtoscale(dBscalar)) -
                     (y_positions[0] * dBtoscale(dBscalar));
    } else {
    	sio->DM.dy = 1.0;
    }

    delete (seg);
}

void SEGYIOManager::WriteTracesDataToFile(std::string const &file_name, std::string const &sort_type, SeIO *sio) 
{
    SUSegy *seg = new SUSegy();

    seg->bh.hns = sio->DM.nt;

    seg->bh.ntrpr = sio->DM.nx * sio->DM.ny;

    seg->bh.hdt = sio->DM.dt * (float) 1000000;


    int ix = 0;

    for (auto it = sio->Atraces.begin(); it != sio->Atraces.end(); it++, ix++) {
        segy trace;
        memset(&trace, 0, sizeof(segy));
        seg->traces.push_back(trace);
        seg->traces[ix].ns = sio->DM.nt;
        seg->traces[ix].fldr = it->TraceMetaData.shot_id;
        seg->traces[ix].sx = it->TraceMetaData.source_location_x;
        seg->traces[ix].sy = it->TraceMetaData.source_location_y;
        seg->traces[ix].dt = sio->DM.dt * (float) 1000000;
        //  sio->Atraces[ix].TraceMetaData.ensembleType = CDP; // for now it shoulld  be modified to translate the numbers of ensemble number

        seg->traces[ix].gx = it->TraceMetaData.receiver_location_x;
        seg->traces[ix].gy = it->TraceMetaData.receiver_location_y;
        seg->traces[ix].tracl = it->TraceMetaData.trace_id_within_line;
        seg->traces[ix].tracr = it->TraceMetaData.trace_id_within_file;

        seg->traces[ix].tracf = it->TraceMetaData.trace_id_for_shot;
        seg->traces[ix].cdpt = it->TraceMetaData.trace_id_for_ensemble;
        seg->traces[ix].trid = it->TraceMetaData.trace_identification_code;
        seg->traces[ix].scalco = it->TraceMetaData.scalar;

        //        cout << "SIO scalr " << sio->Atraces[ix].TraceMetaData.scalar <<
        //        endl;

        memcpy(seg->traces[ix].data, it->TraceData, 32767 * sizeof(float));
    }

    // cout << "value before writing the seg
    // traces"<<seg->traces.at(1000).data[1000]<< endl ;

    seg->WriteHeadersAndTraces(file_name);
    delete (seg);
}

void SEGYIOManager::WriteVelocityDataToFile(std::string const &file_name, std::string const &sort_type, SeIO *sio) 
{

    //  cout << " you are in writing velocity" << endl;

    SUSegy *seg = new SUSegy();

    seg->bh.ntrpr = sio->DM.nx * sio->DM.ny;
    seg->bh.hns = sio->DM.nz;

    seg->bh.hdt = sio->DM.dz * (float) 1000.0;

    int ix = 0;

    for (auto it = sio->Velocity.begin(); it != sio->Velocity.end(); it++, ix++) {

        segy trace;
        memset(&trace, 0, sizeof(segy));
        seg->traces.push_back(trace);
        seg->traces[ix].ns = sio->DM.nz;
        seg->traces[ix].sx = it->TraceMetaData.source_location_x;
        seg->traces[ix].fldr = it->TraceMetaData.shot_id;
        seg->traces[ix].sy = it->TraceMetaData.source_location_y;
        //  sio->Atraces[ix].TraceMetaData.ensembleType = CDP; // for now it shoulld
        //  be modified to translate the numbers of ensemble number

        seg->traces[ix].gx = it->TraceMetaData.receiver_location_x;
        seg->traces[ix].gy = it->TraceMetaData.receiver_location_y;
        seg->traces[ix].tracl = it->TraceMetaData.trace_id_within_line;
        seg->traces[ix].tracr = it->TraceMetaData.trace_id_within_file;

        seg->traces[ix].tracf = it->TraceMetaData.trace_id_for_shot;
        seg->traces[ix].cdpt = it->TraceMetaData.trace_id_for_ensemble;
        seg->traces[ix].trid = it->TraceMetaData.trace_identification_code;
        seg->traces[ix].scalco = it->TraceMetaData.scalar;

        //        cout << "SIO scalr " << sio->Atraces[ix].TraceMetaData.scalar <<
        //        endl;

        memcpy(seg->traces[ix].data, it->TraceData, 32767 * sizeof(float));
    }

    //    cout << "it value " << ix << endl;

    // cout << "value before writing the the velocity
    // "<<seg->traces.at(1000).data[1000]<< endl ;

    seg->WriteHeadersAndTraces(file_name);
    delete (seg);
}

void SEGYIOManager::WriteDensityDataToFile(std::string const &file_name, std::string const &sort_type, SeIO *sio) 
{

    SUSegy *seg = new SUSegy();

    seg->bh.ntrpr = sio->DM.nx * sio->DM.ny;
    seg->bh.hns = sio->DM.nz;

    seg->bh.hdt = sio->DM.dz * (float) 1000.0;

    int ix = 0;

    for (auto it = sio->Density.begin(); it != sio->Density.end(); it++, ix++) {

        segy trace;
        memset(&trace, 0, sizeof(segy));
        seg->traces.push_back(trace);
        seg->traces[ix].ns = sio->DM.nz;
        seg->traces[ix].sx = it->TraceMetaData.source_location_x;
        seg->traces[ix].fldr = it->TraceMetaData.shot_id;
        seg->traces[ix].sy = it->TraceMetaData.source_location_y;

        seg->traces[ix].gx = it->TraceMetaData.receiver_location_x;
        seg->traces[ix].gy = it->TraceMetaData.receiver_location_y;
        seg->traces[ix].tracl = it->TraceMetaData.trace_id_within_line;
        seg->traces[ix].tracr = it->TraceMetaData.trace_id_within_file;

        seg->traces[ix].tracf = it->TraceMetaData.trace_id_for_shot;
        seg->traces[ix].cdpt = it->TraceMetaData.trace_id_for_ensemble;
        seg->traces[ix].trid = it->TraceMetaData.trace_identification_code;
        seg->traces[ix].scalco = it->TraceMetaData.scalar;

        //        cout << "SIO scalr " << sio->Atraces[ix].TraceMetaData.scalar <<
        //        endl;

        memcpy(seg->traces[ix].data, it->TraceData, 32767 * sizeof(float));
    }

    //  cout << "value before writing the density
    //  "<<seg->traces.at(1000).data[1000]<< endl ;

    seg->WriteHeadersAndTraces(file_name);
    delete (seg);
}
