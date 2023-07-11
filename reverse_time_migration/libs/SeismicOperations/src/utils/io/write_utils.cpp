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
// Created by amr on 07/06/2020.
//

#include <operations/utils/io/write_utils.h>

#include <IO/io_manager.h>
#include <Segy/segy_io_manager.h>
#include <seismic-io-framework/datatypes.h>
#include <string>

using namespace std;
using namespace operations::utils::io;


void operations::utils::io::write_adcig_segy(uint nx, uint ny, uint nz, uint nt,
                                             uint n_angles,
                                             float dx, float dy, float dz, float dt,
                                             const float *data,
                                             const string &file_name, bool is_traces) {
    auto *io = new SEGYIOManager();
    auto *sio = new SeIO();

    sio->DM.nx = nx;
    sio->DM.ny = ny;
    sio->DM.nz = nz;
    sio->DM.nt = nt;

    sio->DM.dx = dx;
    sio->DM.dy = dy;
    sio->DM.dz = dz;
    sio->DM.dt = dt;

    for (int theta = 0; theta < n_angles; theta++) {
        for (int y = 0; y < ny; y++) {
            for (int x = 0; x < nx; x++) {
                GeneralTraces GT;
                sio->Velocity.push_back(GT);
                int id = theta * nx * ny + y * nx + x;

                sio->Velocity[id].TraceMetaData.shot_id = theta;
                sio->Velocity[id].TraceMetaData.receiver_location_x = x * dx;

                /// For now it should  be modified to translate the numbers of
                /// ensemble number
                sio->Velocity[id].TraceMetaData.ensembleType = CSP;

                sio->Velocity[id].TraceMetaData.source_location_x = x * dx;
                sio->Velocity[id].TraceMetaData.source_location_y = y * dy;
                sio->Velocity[id].TraceMetaData.source_location_z = 0;

                sio->Velocity[id].TraceMetaData.receiver_location_y = y * dy;
                sio->Velocity[id].TraceMetaData.receiver_location_y = y * dy;
                sio->Velocity[id].TraceMetaData.receiver_location_z = 0;

                sio->Velocity[id].TraceMetaData.trace_id_within_line = id;
                sio->Velocity[id].TraceMetaData.trace_id_within_file = id;
                sio->Velocity[id].TraceMetaData.trace_id_for_shot = 1;
                sio->Velocity[id].TraceMetaData.trace_id_for_ensemble = id;
                sio->Velocity[id].TraceMetaData.trace_identification_code = id;

                sio->Velocity[id].TraceMetaData.scalar = 10;

                for (int z = 0; z < nz; z++) {
                    sio->Velocity[id].TraceData[z] = data[theta * nx * nz * ny + y * nx * nz + z * nx + x];
                }
            }
        }
    }
    io->WriteVelocityDataToFile(file_name, "CSR", sio);
    delete sio;
    delete io;
}

void operations::utils::io::write_segy(uint nx, uint ny, uint nz, uint nt,
                                       float dx, float dy, float dz, float dt,
                                       const float *data, const string &file_name, bool is_traces) {
    auto *IO = new SEGYIOManager();
    SeIO *sio = new SeIO();

    sio->DM.nt = nt;
    sio->DM.dt = dt;
    sio->DM.nx = nx;
    sio->DM.nz = nz;
    sio->DM.ny = ny;
    sio->DM.dx = dx;
    sio->DM.dz = dz;
    sio->DM.dy = dy;

    if (!is_traces) {
        for (int y = 0; y < ny; y++) {
            for (int x = 0; x < nx; x++) {
                GeneralTraces GT;
                sio->Velocity.push_back(GT);
                int id = y * nx + x;

                sio->Velocity[id].TraceMetaData.shot_id = id;
                /// For now it should be modified to translate the numbers of
                /// ensemble number
                sio->Velocity[id].TraceMetaData.ensembleType = CSP;
                sio->Velocity[id].TraceMetaData.source_location_x = x * dx;
                sio->Velocity[id].TraceMetaData.source_location_y = y * dy;
                sio->Velocity[id].TraceMetaData.source_location_z = 0;
                sio->Velocity[id].TraceMetaData.receiver_location_x = x * dx;
                sio->Velocity[id].TraceMetaData.receiver_location_y = y * dy;
                sio->Velocity[id].TraceMetaData.receiver_location_z = 0;
                sio->Velocity[id].TraceMetaData.trace_id_within_line = id;
                sio->Velocity[id].TraceMetaData.trace_id_within_file = id;
                sio->Velocity[id].TraceMetaData.trace_id_for_shot = 1;
                sio->Velocity[id].TraceMetaData.trace_id_for_ensemble = id;
                sio->Velocity[id].TraceMetaData.trace_identification_code = id;
                sio->Velocity[id].TraceMetaData.scalar = 10;

                for (int z = 0; z < nz; z++) {
                    sio->Velocity[id].TraceData[z] = data[y * nx * nz + z * nx + x];
                }
            }
        }
        IO->WriteVelocityDataToFile(file_name, "CSR", sio);
    } else {
        for (int y = 0; y < ny; y++) {
            for (int x = 0; x < nx; x++) {
                GeneralTraces GT;
                sio->Atraces.push_back(GT);
                int id = y * nx + x;

                sio->Atraces[id].TraceMetaData.shot_id = id;
                sio->Atraces[id].TraceMetaData.receiver_location_x = x * dx;
                /// For now it should  be modified to translate the numbers of
                /// ensemble number
                sio->Atraces[id].TraceMetaData.ensembleType = CSP;

                sio->Atraces[id].TraceMetaData.source_location_x = x * dx;
                sio->Atraces[id].TraceMetaData.source_location_y = y * dy;
                sio->Atraces[id].TraceMetaData.source_location_z = 0;

                sio->Atraces[id].TraceMetaData.receiver_location_y = y * dy;
                sio->Atraces[id].TraceMetaData.receiver_location_y = y * dy;
                sio->Atraces[id].TraceMetaData.receiver_location_z = 0;

                sio->Atraces[id].TraceMetaData.trace_id_within_line = id;
                sio->Atraces[id].TraceMetaData.trace_id_within_file = id;
                sio->Atraces[id].TraceMetaData.trace_id_for_shot = 1;
                sio->Atraces[id].TraceMetaData.trace_id_for_ensemble = id;
                sio->Atraces[id].TraceMetaData.trace_identification_code = id;

                sio->Atraces[id].TraceMetaData.scalar = 10;

                for (int z = 0; z < nz; z++) {
                    sio->Atraces[id].TraceData[z] = data[y * nx * nz + z * nx + x];
                }
            }
        }
        IO->WriteTracesDataToFile(file_name, "CSR", sio);
    }
    delete sio;
    delete IO;
}

void operations::utils::io::write_su(const float *temp, uint nx, uint nz,
                                     const char *file_name, bool write_little_endian) {
    std::ofstream stream(file_name, std::ios::out | std::ios::binary);
    if (!stream.is_open()) {
        exit(EXIT_FAILURE);
    }
    char dummy_header[240];
    memset(dummy_header, 0, 240);
    unsigned short tid = 1;
    unsigned short ns = nz;
    bool little_endian = false;
    // little endian if true
    if (*(char *) &tid == 1) {
        little_endian = true;
    }
    bool swap_bytes = true;
    if ((little_endian && write_little_endian) || (!little_endian && !write_little_endian)) {
        swap_bytes = false;
    }
    for (int i = 0; i < nx; i++) {
        stream.write(dummy_header, 14);
        if (swap_bytes) {
            stream.write(((char *) &tid) + 1, 1);
            stream.write((char *) (&tid), 1);
        } else {
            stream.write((char *) &tid, 2);
        }
        stream.write(dummy_header, 98);
        if (swap_bytes) {
            stream.write(((char *) &ns) + 1, 1);
            stream.write((char *) (&ns + 0), 1);
        } else {
            stream.write((char *) &ns, 2);
        }
        stream.write(dummy_header, 124);
        for (int j = 0; j < nz; j++) {
            float val = temp[i + j * nx];
            if (swap_bytes) {
                stream.write(((char *) &val) + 3, 1);
                stream.write(((char *) &val) + 2, 1);
                stream.write(((char *) &val) + 1, 1);
                stream.write(((char *) &val) + 0, 1);
            } else {
                stream.write((char *) &val, 4);
            }
        }
    }
    stream.close();
}

void operations::utils::io::write_binary(float *temp, uint nx, uint nz, const char *file_name) {
    std::ofstream stream(file_name, std::ios::out | std::ios::binary);
    if (!stream.is_open()) {
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < nz; j++) {
            stream.write(reinterpret_cast<const char *>(temp + j * nx + i), sizeof(float));
        }
    }
    stream.close();
}
