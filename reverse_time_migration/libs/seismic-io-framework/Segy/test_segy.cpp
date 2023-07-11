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


#include "../IO/io_manager.h"
#include "segy_io_manager.h"
#include "susegy.h"

using namespace std;

int main() {

    // SUSegy* seg = new SUSegy("../shots0001_0200.segy");

    IOManager *IO = new SEGYIOManager();
    SeIO *sio = new SeIO();
    SUSegy *seg = new SUSegy();
    SeIO *sio1 = new SeIO();

    // cout << "you are here " << endl;
    vector<SEGYelement> conditions;
    conditions.push_back(SEGYelement(&segy::tracr, 50));
    // seg->ReadHeadersAndTraces("../shots0001_0200.segy",&conditions,suselect::ifequal);
    // seg->ReadHeadersAndTraces("../vel_z6.25m_x12.5m_exact.segy",&conditions,suselect::all);

    //  IO->ReadTracesDataFromFile("../shots0001_0200.segy","CSR",sio);
    IO->ReadVelocityDataFromFile("../../vel_z6.25m_x12.5m_exact.segy", "CSR",
                                 sio);

    IO->ReadSelectiveDataFromFile("../../shots0001_0200.segy", "CSR", sio, 50);

    cout << sio->DM.nz << " " << sio->DM.nx << " " << sio->DM.nt << endl;
    cout << sio->Atraces.at(1000).TraceData[1000] << endl;
    cout << sio->Atraces.size() << endl;
    cout << "finishing reading " << endl;

    IO->WriteTracesDataToFile("../trial_write.segy", "CSR", sio);
    IO->WriteVelocityDataToFile("../trial_vel_write.segy", "CSR", sio);

    IO->ReadSelectiveDataFromFile("../trial_write.segy", "CSR", sio1, 50);

    cout << sio1->DM.nz << " " << sio1->DM.nx << " " << sio1->DM.nt << endl;
    cout << "value at reareading " << sio1->Atraces.at(1000).TraceData[1000]
         << endl;
    cout << sio1->Atraces.size() << endl;

    IO->ReadVelocityDataFromFile("../trial_vel_write.segy", "CSR", sio1);

    cout << "value at reareading " << sio1->Velocity.at(1000).TraceData[1000]
         << endl;
    cout << "size of velocity " << sio1->Velocity.size() << endl;
    //  cout << "finishing reading " << endl;
}
