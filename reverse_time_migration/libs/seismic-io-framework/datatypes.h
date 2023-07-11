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
// Created by mennatallah on 12/17/19.
//

#ifndef SEISMIC_IO_DATATYPES_H
#define SEISMIC_IO_DATATYPES_H

// the needed libraries
#include <cstdlib>
#include <iostream>
#include <vector>

#define SU_NFLTS 32767 /**< Arbitrary limit on data array size	*/

enum Ensemble_type {
    unknown,
    noSorting,
    CMP, // CommonMidPoint
    CDP, // CommonDepthPoint
    SingleFoldContinuousProfile,
    HorizontallyStacked,
    CSP, // CommonSourcePoint,
    CRP, // CommonReceiverPoint,
    COP, // CommonOffsetPoint,
    CCP  // CommonConversionPoint
};

class GeneralTraces {
public:
    struct TMD {
        /*!the fldr field in trace header of traces file is the Original field
         * record number and considered as shot_id
         */
        int shot_id;

        /*! the Ensemble number (i.e. CDP- CMP- CRP-...etc) field in trace header of
         * traces represent the type of gathering of traces
         */

        Ensemble_type ensembleType;
        //            /*!the ntrpr field in binary header represents
        //           * the number of receivers in traces file and the nx in velocity
        //           file
        //           */
        //            unsigned int number_of_receivers;

        /*!the sx:Source coordinate - X and sy:Source coordinate - Y in trace header
         * of traces represent  the source location in x and y in
         */
        int source_location_x;
        int source_location_y;
        int source_location_z;

        /*!the gx:Group coordinate - X  and gy:Group coordinate - Y in trace header
         *of traces represent  the receiver location in x and y in in the common
         *shot point gathering we gather for each shot all receivers
         */
        int receiver_location_x;
        int receiver_location_y;
        int receiver_location_z;

        /*! in the traces header of traces file the (tracl) Trace sequence number
         * within line and (tracr) Trace sequence number within SEG Y file represent
         * the trace_id_within_file or line and are file or line  dependent
         */
        int trace_id_within_line;
        int trace_id_within_file;

        /*! in the traces header of traces file the (tracf) Trace number within
         * original field record and the (cdpt) trace number within the ensemble
         * represent the trace_id_within_shot or ensemble and are receiver or
         * ensemble dependent
         */
        int trace_id_for_shot;
        int trace_id_for_ensemble;

        /*!in the traces header of traces the (trid)trace identification code
         *- 1 = seismic data
                    - 2 = dead
                    - 3 = dummy
                    - 4 = time break
                    - 5 = uphole
                    - 6 = sweep
                    - 7 = timing
                    - 8 = water break
                    - 9---, N = optional use (N = 32,767)
         */
        unsigned int trace_identification_code;

        /*!the scalo field in trace header of traces or velocity
         * scalar to be applied to sx,sy,gx,gy
         */
        int scalar;

        TMD() 
            : shot_id(0)
            , ensembleType(unknown)
            , source_location_x(0)
            , source_location_y(0)
            , source_location_z(0)
            , receiver_location_x(0)
            , receiver_location_y(0)
            , receiver_location_z(0)
            , trace_id_within_line(0)
            , trace_id_within_file(0)
            , trace_id_for_shot(0)
            , trace_id_for_ensemble(0)
            , trace_identification_code(0)
            , scalar(0){}



    }; ////TraceMetaData;

    TMD TraceMetaData;

    float TraceData[SU_NFLTS];

    //! A constructor is called each time an object is created
    GeneralTraces() { std::fill(std::begin(TraceData), std::end(TraceData), 0.0f);  }

    // is this the right place fo the function or it shul be at the abstract layer
    // ??

    //! A deconstructor is called automatically when an object is deleted or is no
    //! longer used
    ~GeneralTraces() {}
};


class SeIO {
public:
    typedef struct {
        // int shot_location;
    } GridInformation;

    struct DomainMetaData {

        /*!the ntrpr field in binary header represents
         * the number of receivers in traces file and the nx in velocity file
         */

        unsigned int nr; // number of recievers
        unsigned int nx;

        // field that represents ny in 3D ??
        unsigned int ny;

        // maximum number of receivers, in a general case it shouldn't be the same for
        // all traces
        unsigned int MAX_receiver;

        /*!the hdt field in binary header represents sample interval in micro secs for
         * this reel represents the dt in micro sec or the dz in mm also in the trace
         * header of traces the dt sample interval in micro-seconds represent dt and
         * in velocity represents the dz but with error so we shouldn't use trace
         * header of velocity to get dz
         */
        float dt;
        float dz;

        /*! the hns field in binary header represents number of samples per trace for
         * this reel is nz in velocity file and nt in traces file also the ns (number
         * of samples in this trace) field in the trace header of traces represent the
         * nt and in trace header of velocity represent the nz
         */
        unsigned int nz;
        unsigned int nt;

        /*!for dx it is given from the trace header file for velocity as shown
         * tracesHeaders[0].sx * dBtoscale(tracesHeaders[0].scalco)
         * -tracesHeaders[1].sx * dBtoscale(tracesHeaders[1].scalco) from sx and scalo
         * in trace header
         */
        float dx;
        float dy;

        DomainMetaData() 
            : nr(0)
            , nx(0)
            , ny(0)
            , MAX_receiver(0)
            , dt(0.0f)
            , dz(0.0f)
            , nz(0)
            , nt(0)
            , dx(0.0f)
            , dy(0.0f){}
              

    }; ///DomainMetaData;


    DomainMetaData DM;

    std::vector<GeneralTraces> Atraces;

    std::vector<GeneralTraces> Velocity; // currently to fill the velocity field , may
    // later ned another one for the density

    std::vector<GeneralTraces> Density;

    //! A constructor is called each time an object is created
    SeIO() {}

    /*  // same question about the right place ??
      void ReadDomainMetaData()
      {

      }
    */
    //! A destructor is called automatically when an object is deleted or is no
    //! longer used
    ~SeIO() {}
};

// can this be in another floder specific for segy ?? with all its functi

#endif // SEISMIC_IO_DATATYPES_H
