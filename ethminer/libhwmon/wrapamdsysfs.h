/* 
 * Copyright (C) <2023> Intel Corporation
 * 
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License, as published
 * by the Free Software Foundation; either version 2 of the License,
 * or (at your option) any later version.
 *  
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *  
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, see <http://www.gnu.org/licenses/>.
 *  
 * 
 * SPDX-License-Identifier: GPL-2.0-or-later
 * 
 */ 

/*
 * Wrapper for AMD SysFS on linux, using adapted code from amdcovc by matszpk
 *
 * By Philipp Andreas - github@smurfy.de
   Reworked and simplified by Andrea Lanfranchi (github @AndreaLanfranchi)
 */

#pragma once

typedef struct
{
    int           sysfs_gpucount;
    unsigned int *sysfs_device_id;
    unsigned int *sysfs_hwmon_id;
    unsigned int *sysfs_pci_domain_id;
    unsigned int *sysfs_pci_bus_id;
    unsigned int *sysfs_pci_device_id;
} wrap_amdsysfs_handle;

struct pciInfo
{

    int DeviceId ;
    int HwMonId  ;
    int PciDomain;
    int PciBus   ;
    int PciDevice;

    pciInfo() 
        : DeviceId  (-1)
        , HwMonId   (-1)
        , PciDomain (-1)
        , PciBus    (-1)
        , PciDevice (-1)
    {}
   
};

wrap_amdsysfs_handle *wrap_amdsysfs_create();
int                   wrap_amdsysfs_destroy(wrap_amdsysfs_handle *sysfsh);

int wrap_amdsysfs_get_gpucount(wrap_amdsysfs_handle *sysfsh, int *gpucount);

int wrap_amdsysfs_get_tempC(wrap_amdsysfs_handle *sysfsh, int index, unsigned int *tempC);

int wrap_amdsysfs_get_fanpcnt(wrap_amdsysfs_handle *sysfsh, int index, unsigned int *fanpcnt);

int wrap_amdsysfs_get_power_usage(
    wrap_amdsysfs_handle *sysfsh,
    int                   index,
    unsigned int         *milliwatts);
