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

#pragma once

#include <dbus/dbus.h>

using namespace std;

class DBusInt
{
  public:
    DBusInt()
    {
        dbus_error_init(&err);
        conn = dbus_bus_get(DBUS_BUS_SESSION, &err);
        if (!conn) {
            minelog << "DBus error " << err.name << ": " << err.message;
        }
        dbus_bus_request_name(conn, "eth.miner", DBUS_NAME_FLAG_REPLACE_EXISTING, &err);
        if (dbus_error_is_set(&err)) {
            minelog << "DBus error " << err.name << ": " << err.message;
            dbus_connection_close(conn);
        }
        minelog << "DBus initialized!";
    }

    void send(const char *hash)
    {
        DBusMessage *msg;
        msg = dbus_message_new_signal("/eth/miner/hash", "eth.miner.monitor", "Hash");
        if (msg == nullptr) {
            minelog << "Message is null!";
        }
        dbus_message_append_args(msg, DBUS_TYPE_STRING, &hash, DBUS_TYPE_INVALID);
        if (!dbus_connection_send(conn, msg, nullptr))
            cerr << "Error sending message!";
        dbus_message_unref(msg);
    }

  private:
    DBusError       err;
    DBusConnection *conn;
};
