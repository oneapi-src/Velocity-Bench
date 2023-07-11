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


/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 * File:   segyelement.h
 * Author: krm
 *
 * Created on July 8, 2018, 2:42 PM
 */

#ifndef SEGYELEMENT_H
#define SEGYELEMENT_H

#include "suheaders.h"
#include <typeinfo>

class SEGelement {
protected:
    char classType;

public:
    virtual void foo() {}
};

class SEGYelement : public SEGelement {
private:
    char type;

public:
    Value segy::*member;
    Value value;
    bool valueExist;

    SEGYelement() {}

    template<typename T>
    SEGYelement(T segy::*m, T v) {
        Set(m, v);
        classType = 's';
        valueExist = true;
    }

    template<typename T>
    SEGYelement(T segy::*m) {
        Set(m);
        classType = 's';
        valueExist = false;
    }

    template<typename T>
    void Set(T segy::*m) {
        if (std::is_same<T, int>::value) {
            type = 'i';
            member = reinterpret_cast<Value segy::*>(m);
        } else if (std::is_same<T, short>::value) {
            type = 'h';
            member = reinterpret_cast<Value segy::*>(m);
        } else if (std::is_same<T, unsigned short>::value) {
            type = 'u';
            member = reinterpret_cast<Value segy::*>(m);
        } else {
            std::cout << "unsupported type: " << type << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    ~SEGYelement() {}

    char GetType() { return type; }

    template<typename T>
    void Set(T segy::*m, T v) { // sets the value of segyelement
        if (std::is_same<T, int>::value) {
            type = 'i';
            member = reinterpret_cast<Value segy::*>(m);
            value.i = v;
        } else if (std::is_same<T, short>::value) {
            type = 'h';
            member = reinterpret_cast<Value segy::*>(m);
            value.h = v;
        } else {
            std::cout << "unsupported type: " << type << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    void Write(segy *trace) { // writes the value of segyelement to a trace
        if (type == 'i') {
            int segy::*member_int = reinterpret_cast<int segy::*>(member);
            trace->*member_int = value.i;
        } else if (type == 'h') {
            short segy::*member_short = reinterpret_cast<short segy::*>(member);
            trace->*member_short = value.h;
        } else {
            std::cout << "unsupported type: " << type << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    bool IsEqual(segy trace) {
        if (this->type == 'i') {
            return (trace.*(this->member)).i == this->value.i;
        } else if (this->type == 'h') {
            return (trace.*(this->member)).h == this->value.h;
        } else {
            std::cout << "unsupported type: " << this->type << std::endl;
            exit(EXIT_FAILURE);
        }
    }
};

class BHelement : public SEGelement {
private:
    // indication of the data type of the bhed member.
    char bh_type;

public:
    /*! pointer to an element in bhed struct of type Value called bh_member
     * Value is a union containing the different data types that
     * can be used
     */
    Value bhed::*bh_member;
    Value bh_value;

    // the constructor if it is not given any attributes , it will do nothing
    BHelement() {}

    /*! if the constructor is called with giving it parameters
     * it will perform the set function and return a variable of type T
     * @tparam T : the return values is of type T  which  is a template can be any
     * datatype.
     * @param bh_m : input parameter pointer to a member in bhed struct of type T
     * @param bh_v : input parameter variable of type T
     */
    template<typename T>
    BHelement(T bhed::*bh_m, T bh_v) {
        Set(bh_m, bh_v);
        classType = 'b';
    }

    template<typename T>
    BHelement(T bhed::*bh_m) {
        Set(bh_m);
        classType = 'b';
    }

    template<typename T>
    void Set(T bhed::*bh_m) {

        /*!is_same compare two types It will evaluate as boolean,
         *  true if the types are the same and false if otherwise.
         *  ::value to convert the boolean values to constant integral
         */

        // if the type of bh_m is integer
        if (std::is_same<T, int>::value) {
            // set the bh_type variable inside class to 'i'
            bh_type = 'i';
            /*!converting the bh_m to a (pointer to a member of bhed struct of type
             * Value) in bh_member variable so now bh_member variable is pointer to a
             * member of bhed struct of type Value and points to the bh_m
             */
            bh_member = reinterpret_cast<Value bhed::*>(bh_m);
        }
            // if the type of bh_m is short
        else if (std::is_same<T, short>::value) {
            // set the bh_type variable inside class to 'h'
            bh_type = 'h';
            /*!converting the bh_m to a (pointer to a member of bhed struct of type
             * Value) in bh_member variable so now bh_member variable is pointer to a
             * member of bhed struct of type Value and points to the bh_m
             */
            bh_member = reinterpret_cast<Value bhed::*>(bh_m);

            // if the type of bh_m is unsigned short
        } else if (std::is_same<T, unsigned short>::value) {
            // set the bh_type variable inside class to 'u'
            bh_type = 'u';
            /*!converting the bh_m to a (pointer to a member of bhed struct of type
             * Value) in member variable so now bh_member variable is pointer to a
             * member of bhed struct of type Value and points to the bh_m
             */
            bh_member = reinterpret_cast<Value bhed::*>(bh_m);
        } else {
            std::cout << "unsupported type: " << bh_type << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    // the deconstructor do no thing
    ~BHelement() {}

    // return the type of the member passed from the bhed struct
    char GetType() { return bh_type; }

    /*! the set function returns a variable of type T
     * @tparam T : is of type T  which  is a template  can be any datatype.
     * @param bh_m : input parameter pointer to a member in bhed struct of type T
     * @param bh_v : input parameter variable of type T
     */
    template<typename T>
    void Set(T bhed::*bh_m, T bh_v) {
        std::cout << typeid(bh_member).name() << std::endl;
        std::cout << typeid(bh_m).name() << std::endl;
        // if the type of bh_m is integer
        if (std::is_same<T, int>::value) {
            // set the bh_type variable inside class to 'i'
            bh_type = 'i';
            /*!converting the bh_m to a (pointer to a member of bhed struct of type
             * Value) in bh_member variable so now bh_member variable is pointer to a
             * member of bhed struct of type Value and points to the bh_m
             */
            bh_member = reinterpret_cast<Value bhed::*>(bh_m);
            // the variable i inside bh_Value construct which is of type int will
            // equal bh_v
            bh_value.i = bh_v;
        }
            // if the type of m is short
        else if (std::is_same<T, short>::value) {
            // set the bh_type variable inside class is 'h'
            bh_type = 'h';
            /*!converting the bh_m to a (pointer to a member of bhed struct of type
             * Value) in bh_member variable so now bh_member variable is pointer to a
             * member of bhed struct of type Value and points to the bh_m
             */
            bh_member = reinterpret_cast<Value bhed::*>(bh_m);
            // the variable h inside b.Value construct which is of type short will
            // equal bh_v
            bh_value.h = bh_v;
        } else if (std::is_same<T, unsigned short>::value) {
            // set the bh_type variable inside class to 'u'
            bh_type = 'u';
            /*!converting the bh_m to a (pointer to a member of bhed struct of type
             * Value) in member variable so now bh_member variable is pointer to a
             * member of bhed struct of type Value and points to the bh_m
             */
            bh_member = reinterpret_cast<Value bhed::*>(bh_m);
            // the variable u inside b.Value construct which is of type short will
            // equal bh_v
            bh_value.u = bh_v;
        } else {
            std::cout << "unsupported type: " << bh_type << std::endl;
            exit(EXIT_FAILURE);
        }
        std::cout << typeid(bh_member).name() << std::endl;
        std::cout << typeid(bh_m).name() << std::endl;
    }

    /*!
     *
     * @param bh_data :input parameter a pointer to struct of type bhed
     */
    void Write(bhed *bh_data) { // writes the value of bhelement to a bh_data
        // if the type is integer
        if (bh_type == 'i') {
            /*!converting the bh_member to a (pointer to a member of bhed struct of
             * type int) in bh_member_int so now bh_member_int variable is pointer to
             * a member of bhed struct of type int and points to the bh_member
             */
            int bhed::*bh_member_int = reinterpret_cast<int bhed::*>(bh_member);
            // write the bh_value .i to bh_member_init which writes the value inside
            // the address pointed by bh_member
            bh_data->*bh_member_int = bh_value.i;
        }
            // if the type is short
        else if (bh_type == 'h') {
            /*!converting the bh_member to a (pointer to a member of bhed struct of
             * type int) in bh_member_int so now bh_member_short variable is pointer
             * to a bh_member of bhed struct of type short and points to the bh_member
             */
            short bhed::*bh_member_short = reinterpret_cast<short bhed::*>(bh_member);
            // write the bh_value .h to bh_member_short which writes the value inside
            // the address pointed by bh_member
            bh_data->*bh_member_short = bh_value.h;
        } else if (bh_type == 'u') {
            /*!converting the bh_member to a (pointer to a member of bhed struct of
             * type unsigned short) in bh_member_unsigned_short so now
             * bh_member_unsigned_short variable is pointer to a bh_member of bhed
             * struct of type unsigned_short and points to the bh_member
             */
            unsigned short bhed::*bh_member_unsigned_short =
                    reinterpret_cast<unsigned short bhed::*>(bh_member);
            // write the bh_value .u to bh_member_unsigned_short which writes the
            // value inside the address pointed by bh_member
            bh_data->*bh_member_unsigned_short = bh_value.u;
            std::cout << typeid(bh_member_unsigned_short).name() << std::endl;
        } else {
            std::cout << "unsupported type: " << bh_type << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    bool IsEqual(bhed bh_data) {
        if (this->bh_type == 'i') {
            // go to bh_member which is pointer to bhed element of type int if the i
            // inside it equals value.i  return 1
            return (bh_data.*(this->bh_member)).i == this->bh_value.i;
        } else if (this->bh_type == 'h') {
            // go to bh_member which is pointer to bhed element of type short if the h
            // inside it equals value.h  return 1
            return (bh_data.*(this->bh_member)).h == this->bh_value.h;
        } else if (this->bh_type == 'u') {
            // go to bh_member which is pointer to bhed element of type unsigned short
            // if the u inside it equals value.h  return 1
            return (bh_data.*(this->bh_member)).u == this->bh_value.u;
        } else {
            std::cout << "unsupported type: " << this->bh_type << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    void ShowType() {
        std::cout << classType << " with dat_type of : " << bh_type
                  << " at address: " << (this->bh_member)
                  << " with value: " << bh_value.u << std::endl;
    };
};

#endif /* SEGYELEMENT_H */
