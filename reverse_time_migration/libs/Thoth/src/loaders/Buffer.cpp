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
// Created by pancee on 12/15/20.
//

#include <thoth/loaders/Buffer.hpp>

#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <sys/stat.h>

#define BYTE 1

using namespace thoth::streams;

Buffer::Buffer(int aGapSize) : mGapSize(aGapSize) {
    InitBuffer(this->mGapSize);
}

Buffer::Buffer(FILE *aFile, int aGapSize) : mGapSize(aGapSize) {

    // get the size of the file then create
    // a Buffer of file size + mGapSize
    struct stat buf{};
    fstat(fileno(aFile), &buf);
    long file_size = buf.st_size; //get file size from status structure defined

    InitBuffer(file_size + this->mGapSize);
    MoveGapStartToPoint();
    ExpandGap((int) file_size);
    unsigned int bytes_num = fread(this->mpGapStart, BYTE, file_size, aFile);

    this->mpGapStart += bytes_num;
}

//Copy constructor
Buffer::Buffer(const Buffer &bl) {
    this->mGapSize = bl.mGapSize;

    this->mpBufferHead = (char *) malloc(bl.mpBufferEnd - bl.mpBufferHead);

    strcpy(this->mpBufferHead, bl.mpBufferHead);

    this->mpBufferEnd = this->mpBufferHead + (bl.mpBufferEnd - bl.mpBufferHead);
    this->mpGapStart = this->mpBufferHead + (bl.mpGapStart - bl.mpBufferHead);
    this->mpGapEnd = this->mpGapStart + (bl.mpGapEnd - bl.mpGapStart);
    this->mpPoint = this->mpBufferHead + (bl.mpPoint - bl.mpBufferHead);
}

Buffer::~Buffer() {
    if (this->mpBufferHead) {
        free(this->mpBufferHead);
    }
}

int Buffer::CopyBytes(char *aDestination, char *aSource, unsigned int aLength) {

    if ((aDestination == aSource) || (aLength == 0)) {
        return 1;
    }
    // Check if we are moving the character toward the front of the buffer filled area
    if (aSource > aDestination) {

        // Check that last byte to be copied at the source is within the buffer filled area
        if ((aSource + aLength) >= this->mpBufferEnd) {
            return 0;
        }

        for (; aLength > 0; aLength--) {
            *(aDestination++) = *(aSource++);
        }

    } else {

        // Copy direction is reversed, start from the last byte at source to destination
        // till reaching the first one, to ensure successful copying
        aSource += aLength - 1;
        aDestination += aLength - 1;

        for (; aLength > 0; aLength--) {
            *(aDestination--) = *(aSource--);
        }
    }
    return 1;
}

void Buffer::ExpandBuffer(unsigned int aNewSize) {

    // Check if we actually need to expand the buffer
    if (((this->mpBufferEnd - this->mpBufferHead) + aNewSize) > this->GetBufferSize()) {

        char *old_buffer_head = this->mpBufferHead;

        unsigned long new_buffer_size = (this->mpBufferEnd - this->mpBufferHead) + aNewSize + this->mGapSize;

        //Reallocate mpBufferHead by new buffer size
        this->mpBufferHead = (char *) realloc(this->mpBufferHead, new_buffer_size);

        unsigned long buffer_offset = this->mpBufferHead - old_buffer_head;
        this->mpPoint += buffer_offset;
        this->mpBufferEnd += buffer_offset;
        this->mpGapStart += buffer_offset;
        this->mpGapEnd += buffer_offset;
    }
}


//Move the gap to the current position of mpPoint
//The point should end in same location as mpGapStart.
void Buffer::MoveGapStartToPoint() {

    //If already mpPoints points to the first byte at buffer gap area
    //return, as there is nothing to do
    if (this->mpPoint == this->mpGapStart) {
        return;
    }

    //If mpPoint points to the end of buffer gap area
    //reset mpPoint to point to the start of buffer gap area
    //then return
    if (this->mpPoint == this->mpGapEnd) {
        this->mpPoint = this->mpGapStart;
        return;
    }

    //If mpPoint proceeds start of buffer gap area
    if (this->mpPoint < this->mpGapStart) {
        // Move buffer gap area upwards
        CopyBytes(this->mpPoint + (this->mpGapEnd - this->mpGapStart),
                  this->mpPoint, this->mpGapStart - this->mpPoint);
        this->mpGapEnd -= (this->mpGapStart - this->mpPoint);
        this->mpGapStart = this->mpPoint;
    }
        //If start of buffer gap area preceeds mpPoint
    else {
        CopyBytes(this->mpGapStart, this->mpGapEnd, this->mpPoint - this->mpGapEnd);
        this->mpGapStart += (this->mpPoint - this->mpGapEnd);
        this->mpGapEnd = this->mpPoint;
        this->mpPoint = this->mpGapStart;
    }
}

// Expand the size of the gap
// If the required size is less then the current gap size, do nothing
// If the size is greater than the current size, increase the gap to the default
// buffer gap size + aNewSize
void Buffer::ExpandGap(unsigned int aNewSize) {

    if (aNewSize > this->GetSizeOfGap()) {
        aNewSize += this->mGapSize;
        ExpandBuffer(aNewSize);
        CopyBytes(this->mpGapEnd + aNewSize, this->mpGapEnd,
                  this->mpBufferEnd - this->mpGapEnd);

        this->mpGapEnd += aNewSize;
        this->mpBufferEnd += aNewSize;
    }
}

void Buffer::SetPoint(int aOffset) {

    this->mpPoint = this->GetBufferHead() + aOffset;

    if (this->mpPoint > this->mpGapStart) {
        this->mpPoint += (this->mpGapEnd - this->mpGapStart);
    }
}

void Buffer::SetPoint(char *aAddress) {

    this->mpPoint = aAddress;

    if (this->mpPoint > this->mpGapStart) {
        this->mpPoint += (this->mpGapEnd - this->mpGapStart);
    }
}

char *Buffer::GetPoint() {
    return this->mpPoint;
}

char *Buffer::GetGapStart() {
    return this->mpGapStart;
}

char *Buffer::GetGapEnd() {
    return this->mpGapEnd;
}

char *Buffer::GetBufferHead() {
    return this->mpBufferHead;
}

char *Buffer::GetBufferEnd() {
    return this->mpBufferEnd;
}

unsigned long Buffer::GetSizeOfGap() {
    return this->mpGapEnd - this->mpGapStart;
}

unsigned int Buffer::PointOffsetFromStart() {
    unsigned int offset;
    if (this->mpPoint > this->mpGapEnd) {
        offset = ((this->mpPoint - this->mpBufferHead) - (this->mpGapEnd - this->mpGapStart));
    } else {
        offset = (this->mpPoint - this->mpBufferHead);
    }
    return offset;
}

char Buffer::ToChar() {
    if (this->mpPoint == this->mpGapStart) {
        this->mpPoint = this->mpGapEnd;
    }
    return *this->mpPoint;
}

void Buffer::ReplaceChar(char aNewChar) {
    if (this->mpPoint == this->mpGapStart) {
        this->mpPoint = this->mpGapEnd;
    }
    if (this->mpPoint == this->mpBufferEnd) {
        ExpandBuffer(BYTE);
        this->mpBufferEnd++;
    }
    *this->mpPoint = aNewChar;
}

char Buffer::GetNextChar() {
    // mpPoint should not be in the gap
    if (this->mpPoint == this->mpGapStart) {
        this->mpPoint = this->mpGapEnd;
        return *this->mpPoint;
    }
    return *(++this->mpPoint);
}

void Buffer::InsertCharAndAdvancePoint(char aNewChar) {
    InsertChar(aNewChar);
    this->mpPoint++;
}

void Buffer::InsertChar(char aNewChar) {
    // Here we need to move the gap if the point
    // is not already at the start of the gap
    if (this->mpPoint != this->mpGapStart) {
        MoveGapStartToPoint();
    }
    // Check if there is space in gap, if there is no space
    // expand gap with one byte
    if (this->mpGapStart == this->mpGapEnd) {
        ExpandGap(BYTE);
    }
    *(this->mpGapStart++) = aNewChar;
}

void Buffer::DeleteChars(unsigned int aBytesLength) {
    if (this->mpPoint != this->mpGapStart) {
        MoveGapStartToPoint();
    }
    this->mpGapEnd += aBytesLength;
}

int Buffer::InitBuffer(unsigned int aSize) {

    if (this->mpBufferHead) {
        free(this->mpBufferHead);
    }

    this->mpBufferHead = (char *) malloc(aSize);

    //Make sure mpBufferHead is allocated correctly
    if (!this->mpBufferHead) {
        return 0;
    }

    this->mpPoint = this->mpBufferHead;

    //Still there is no bytes in buffer filled area
    this->mpGapStart = this->mpBufferHead;

    this->mpGapEnd = this->mpBufferHead + aSize;
    this->mpBufferEnd = this->mpGapEnd;

    return 1;
}

unsigned long Buffer::GetBufferSize() {
    return (this->mpBufferEnd - this->mpBufferHead) - (this->mpGapEnd - this->mpGapStart);
}

void Buffer::PrintBuffer() {
    char *temp = this->mpBufferHead;

    //While temp pointer is still in buffer filled area
    while (temp < this->mpBufferEnd) {
        //If buffer is empty, do nothing
        if ((temp >= this->mpGapStart) && (temp < this->mpGapEnd)) {
            std::cout << "_";
            temp++;
        } else {
            std::cout << *(temp++);
        }
    }
    std::cout << std::endl;
}
