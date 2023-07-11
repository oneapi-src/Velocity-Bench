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

#ifndef THOTH_LOADERS_BUFFER_HPP
#define THOTH_LOADERS_BUFFER_HPP

#include <cstdio>

namespace thoth {
    namespace streams {
        class Buffer {
            /**
             * @brief Buffer class, loads bytes from any file with any format to be parsed by the Parser
             * A buffer structure has Buffer Gap (empty area) and Buffer Filled Area (denoted here as buffer)
             * A buffer size is expandable, default size = 3200
             * A file Buffer can be used by reader and writer; input byte and output byte streaming
             * Buffer Keeps track of 5 pointers and a sequential block inside the buffer for
             * inserting new bytes(chars) from file. The five pointers are:
             * 1. Head of the buffer (start of buffer filled area)
             * 2. Start of the buffer gap
             * 3. First location outside the buffer gap area
             * 4. End of the buffer
             * 5. Control pointer, can be set at any location within the buffer,
             *    MUST BE in buffer filled area, denoted here as Point
             */
        public:
            static const int DEFAULT_GAP_SIZE = 3200;

            /**
             *
             * @brief Buffer constructor with default gap size
             *
             * @param[in] aGapSize
             *
             */
            explicit Buffer(int aGapSize = DEFAULT_GAP_SIZE);

            /**
             *
             * @brief Buffer constructor with an existing file given its pointer as an argument
             *
             * @param[in] aGapSize, aFile
             *
             */
            explicit Buffer(FILE *aFile, int aGapSize = DEFAULT_GAP_SIZE);

            /**
             * @brief Buffer copy constructor
             */
            Buffer(const Buffer &bf);

            /**
             * @brief Buffer destructor
             */
            ~Buffer();

            /**
             *
             * @brief Initialize Buffer
             *
             * @param aNewSize
             *
             * @return 1 for successful buffer initialization, 0 otherwise
             */
            int InitBuffer(unsigned int aNewSize);

            /**
             *
             * @brief Copy the characters from one location to another
             *
             * @param[in] aDestination, aSource, aLength
             *
             * @return 1 upon successful copy, 0 otherwise
             *
             */
            int CopyBytes(char *aDestination, char *aSource, unsigned int aLength);

            /**
             *
             * @brief Expand the size of the buffer
             *
             * @param[in] aNewSize
             *
             */
            void ExpandBuffer(unsigned int aNewSize);

            /**
             *
             * @brief Expand the size of the buffer gap
             *
             * @param[in] aNewSize
             *
             */
            void ExpandGap(unsigned int aNewSize);

        public:
            /**
             *
             *  @brief Getter for buffer size
             *
             *  @return the size of the buffer minus the gap (size of the filled are)
             *
             */
            unsigned long GetBufferSize();

            /**
             *  @brief Move the gap to the current position of the mpPoint
             */
            void MoveGapStartToPoint();

            /**
             *
             * @brief Set mpPoint to offset given from the starting location of buffer
             *
             * @param[in] aOffset
             *
             */
            void SetPoint(int aOffset);

            /**
             *
             * @brief Set mpPoint to given address
             *
             * @param[in] aAddress
             *
             */
            void SetPoint(char *aAddress);

            /**
             *
             * @brief Get mpPoint
             *
             */
            char *GetPoint();

            /**
             *
             * @brief Get mpGapStart
             *
             */
            char *GetGapStart();

            /**
             *
             * @brief Get mpGapEnd
             *
             */
            char *GetGapEnd();

            /**
             *
             * @brief Get mpBufferHead
             *
             */
            char *GetBufferHead();

            /**
             *
             * @brief Get mpBufferEnd
             *
             */
            char *GetBufferEnd();

            /**
             * @brief Getter for the current size of the buffer gap area
             */
            unsigned long GetSizeOfGap();

            /**
             *  @brief Function to get the offset from the starting of buffer to mpPoint current location
             */
            unsigned int PointOffsetFromStart();

            /**
             *
             * @brief Convert to character that mpPoint is pointing to
             * If mpPoint is inside the buffer gap area, then return the
             * the first character outside the gap
             *
             * @return Character representation to what mpPoint is pointing to
             * mpPoint must be in buffer filled area or at buffer gap start
             *
             */
            char ToChar();

            /**
             *
             * @brief Replace the character pointed by mpPoint by the given one
             * Does not set the buffer gap, just replacing
             *
             * @param[in] aNewChar
             *
             */
            void ReplaceChar(char aNewChar);

            /**
             *
             * @brief Getter to the next character pointed to by mpPoint and increment mpPoint
             *
             * @return Character pointed to by mpPoint before setting
             *
             */
            char GetNextChar();

            /**
             *
             *  @brief Inserts the given character at mpPoint location
             *  and advance mpPoint
             *
             *  @param[in] aNewChar
             *
             */
            void InsertCharAndAdvancePoint(char aNewChar);

            /**
             *
             * @brief  Insert character at mpPoint position
             *  does NOT advance the point
             *  If there is no space in gap area, expand buffer gap area
             *  with one byte
             *
             *  param[in] aNewChar
             *
             */
            void InsertChar(char aNewChar);

            /**
             *
             * @brief Delete given number of characters
             *
             * @param[in] aCharsNum
             *
             */
            void DeleteChars(unsigned int aBytesLength);

            /**
             * @brief Prints out the current buffer from start to end
             */
            void PrintBuffer();


        private:
            ///Byte pointer must be inside buffer filled area and can not be inside buffer gap
            ///can point to any byte inside buffer filled area
            char *mpPoint;

            ///Pointer to first byte inside buffer filled area (head of buffer)
            char *mpBufferHead;

            ///Pointer to first byte outside the buffer filled area
            char *mpBufferEnd;

            ///Pointer to first bye inside buffer gap area
            char *mpGapStart;

            ///Pointer to first byte outside buffer gap area
            char *mpGapEnd;

            ///Gap size, used in buffer size expansion function
            unsigned int mGapSize;

        };
    }//namespace streams
}//namespace thoth


#endif //THOTH_LOADERS_BUFFER_HPP
