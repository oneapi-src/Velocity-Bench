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

#ifndef THOTH_LOADERS_BUFFER_LOADER_HPP
#define THOTH_LOADERS_BUFFER_LOADER_HPP

#include <thoth/loaders/Buffer.hpp>

#include <stdio.h>

namespace thoth {
    namespace streams {
        class BufferLoader {
            /**
             * @brief BufferLoader class, loads bytes from any file with any format to be parsed by the Parser.
             * A buffer structure has Buffer Gap (empty area) and Buffer Filled Area (denoted down as buffer)
             * A buffer size is expandable, default size = 3200
             * A file BufferLoader can be used by reader and writer; input byte and output byte streaming
             * BufferLoader Keeps track of 5 pointers and a sequential block inside the buffer for
             * inserting new bytes(chars) from file. The five pointers are:
             * 1. Head of the buffer (start of buffer filled area)
             * 2. Start of the buffer gap
             * 3. First location outside the buffer gap
             * 4. End of the buffer
             * 5. Any Location within the buffer
             */
        public:
            static const int DEFAULT_GAP_SIZE = 3200;

            /**
             *
             * @brief BufferLoader constructor with default gap size
             *
             * @param[in] aGapSize
             *
             */
            explicit BufferLoader(int aGapSize = DEFAULT_GAP_SIZE);

            /**
             *
             * @brief BufferLoader constructor with an existing file given its pointer as an argument
             *
             * @param[in] aGapSize, aFile
             *
             */
            explicit BufferLoader(FILE *aFile, int aGapSize = DEFAULT_GAP_SIZE);

            /**
             * @brief BufferLoader copy constructor
             */
            BufferLoader(const BufferLoader &bf);

            /**
             * @brief BufferLoader destructor
             */
            ~BufferLoader() = default;

            /**
             *
             * @brief Initialize Buffer
             *
             * @param aNewSize
             *
             * @return
             */
            int InitBuffer(unsigned int aNewSize);

            /**
             *
             * @brief Copy the characters from one location to another
             *
             * @param[in] aDestination, aSource, aLength
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
            int GetBufferSize();

            /**
             *  @brief Move the gap to the current position of the mpPoint
             */
            void MoveGapToPoint();

            /**
             *
             * @brief Set mpPoint to offset given from the starting location of buffer
             *
             * @param[in] aOffset
             *
             */
            void SetPoint(unsigned int aOffset);

            /**
             * @brief Getter for the current size of the buffer gap area
             */
            int GetSizeOfGap();

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
             *
             */
            char ToChar();

            /**
             *
             * @brief Getter to the previous character pointed to by mpPoint
             * makes mpPoint to its previous position
             *
             * @return previous character pointed to by mpPoint, before its setting
             *
             */
            char GetPreviousChar();

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
            void PutChar(char aNewChar);

            /**
             *
             * @brief  Insert character at mpPoint position
             *  does NOT advance the point
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
            void DeleteChars(unsigned int aCharsNum);

            /**
             * @brief Inserts a given length size of chars at mpPoint
             * does NOR advance the point
             */
            void InsertChars(char *apNewChars, unsigned int aLength);

            /**
             * @brief Prints out the current buffer from start to end
             */
            void PrintBuffer();

            /**
             *
             * @brief Saves the number of bytes starting from
             * the mpPoint to the given file
             *
             * @param[in] aFile, aOutBytes
             *
             */
            int SaveBufferToFile(FILE *aFile, unsigned int aOutBytes);

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


#endif //THOTH_LOADERS_BUFFER_LOADER_HPP
