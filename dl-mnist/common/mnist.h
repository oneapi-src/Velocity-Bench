/* Copyright (C) 2023 Intel Corporation
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom
 * the Software is furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
 * OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
 * OR OTHER DEALINGS IN THE SOFTWARE.
 * 
 * SPDX-License-Identifier: MIT
 */


#ifndef DEEP_LEARNING_MNIST_H_
#define DEEP_LEARNING_MNIST_H_

#include <stdio.h>
#include <iostream>
#include <fstream>
#include "exec_policies.h" 

using namespace std;

unsigned char* read_mnist_labels(struct Exec_setup exec_setup, string full_path, int& number_of_labels);
unsigned char** read_mnist_images(struct Exec_setup exec_setup, string full_path, int& number_of_images, int& image_size);
float** read_mnist_images_float(struct Exec_setup exec_setup, string full_path, int& number_of_images, int& image_size);
float* read_mnist_images_float2(struct Exec_setup exec_setup, string full_path, int& total_dataset_images_count, int& image_size);
float* read_mnist_images_float2(int no_of_images, string full_path, int& total_dataset_images_count, int& image_size);
void print_images_and_labels(unsigned char* _dataset, int labelCount, unsigned char** mnist_images, int number_of_images, int image_size);
void print_images_and_labels_first10(unsigned char* _dataset, int labelCount, unsigned char** mnist_images, int number_of_images, int image_size);
void print_images_and_labels_float(unsigned char* _dataset, int labelCount, float** mnist_images, int number_of_images, int image_size);
void print_images(float* mnist_images, int number_of_images, int image_size);
#endif