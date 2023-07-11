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


// The code below contains code snippets taken from a publicly avaliable post on stackoverflow
// and modified. It is on how to read MNIST files:
// https://stackoverflow.com/questions/8286668/how-to-read-mnist-data-in-c


#include <stdio.h>
#include <iostream>
#include <fstream>

#include "mnist.h" 
#include "exec_policies.h"
#include "utils.h"
#include "tracer.h"

using namespace std;
using namespace dl_infra::common;

unsigned char* read_mnist_labels(struct Exec_setup exec_setup, string full_path, int& number_of_labels) {
    Tracer::func_begin("read_mnist_labels");   

    auto reverseInt = [](int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };

    typedef unsigned char uchar;

    ifstream file(full_path, ios::binary);

    if(file.is_open()) {
        int magic_number = 0;
        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if(magic_number != 2049) throw runtime_error("Invalid MNIST label file!");

        file.read((char *)&number_of_labels, sizeof(number_of_labels)), number_of_labels = reverseInt(number_of_labels);

        uchar* _dataset = new uchar[number_of_labels];
        for(int i = 0; i < number_of_labels; i++) {
            file.read((char*)&_dataset[i], 1);
        }

        Tracer::func_end("read_mnist_labels");         

        return _dataset;
    } else {
        throw runtime_error("Unable to open file `" + full_path + "`!");
    }
}

unsigned char** read_mnist_images(struct Exec_setup exec_setup, string full_path, int& number_of_images, int& image_size) {
    Tracer::func_begin("read_mnist_images");   

    auto reverseInt = [](int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };

    typedef unsigned char uchar;

    ifstream file(full_path, ios::binary);

    if(file.is_open()) {
        int magic_number = 0, n_rows = 0, n_cols = 0;

        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if(magic_number != 2051) throw runtime_error("Invalid MNIST image file!");

        file.read((char *)&number_of_images, sizeof(number_of_images)), number_of_images = reverseInt(number_of_images);
        file.read((char *)&n_rows, sizeof(n_rows)), n_rows = reverseInt(n_rows);
        file.read((char *)&n_cols, sizeof(n_cols)), n_cols = reverseInt(n_cols);

        image_size = n_rows * n_cols;

        uchar** _dataset = new uchar*[number_of_images];
        for(int i = 0; i < number_of_images; i++) {
            _dataset[i] = new uchar[image_size];
            file.read((char *)_dataset[i], image_size);
        }

        Tracer::func_end("read_mnist_images");    

        return _dataset;
    } else {
        throw runtime_error("Cannot open file `" + full_path + "`!");
    }
}

float** read_mnist_images_float(struct Exec_setup exec_setup, string full_path, int& total_dataset_images_count, int& image_size) {
    Tracer::func_begin("read_mnist_images_float");   

    auto reverseInt = [](int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };

    typedef unsigned char uchar;

    ifstream file(full_path, ios::binary);

    if(file.is_open()) {
        int magic_number = 0, n_rows = 0, n_cols = 0;

        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if(magic_number != 2051) throw runtime_error("Invalid MNIST image file!");

        file.read((char *)&total_dataset_images_count, sizeof(total_dataset_images_count)), total_dataset_images_count = reverseInt(total_dataset_images_count);
        file.read((char *)&n_rows, sizeof(n_rows)), n_rows = reverseInt(n_rows);
        file.read((char *)&n_cols, sizeof(n_cols)), n_cols = reverseInt(n_cols);

        image_size = n_rows * n_cols;

        float** _dataset = new float*[total_dataset_images_count];
        for(int i = 0; i < total_dataset_images_count; i++) {
            uchar* charIm = new uchar[image_size];
            file.read((char *)charIm, image_size);
             _dataset[i] = new float[image_size];
            for(int j=0; j<image_size; j++) {
                _dataset[i][j] = charIm[j];         
            }
            delete[] charIm;
        }

        Tracer::func_end("read_mnist_images_float");    

        return _dataset;
    } else {
        throw runtime_error("Cannot open file `" + full_path + "`!");
    }
}


float* read_mnist_images_float2(int no_of_images, string full_path, int& total_dataset_images_count, int& image_size) {
    Tracer::func_begin("read_mnist_images_float2");   

    auto reverseInt = [](int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };

    typedef unsigned char uchar;

    ifstream file(full_path, ios::binary);

    if(file.is_open()) {
        int magic_number = 0, n_rows = 0, n_cols = 0;

        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if(magic_number != 2051) throw runtime_error("Invalid MNIST image file!");

        file.read((char *)&total_dataset_images_count, sizeof(total_dataset_images_count)), total_dataset_images_count = reverseInt(total_dataset_images_count);
        file.read((char *)&n_rows, sizeof(n_rows)), n_rows = reverseInt(n_rows);
        file.read((char *)&n_cols, sizeof(n_cols)), n_cols = reverseInt(n_cols);

        image_size = n_rows * n_cols;

        if(no_of_images > total_dataset_images_count) {
            throw runtime_error("Not enough images in dataset");
        }

        int total_size = no_of_images *image_size;
        uchar* charIm = new uchar[total_size];
       // cout << "read_mnist_images_float2 - before reading the file, total_size = " << total_size << std::endl;
        file.read((char *)charIm, total_size);
        //cout << "read_mnist_images_float2 - after reading the file" << std::endl;
        
        float* floatIm = new float[total_size];

        for(int i=0; i<total_size; i++) {
            floatIm[i] =  static_cast<float>(charIm[i]);
        }
        //cout << "read_mnist_images_float2 - after converting from uchar to float" << std::endl;
        // for(int i=0; i<28*28; i++) {
        //     cout << static_cast<unsigned>(charIm[i]) << std::endl;
        // }

        // for(int i=0; i<28*28; i++) {
        //     cout << static_cast<float>(floatIm[i]) << std::endl;
        // }


        delete[] charIm;

        Tracer::func_end("read_mnist_images_float2");    

        return floatIm;
    } else {
        throw runtime_error("Cannot open file `" + full_path + "`!");
    }
}

float* read_mnist_images_float2(struct Exec_setup exec_setup, string full_path, int& total_dataset_images_count, int& image_size) {
    return read_mnist_images_float2(exec_setup.no_of_images_per_minibatch, std::move(full_path), total_dataset_images_count, image_size);
}


void print_images_and_labels_first10(unsigned char* _dataset, int labelCount, unsigned char** mnist_images, int number_of_images, int image_size) {
    Tracer::func_begin("print_images_and_labels_first10");   

   printf("Printing first 10 labels");
   for(int i=0; i<10; i++) {
       cout << static_cast<unsigned>(_dataset[i]) << std::endl;
   }
   printf("Label count = %d\n", labelCount);

   printf("Printing first pixel values labels");
   for(int i=0; i<1; i++) {
       for(int j=0; j<10; j++) {
           cout << static_cast<unsigned>(mnist_images[i][j]) << ", ";
       }
       cout << std::endl;
   }
   printf("Number of images = %d\n", number_of_images);
   printf("Image size = %d\n", image_size);

    Tracer::func_end("print_images_and_labels_first10");    

}


void print_images_and_labels(unsigned char* _dataset, int labelCount, unsigned char** mnist_images, int number_of_images, int image_size) {
    Tracer::func_begin("print_images_and_labels");   

   for(int i=0; i<labelCount; i++) {
       cout << static_cast<unsigned>(_dataset[i]) << std::endl;
   }
   printf("Label count = %d\n", labelCount);

   
   for(int i=0; i<number_of_images; i++) {
       for(int j=0; j<image_size; j++) {
           if( static_cast<float>(mnist_images[i][j]) != 0.0) {
            cout << static_cast<float>(mnist_images[i][j]) << std::endl;
           }
       }
   }
   printf("Number of images = %d\n", number_of_images);
   printf("Image size = %d\n", image_size);

    Tracer::func_end("print_images_and_labels");    
}

void print_images_and_labels_float(unsigned char* _dataset, int labelCount, float** mnist_images, int number_of_images, int image_size) {
    Tracer::func_begin("print_images_and_labels_float");   

   for(int i=0; i<labelCount; i++) {
       cout << static_cast<unsigned>(_dataset[i]) << std::endl;
   }
   printf("Label count = %d\n", labelCount);

   
   for(int i=0; i<number_of_images; i++) {
       for(int j=0; j<image_size; j++) {
           if( mnist_images[i][j] != 0.0) {
            cout << mnist_images[i][j] << std::endl;
           }
       }
   }
   printf("Number of images = %d\n", number_of_images);
   printf("Image size = %d\n", image_size);

    Tracer::func_end("print_images_and_labels_float");    

}


void print_images(float* mnist_images, int number_of_images, int image_size) {
    Tracer::func_begin("print_images");   

   for(int i=0; i<number_of_images*image_size; i++) {
       if(mnist_images[i] != 0.0) {
           cout << mnist_images[i] << "\t";
       }
   }
   cout << std::endl;
   printf("Number of images = %d\n", number_of_images);
   printf("Image size = %d\n", image_size);

    Tracer::func_end("print_images");    
}


