/*
MIT License

Copyright (c) 2015 University of West Bohemia

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

/*
MIT License

Modifications Copyright (C) 2023 Intel Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

SPDX-License-Identifier: MIT License
*/

#ifndef _UTILS_H_
#define _UTILS_H_

#include <iostream>

#define SUCCESS 0
#define FAILURE 1
#define INPUT_ARGS_MIN 3

#define TRUE 1
#define FALSE 0

#define LINEAR_STRING "linear"
#define POLY_STRING "polynomial"
#define RBF_STRING "rbf"
#define SIGMOID_STRING "sigmoid"

#define LOAD_FLAG_ALL_RAM 0
#define LOAD_FLAG_ALPHAS_WC 1
#define LOAD_FLAG_VECTORS_WC 2 
#define LOAD_FLAG_ALL_WC 3
#define LOAD_FLAG_TRANSPOSE 4
#define LOAD_FLAG_FILE_BUFFER 8

#define ALLOC_ALPHAS_WC(X) (X & 1)
#define ALLOC_VECTORS_WC(X) (X & 2)
#define DO_TRANSPOSE_MATRIX(X) (X & 4)
#define USE_BUFFER(X) (X & 8)

#define ALIGN_UP(x, align) ((align) * (((x) + align - 1) / (align)))

#define SAFE_CALL(call) {int err = call; if (SUCCESS != err) { fprintf (stderr, "Error %i in file %s at line %i.", err, __FILE__, __LINE__ ); exit(EXIT_FAILURE);}}

#define REPORT_ERROR(error_string) { fprintf (stderr, "Error ""%s"" thrown in file %s at line %i.", error_string, __FILE__, __LINE__ ); exit(EXIT_FAILURE);}
#define REPORT_WARNING(error_string) { fprintf (stderr, "Warning ""%s"" raised in file %s at line %i.", error_string, __FILE__, __LINE__ );}

#define FILE_SAFE_OPEN(FID, FNAME, MODE) if ((FID = fopen(FNAME, MODE)) == 0) { \
	std::cerr << "Error: Can't open file \"" << FNAME << "\"!\n"; \
	exit(EXIT_FAILURE); \
}

#define MEM_SAFE_ALLOC(MEM_PTR, TYPE, LEN) if (MEM_PTR != NULL) { \
	free(MEM_PTR); \
} \
if ((MEM_PTR = (TYPE *) malloc((LEN) * sizeof(TYPE))) == NULL) { \
	std::cerr << "Error: Unable to allocate " << (LEN) * sizeof(TYPE) << " B of memory!\n"; \
	exit(EXIT_FAILURE); \
}

#define MEM_SAFE_FREE(MEM_PTR) if (MEM_PTR != NULL) { \
	free(MEM_PTR); \
	MEM_PTR = NULL; \
}

#define BUFFER_LENGTH 16
#define LINE_LENGTH 65535
#define EOL '\n'

/* kernel_type */
#define K_LINEAR 1
#define K_POLYNOMIAL 2
#define K_RBF 3
#define K_SIGMOID 4

#define INIT_DIST 10.0F

#define ABS(X) (((X) < 0) ? -(X) : (X))

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

#include "svm.h" //some common types used from libSVM
#include "svm_template.h"
#include "cuda_utils.h"

static const char *kernel_type_table[]=
{
	"linear","polynomial","rbf","sigmoid","precomputed",NULL
};

void print_help();

void exit_input_error(int line_num);

//void load_data_dense(FILE * &fid, float * &alphas, float * &vectors, unsigned int &height,
//					 unsigned int &width, svm_memory_dataformat format_type);

void cusvm_load_model2(char *filename, struct cuSVM_model * &model);

int get_closest_label(int * labels, int labels_len, float alpha);

void malloc_host_WC(void** ptr, size_t size);
void malloc_host_PINNED(void** ptr, size_t size);

void select_device(int device_id, unsigned int sm_limit, int is_max);

int equals(char *str1, char *str2);
char *next_string(char *str);
char *next_string_spec(char *str);
char *next_string_spec_space(char *str);
char *next_string_spec_space_plus(char *str);
char *next_string_spec_colon(char *str);
char * next_eol(char * str);
char *last_string_spec_colon(char *str);
unsigned int strtoi_reverse(char * str);
float strtof_fast(char * str, char ** end_ptr);

long long parse_line(FILE * & fid, char * & line, char * & line_end, size_t & line_len);

#endif /* _UTILS_H_ */
