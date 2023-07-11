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

//#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#ifndef __NO_CUDA
#include <hip/hip_runtime.h>
#include <hip/hip_runtime.h>
#endif

using namespace std;

#include "utils.h"
#include "my_stopwatch.h"

void exit_input_error(int line_num) {
    fprintf(stderr, "Wrong input format at line %d\n", line_num);
    exit(EXIT_FAILURE);
}

void load_data_dense(FILE * & fid, float * & labels, float * & alphas, float * & vectors, unsigned int &height,
                     unsigned int &width, unsigned int flags, int alignment) {
    char *buf,
        *sbuf = NULL,
        *sbuf_tmp,
        *line = NULL,
        *line_end;
    size_t i,
        it = 0,
        j,
        start_pos,
        sbuf_size = 65536,
        read_chars,
        new_len,
        line_len;
    long long line_size;
    int ret;

    if ((start_pos = ftell(fid)) == EOF) {
        cerr << "File is not openned!\n";
        exit(EXIT_FAILURE);
    }

    ret = SUCCESS;
    height = 0;
    width = 0;
    alphas = NULL;
    vectors = NULL;

    if (USE_BUFFER(flags)) {
        /* Try to store file into buffer. */
        MEM_SAFE_ALLOC(sbuf, char, sbuf_size)
        read_chars = 0;

        while (1) {
            new_len = (unsigned int) fread(sbuf + read_chars, sizeof(char), sbuf_size - read_chars, fid);

            read_chars += new_len;
            if (read_chars == sbuf_size) {
                /* Expanding buffer size. */
                sbuf_size <<= 1;
                sbuf_tmp = (char *) realloc(sbuf, sbuf_size * sizeof(float));

                if (sbuf_tmp == NULL) {
                    /* Not enough memory. */
                    printf("Warning: Not enough memory - buffering disabled!\n");
                    sbuf_size = 0;
                    free(sbuf);
                    if (fseek(fid, start_pos, SEEK_SET) != 0)
                        exit(EXIT_FAILURE);

                    break;
                } else {
                    sbuf = sbuf_tmp;
                }
            } else {
                /* File loaded successfully. */
                sbuf[read_chars++] = 0;
                sbuf_size = read_chars;
                sbuf = (char *) realloc(sbuf, sbuf_size * sizeof(float));
                printf("Buffering input text file (%d B).\n", sbuf_size);
                break;
            }
        }
    }

    if (!USE_BUFFER(flags) || sbuf_size == 0) {
        /* Count lines and elements. */
        line = (char *) malloc((line_len = LINE_LENGTH) * sizeof(char));
        while (1) {
            line_size = parse_line(fid, line, line_end, line_len);
            if (line_size < 1) {
                if (line_size == 0) {
                    /* Empty line. */
                    exit_input_error(height + 1);
                } else {
                    /* End of file. */
                    break;
                }
            }
            /* Skip alpha. */
            buf = next_string_spec_space(line);
            ret |= *buf ^ ' ';
            buf++;


            while (!ret && (buf < line_end) && (*buf != 0)) {
                buf = next_string_spec_colon(buf);
                ret |= *(buf++) ^ ':';
                buf = next_string_spec_space(buf);
                if (*buf == '\n' || *buf == 0) {
                    break;
                }
                ret |= *buf ^ ' ';
                buf++;
            }

            if (ret != SUCCESS) {
                exit_input_error(height + 1);
            }

            i = strtoi_reverse(last_string_spec_colon(buf));

            if (*buf == '\n') {
                buf++;
            }

            if (i > width) {
                width = i;
            }
            height++;
        }
        /* Set width to fit in float4. */
        width = ALIGN_UP(width, alignment);

         if (fseek(fid, start_pos, SEEK_SET) != 0)
            exit(EXIT_FAILURE);

        if (ALLOC_ALPHAS_WC(flags)) {
            malloc_host_WC((void **) & alphas, sizeof(float) * height);
        } else {
            alphas = (float *) malloc(sizeof(float) * height);
        }
        if (ALLOC_VECTORS_WC(flags)) {
            malloc_host_WC((void **) & vectors, sizeof(float) * width * height);
        } else {
            vectors = (float *) malloc(sizeof(float) * width * height);
        }

        j = width * height;
        for (i = 0; i < j; i++) {
            vectors[i] = 0.0F;
        }

        i = 0;
        if (DO_TRANSPOSE_MATRIX(flags)) {
            while (1) {
                if (parse_line(fid, line, line_end, line_len) == -1) {
                    /* End of file. */
                    break;
                }
                /* Read alpha. */
                alphas[i] = strtof_fast(line, &buf);

                while (buf < line_end) {
                    /* Read index. */
                    buf = next_string_spec_colon(buf);
                    j = strtoi_reverse(buf - 1) - 1;
                    buf++;

                    /* Read value. */
                    vectors[j * height + i] = strtof_fast(buf, &buf);
                    if (*buf == '\n' || *buf == 0) {
                        break;
                    }
                    buf++;
                }
                i++;
            }
        } else {
            while (1) {
                if (parse_line(fid, line, line_end, line_len) == -1) {
                    /* End of file. */
                    break;
                }
                /* Read alpha. */
                alphas[i] = strtof_fast(line, &buf);

                while (buf < line_end) {
                    /* Read index. */
                    buf = next_string_spec_colon(buf);
                    j = strtoi_reverse(buf - 1) - 1;
                    buf++;

                    /* Read value. */
                    vectors[i * width + j] = strtof_fast(buf, &buf);
                    if (*buf == '\n' || *buf == 0) {
                        break;
                    }
                    buf++;
                }
                i++;
            }
        }

        /* Free memory. */
        free(line);
    } else {
        /* Count lines and elements. */
        buf = sbuf;
        while (*buf != 0) {
            /* Skip alpha. */
            buf = next_string_spec_space(buf);
            ret |= *buf ^ ' ';
            buf++;

            while (!ret && (*buf != '\n') && (*buf != 0)) {
                buf = next_string_spec_colon(buf);
                ret |= *(buf++) ^ ':';
                buf = next_string_spec_space(buf);
                if (*buf == '\n' || *buf == 0) {
                    break;
                }
                ret |= *buf ^ ' ';
                buf++;
            }

            if (ret != SUCCESS) {
                exit_input_error(height + 1);
            }

            i = strtoi_reverse(last_string_spec_colon(buf));

            if (*buf == '\n') {
                buf++;
            }

            if (i > width) {
                width = i;
            }
            height++;
        }
        /* Set width to fit in float4. */
        width = (width & 0xFFFFFFFC) + ((width & 3) ? 4 : 0);

        fseek(fid, start_pos, SEEK_SET);

        if (ALLOC_ALPHAS_WC(flags)) {
            malloc_host_WC((void **) & alphas, sizeof(float) * height);
        } else {
            alphas = (float *) malloc(sizeof(float) * height);
        }
        if (ALLOC_VECTORS_WC(flags)) {
            malloc_host_WC((void **) & vectors, sizeof(float) * width * height);
        } else {
            vectors = (float *) malloc(sizeof(float) * width * height);
        }

        j = width * height;
        for (i = 0; i < j; i++) {
            vectors[i] = 0.0F;
        }

        i = 0;
        buf = sbuf;
        if (DO_TRANSPOSE_MATRIX(flags)) {
            while (*buf != 0) {
                /* Read alpha. */
                alphas[i] = strtof_fast(buf, &buf);
                buf++;

                while ((*buf != '\n') && (*buf != 0)) {
                    /* Read index. */
                    buf = next_string_spec_colon(buf);
                    j = strtoi_reverse(buf - 1) - 1;
                    buf++;

                    /* Read value. */
                    vectors[j * height + i] = strtof_fast(buf, &buf);
                    if ((*buf == '\n') || (*buf == 0)) {
                        break;
                    }
                    buf++;
                }
                if (*buf == '\n') {
                    buf++;
                }
                i++;
            }
        } else {
            while (*buf != 0) {
                /* Read alpha. */
                alphas[i] = strtof_fast(buf, &buf);
                buf++;

                while ((*buf != '\n') && (*buf != 0)) {
                    /* Read index. */
                    buf = next_string_spec_colon(buf);
                    j = strtoi_reverse(buf - 1) - 1;
                    buf++;

                    /* Read value. */
                    vectors[i * width + j] = strtof_fast(buf, &buf);
                    if ((*buf == '\n') || (*buf == 0)) {
                        break;
                    }
                    buf++;
                }
                if (*buf == '\n') {
                    buf++;
                }
                i++;
            }
        }

        /* Free memory. */
        free(sbuf);
    }
}

//void cusvm_load_model2(char *filename, struct cuSVM_model * &model) {
//  char c,
//      *buf,
//      *tmp;
//  int values,
//      buf_len,
//      buf_idx;
//  float *alphas,
//      *vectors;
//  FILE *fid;
//
//  alphas = NULL;
//  vectors = NULL;
//  if ((fid = fopen(filename, "r")) == 0) {
//      cerr << "Error openning file " << filename << "!\n";
//      exit(EXIT_FAILURE);
//  }
//
//  model = (struct cuSVM_model *) malloc(sizeof(struct cuSVM_model));
//  if (model == NULL) {
//      cerr << "unable to allocate memory (model)!\n";
//      exit(EXIT_FAILURE);
//  }
//
//  buf = (char *) malloc((buf_len = BUFFER_LENGTH) * sizeof(char));
//  if (buf == NULL) {
//      cerr << "unable to allocate memory (buf)!\n";
//      exit(EXIT_FAILURE);
//  }
//
//  /* Read header. */
//  while (1) {
//      /* Read line. */
//      buf_idx = 0;
//      values = 0;
//      while ((c = fgetc(fid)) != EOL) {
//          if (c == ' ') {
//              c = 0;
//              values++;
//          }
//
//          if ((buf_idx + 1) == buf_len) {
//              buf = (char *) realloc(buf, (buf_len <<= 1) * sizeof(char));
//          }
//          buf[buf_idx++] = c;
//      }
//      buf[buf_idx++] = 0;
//
//      /* Validate line. */
//      if (values < 1) {
//          if (equals(buf, "SV")) {
//              break;
//          } else {
//              cerr << "Error: Unrecognized parameter \"" << buf << "\"!\n";
//              exit(EXIT_FAILURE);
//          }
//      } else {
//          if (values == 1) {
//              if (equals(buf, "svm_type")) {
//                  if (!equals(tmp = next_string(buf), "c_svc")) {
//                      cerr << "Error: Incompatible SVM type \"" << tmp << "\"!\n";
//                      exit(EXIT_FAILURE);
//                  }
//                  continue;
//              }
//              if (equals(buf, "kernel_type")) {
//                  tmp = next_string(buf);
//                  if (equals(tmp, "rbf")) {
//                      /* Correct. */
//                      continue;
//                  }
//                  cerr << "Error: Incompatible Kernel type \"" << tmp << "\"!\n";
//                  exit(EXIT_FAILURE);
//              }
//              if (equals(buf, "nr_class")) {
//                  if (!equals(tmp = next_string(buf), "2")) {
//                      cerr << "Error: Incompatible number of classes: " << tmp << "!\n";
//                      exit(EXIT_FAILURE);
//                  }
//                  continue;
//              }
//              if (equals(buf, "total_sv")) {
//                  /* Not needed. */
//                  continue;
//              }
//              if (equals(buf, "rho")) {
//                  model->beta = -(float) atof(next_string(buf));
//                  continue;
//              }
//              if (equals(buf, "gamma")) {
//                  model->lambda = (float) atof(next_string(buf));
//                  continue;
//              }
//              if (equals(buf, "degree")) {
//                  /* Unused value. */
//                  continue;
//              }
//              if (equals(buf, "coef0")) {
//                  /* Unused value. */
//                  continue;
//              }
//              cerr << "Error: Unrecognized parameter \"" << buf << "\"!\n";
//              exit(EXIT_FAILURE);
//          } else {
//              if (values == 2) {
//                  if (equals(buf, "label")) {
//                      /* Not neded. */
//                      continue;
//                  }
//                  if (equals(buf, "nr_sv")) {
//                      /* Not neded. */
//                      continue;
//                  }
//                  cerr << "Error: Unrecognized parameter \"" << buf << "\"!\n";
//                  exit(EXIT_FAILURE);
//              } else {
//                  cerr << "Error: Incompatible model file or header problem near by \"" << buf << "\"!\n";
//                  exit(EXIT_FAILURE);
//              }
//          }
//      }
//
//  }
//
//  /* Read support vectors. */
//  load_data_dense(fid, model->alphas, model->vectors, model->nof_vectors,
//      model->wof_vectors, LOAD_FLAG_ALL_RAM | LOAD_FLAG_TRANSPOSE);
//
//  /* Transfer data to GPU memory. */
//  //cusvm_create_model(model, alphas, vectors);
//
//  /* Release memory. */
//  free(buf);
//  //free_host(alphas);
//  //free_host(vectors);
//  fclose(fid);
//}

void malloc_host_WC(void** ptr, size_t size) {
#ifdef __NO_CUDA
        fprintf(stderr, "Error: __NO_CUDA deffined\n");
#else
    hipError_t err;

    int e = hipGetLastError();

    err = hipHostAlloc(ptr, size, hipHostMallocDefault);

    if (err != hipSuccess) {
        printf("Unable to allocate %d Bytes of Write Combined memory on host (%s)! Exiting...\n",
            size, hipGetErrorString(err));
        exit(err);
    }
#endif
}

void malloc_host_PINNED(void** ptr, size_t size) {
#ifdef __NO_CUDA
        fprintf(stderr, "Error: __NO_CUDA deffined\n");
#else
    hipError_t err;

    int e = hipGetLastError();

    err = hipMallocHost(ptr, size);

    if (err != hipSuccess) {
        printf("Unable to allocate %d Bytes of Pinned memory on host (%s)! Exiting...\n",
            size, hipGetErrorString(err));
        exit(err);
    }
#endif
}

void select_device(int device_id, unsigned int sm_limit, int is_max) {
    #ifndef __NO_CUDA
    void *dummy;
    unsigned int sm,
        dev_prop,
        cores;
    int i,
        devices,
        dev;
    hipError_t err;
    hipError_t cu_err;
    hipCtx_t dev_context = NULL;
    hipDeviceProp_t prop;

    cu_err = hipInit(0);
    if (cu_err != hipSuccess) {
        printf("Unable to initialize CUDA API! Exiting...\n");
        exit(cu_err);
    }
    cu_err = hipCtxGetCurrent(&dev_context);
    if (cu_err != hipSuccess) {
        printf("Unable to get current context of CUDA device! Exiting...\n");
        exit(cu_err);
    }

    /* If GPU not yet initialized. */
    if (dev_context == NULL) {
        /* Check for available GPUs. */
        err = hipGetDeviceCount(&devices);
        switch (err) {
        case hipSuccess:
            dev = -1;
            dev_prop = 0;

            for (i = 0; i < devices; i++) {
                /* Create context for the device (save o lot of time later). */
                cu_err = hipCtxCreate(&dev_context, 0, i);

                err = hipGetDeviceProperties(&prop, i);
                if (err != hipSuccess) {
                    printf("Unable to get device properties! Exiting...\n");
                    exit(err);
                }

                sm = prop.major * 10 + prop.minor;

                switch (prop.major) {
                case 1:
                    cores = prop.multiProcessorCount * 8;
                    break;
                case 2:
                    if (prop.minor == 0) {
                        cores = prop.multiProcessorCount * 32;
                    } else {
                        cores = prop.multiProcessorCount * 48;
                    }
                    break;
                case 3:
                    cores = prop.multiProcessorCount * 192;
                    break;
                case 5:
                    cores = prop.multiProcessorCount * 128;
                    break;
                default:
                    /* It is not possible to know the value of cores per multiprocesor
                       in the future architectures! */
                    cores = prop.multiProcessorCount * 128;
                    break;
                }

                if (i == device_id) { /* Use selected device. */
                    dev = i;
                    break;
                } else {
                    if (device_id < 0) { /* Choose the best. */
                        if (prop.clockRate * cores > dev_prop) {
                            if ((is_max && sm <= sm_limit)
                                || (!is_max && sm >= sm_limit)) {
                                dev_prop = prop.clockRate * cores;
                                dev = i;
                            }
                        }
                    }
                }
            }

            if (dev < 0) {
                printf("No compatible device found (supported only sm = %d)!\nExiting...\n", sm_limit);
                exit(EXIT_FAILURE);
            }


            err = hipGetDeviceProperties(&prop, dev);
            if (err != hipSuccess) {
                printf("Unable to get device properties! Exiting...\n");
                exit(err);
            }
            printf("\nSelected device:\n%s; sm = %d.%d\n\n", prop.name, prop.major, prop.minor);

            err = hipSetDevice(dev);
            if (err != hipSuccess) {
                printf("Unable to select CUDA device! Exiting...\n");
                exit(err);
            }
            cu_err = hipCtxCreate(&dev_context, 0, dev);
            if (cu_err != hipSuccess) {
                printf("Unable to create active context on CUDA device! Exiting...\n");
                exit(cu_err);
            }
            cu_err = hipCtxSetCurrent(dev_context);
            if (cu_err != hipSuccess) {
                printf("Unable to set active context of CUDA device! Exiting...\n");
                exit(cu_err);
            }

            hipMalloc(&dummy, 1);
            hipFree(dummy);

            break;
        case hipErrorNoDevice:
            printf("No CUDA capable device found!\nExiting...\n");
            exit(hipErrorNoDevice);
            break;
        case hipErrorInsufficientDriver:
            printf("Driver version too old!\nPlease install new version. Exiting...\n");
            exit(hipErrorInsufficientDriver);
            break;
        default:
            printf("Unknown Error [%X]!\nExiting...\n", err);
            exit(err);
            break;
        }
    }
#endif // not __NO_CUDA
}

int get_closest_label(int * labels, int labels_len, float alpha) {
      int i = 0;
    int index = 0;
    float distance = 0;
    float  tmp = 0;

    distance = INIT_DIST;
    for (i = 0; i < labels_len; i++) {
        if ((tmp = ABS(labels[i] - alpha)) < distance) {
            index = i;
            distance = tmp;
        }
    }

    return index;
}

/* Basic functions. */
int equals(char *str1, char *str2) {
    while (*str1 == *str2) {
        if (*str1 == 0) {
            return TRUE;
        }
        str1++;
        str2++;
    }

    return FALSE;
}

char *next_string(char *str) {
    char *out = str;

    while (*(out++)) {}

    return out;
}

char *next_string_spec(char *str) {
    char *out = str;

    while ((*out) != ':' && (*out) != ' ') {
        out++;
    }

    return out;
}

char *next_string_spec_space(char *str) {
    char *out = str;

    while ((*out != ' ') && (*out != 0) && (*out != '\n')) {
        out++;
    }

    return out;
}

char *next_string_spec_space_plus(char *str) {
    char *out = str;

    while ((*out != ' ') && (*out != '\n') && (*out != 0)) {
        out++;
    }

    return out;
}

char * next_string_spec_colon(char *str) {
    char * out = str;

    while ((*out != ':') && (*out != 0)) {
        out++;
    }

    return out;
}

char * last_string_spec_colon(char * str) {
    char * out = str;

    while (*out != ':') {
        --out;
    }

    return --out;
}

char * next_eol(char * str) {
    char *out = str;

    while ((*out != '\n') && (*out != 0)) {
        out++;
    }

    return out;
}

unsigned int strtoi_reverse(char * str) {
    unsigned int val = 0,
        mul = 1;

    do {
        val += mul * (*str - 0x30);
        mul *= 10;
    } while (*(--str) != ' ');

    return val;
}

/**
  * Used for loading alpha value and vector values.
  */
float strtof_fast(char * str, char ** end_ptr) {
    char phase = 0;
    int exp = 0;
    float val = 0.0F,
        mul = 0.1F,
        sign;

    if (*str == '-') {
        sign = -1.0F;
        str++;
    } else {
        if (*str == '+') {
            str++;
        }
        sign = 1.0F;
    }

    do {
        if (phase == 0) {
            if (*str == '.') {
                phase = 1;
            } else {
                val *= 10;
                val += (*str - 0x30);
            }
        } else {
            if (phase == 1) {
                if (*str != 'e' && *str != 'E') {
                    val += (*str - 0x30) * mul;
                    mul *= 0.1F;
                } else {
                    phase = 2; // load exponent
                    mul = 1.f;
                }
            } else {
                if (*str != '-') {
                    if (*str != '+') {
                        exp *= 10;
                        exp += (*str - 0x30);
                    }
                } else {
                    mul *= -1.f;
                }
            }
        }
        str++;
    } while (*str != ' ' && *str != '\n');

    *end_ptr = str;

    if (phase < 2) {
        return val * sign;
    } else {
        return val * sign * pow(10.f, (int) mul * exp);
    }
}

/* Returns length of line. */
long long parse_line(FILE * & fid, char * & line, char * & line_end, size_t & line_len) {
    char c;
    size_t idx;

    idx = 0;
    c = fgetc(fid);
    while (c != EOL && c != EOF) {
        if ((idx + 2) > line_len) {
            line = (char *) realloc(line, (line_len <<= 1) * sizeof(char));
        }
        line[idx++] = c;
        c = fgetc(fid);
    }
    line_end = line + idx;
    memset(line_end, 0, line_len - idx);
    if (c == EOF) {
        return -1;
    } else {
        return idx;
    }
}
