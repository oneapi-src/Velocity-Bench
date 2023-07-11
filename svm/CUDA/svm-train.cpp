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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef COMPILE_WITH_LIBSVM
#include "libSVM_wrapper.h"
#endif

#include "cuSVM_wrapper.h"

#include "utils.h"
#include "svm.h"
#include "my_stopwatch.h"

using namespace libsvm;

int g_cache_size = 0;
bool g_step_on_cpu = false;
int g_ws_size = 0;
std::string g_imp_spec_arg;

int help(int argc, char **argv, SvmData * &data, SvmModel * &model, struct svm_params * params, SVM_FILE_TYPE *file_type, SVM_DATA_TYPE *data_type, SVM_MODEL_FILE_TYPE *model_file_type);

int main(int argc, char **argv) {
    struct svm_params params;
    struct svm_trainingInfo trainingInfo;
    SVM_FILE_TYPE file_type = LIBSVM_TXT;
    SVM_DATA_TYPE data_type = UNKNOWN;
    SVM_MODEL_FILE_TYPE model_file_type = M_LIBSVM_TXT;
    MyStopWatch clAll, clLoad, clProc, clStore;
    SvmData *data;
    SvmModel *model;

    /* Check input arguments. */
    if(help(argc, argv, data, model, &params, &file_type, &data_type, &model_file_type) != SUCCESS) {
        return EXIT_SUCCESS;
    }

    clAll.start();

    /* Load data. */
    clLoad.start();
    if(data->Load(argv[1], file_type, data_type) != SUCCESS) {
        return EXIT_FAILURE;
    }
    clLoad.stop();
    //printf("Load Done \n");


    clProc.start();
    /* Train model. */
    if(model->Train(data, &params, &trainingInfo) != SUCCESS) {
        return EXIT_FAILURE;
    }
    clProc.stop();
    //printf("Training done \n");
    clStore.start();
    /* Predict values. */
    if(model->StoreModel(argv[2], model_file_type) != SUCCESS) {
        return EXIT_FAILURE;
    }

    /* Clean memory. */
    delete model;
    delete data;

    clStore.stop();

    clAll.stop();

    /* Print results. */
    printf("\nLoading    elapsed time : %0.4f s\n", clLoad.getTime());
    printf("Processing elapsed time : %0.4f s\n", clProc.getTime());
    printf("Storing    elapsed time : %0.4f s\n", clStore.getTime());
    printf("Total      elapsed time : %0.4f s\n", clAll.getTime());
    if ((params.rho < 0.06) && (params.rho > 0.05)) {
        printf("Result's are correct: %0.4f \n", params.rho);
    } else {
        printf("Result's are incorrect : %0.4f \n", params.rho);
        return EXIT_FAILURE;
    }


    return EXIT_SUCCESS;
}

int help(int argc, char **argv, SvmData * &data, SvmModel * &model, struct svm_params *params, SVM_FILE_TYPE *file_type, 
        SVM_DATA_TYPE *data_type, SVM_MODEL_FILE_TYPE *model_file_type) {
    int i, imp;

    params->kernel_type = RBF;/*k*/
    params->C = 1;                /*c, cost value*/
    params->eps = 1e-3;            /*e, stopping criteria */
    params->degree = 3;            /*d*/
    params->gamma = 0.5;            /*g*/
    params->coef0 = 0;            /*f*/
    params->nu = 0.5;            /*n*/
    params->p = 0.1;            /*p, regression parameter epsilon*/
    imp = 3;                    /*i*/
    
    params->C = 1;                /*c, cost value*/
    params->eps = 1e-3;            /*e, stopping criteria */
    params->degree = 3;            /*d*/
    params->gamma = 0.5;            /*g*/
    params->coef0 = 0;            /*f*/
    params->nu = 0.5;            /*n*/
    params->p = 0.1;            /*p, regression parameter epsilon*/
    imp = 3;                    /*i*/
   
    printf("Using cuSVM (Carpenter)...\n\n");
    data = new CuSvmData;
    model = new CuSvmModel;
    return SUCCESS;
}

/* Print help information. */
void print_help() {
    printf("SVM-train benchmark\n"
        "Use: SVMbenchmark.exe <data> <model> [-<attr1> <value1> ...]\n\n"
        " data   File containing data to be used for training.\n"
        " model  Where to store model in LibSVM format.\n"
        " attrx  Attribute to set.\n"
        " valuex Value of attribute x.\n\n"
        " Attributes:\n"
        "  k  SVM kernel. Corresponding values:\n"
        "         l   Linear\n"
        "         p   Polynomial\n"
        "         r   RBF\n"
        "         s   Sigmoid\n"
        "  c  Training parameter C.\n"
        "  e  Stopping criteria.\n"
        "  n  Training parameter nu.\n"
        "  p  Training parameter p.\n"
        "  d  Kernel parameter degree.\n"
        "  g  Kernel parameter gamma.\n"
        "  f  Kernel parameter coef0.\n"
        "  t  Force data type. Values are:\n"
        "         d   Dense data\n"
        "         s   Sparse data\n"
        "  i  Select implementation to use. Corresponding values:\n"
#ifdef COMPILE_WITH_LIBSVM
        "         1   LibSVM (default)\n"
#endif

        "         3   CuSVM (Carpenter)\n"


        "  b  Read input data in binary format (lasvm dense or sparse format)\n"
        "  w  Working set size (currently only for implementation 16)\n"
        "  r  Cache size in MB\n"
        "  x  Implementation specific parameter:\n"
        "     OHD-SVM: Two numbers separated by comma specifying EllR-T\n"
        "              storage format dimensions: sliceSize,threadsPerRow\n"
        "              Value 0,0 means automatic parameter selection.\n"
        "              Specifying -x implies EllR-T, do not specify -x\n"
        "              to use JDS.\n"
        );
}
