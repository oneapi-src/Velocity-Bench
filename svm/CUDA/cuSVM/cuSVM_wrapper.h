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

#ifndef _CUSVM_WRAPPER_H_
#define _CUSVM_WRAPPER_H_

#include "svm_template.h"

#define MAX_SUPPORTED_SM 70 /* max CU level with correct # corres calculation */

/* Train */
/**
  * mexalpha     Output alpha values.
  * beta         Calculated SVM threshold (rho).
  * y            Input labels.
  * x            input matrix of training vectors (transposed).
  * C            Training parameter C.
  * kernelwidth  The parameter gamma.
  * eps          Thre regression parameter epsilon.
  * m            Number of rows (# training vectors).
  * n            Number of columns (width).
  * StoppingCrit Stopping criteria.
  */
//extern "C" void SVRTrain(float *mexalpha,float* beta,float*y,float *x ,float C, float _kernelwidth, float eps, int m, int n, float StoppingCrit);

extern "C" void SVMTrain(float *mexalpha,float* beta,float*y,float *x ,float C, float kernelwidth, int m, int n, float StoppingCrit);


/*paddedm = (m & 0xFFFFFFE0) + ((m & 0x1F) ? 0x20 : 0);*/
/*  int ceiled_pm_ni = (paddedm + NecIterations - 1) / NecIterations;
    int RowsPerIter = (ceiled_pm_ni & 0xFFFFFFE0) + ((ceiled_pm_ni & 0x1F) ? 0x20 : 0);*/

//struct cuSVM_model {
//  unsigned int nof_vectors;
//  unsigned int wof_vectors;
//  float lambda;
//  float beta;
//  float *alphas;
//  float *vectors;
//};

class SvmData;
class SvmModel;

class CuSvmData : public SvmData {
private:
protected:
public:
    CuSvmData();
    //~CuSvmData();

    int Load(char *filename, SVM_FILE_TYPE file_type, SVM_DATA_TYPE data_type);
    //int Delete();
};

class CuSvmModel : public SvmModel {
private:
    //float * alphas;
    //void ExtractSupportVectors(struct cuSVM_data * dataStruct);
protected:
    //struct cuSVM_model * model;
    //struct svm_params * params;
public:
    CuSvmModel();
    //~CuSvmModel();

    int Train(SvmData *data, struct svm_params * params, struct svm_trainingInfo *trainingInfo);
    int StoreModel(char *model_file_name, SVM_MODEL_FILE_TYPE type);
    //int Delete();
};

#endif //_CUSVM_WRAPPER_H_
