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

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cctype>

using namespace std;

#include "utils.h"
#include "cuSVM_wrapper.h"

CuSvmData::CuSvmData() {
    //select_device(-1, MAX_SUPPORTED_SM, 1);
}

int CuSvmData::Load(char *filename, SVM_FILE_TYPE file_type, SVM_DATA_TYPE data_type) {
    Delete();

    svm_memory_dataformat req_data_format;
    req_data_format.allocate_pinned = false;
    req_data_format.allocate_write_combined = false;
    req_data_format.dimAlignment = 1;
    req_data_format.vectAlignment = 1;
    req_data_format.transposed = true;
    req_data_format.labelsInFloat = true;
    req_data_format.supported_types = SUPPORTED_FORMAT_DENSE;

    SAFE_CALL(SvmData::Load(filename, file_type, data_type, &req_data_format));

    /* Read data from file. */
    //load_data_dense(fid, data->ref_results, data->vectors, data->nof_vectors,
    //  data->wof_vectors, LOAD_FLAG_ALL_RAM | LOAD_FLAG_TRANSPOSE | LOAD_FLAG_FILE_BUFFER);

    return SUCCESS;
}

//int CuSvmData::Delete() {
//  return SUCCESS;
//}

/////////////////////////////////////////////////////////////
//MODEL
CuSvmModel::CuSvmModel() {
    //select_device(-1, MAX_SUPPORTED_SM, 1);
    //model = NULL;
}

int CuSvmModel::Train(SvmData *_data, struct svm_params * _params, struct svm_trainingInfo *trainingInfo) {
    unsigned int numSVs, corr;
    float beta;

    data = (CuSvmData *) _data;
//  struct cuSVM_data *dataStruct;
    params = _params;

    if (data == NULL || params == NULL) {
        return FAILURE;
    }

    //dataStruct = data->GetDataStruct();
    numSVs = data->GetNumVects();
    if (params->gamma == 0) {
        params->gamma = 1.0 / data->GetDimVects();
    }

    MEM_SAFE_ALLOC(alphas, float, numSVs)
    corr = 0;
    printf("Starting Training \n");

    SVMTrain(alphas, &beta, (float *)data->GetVectorLabelsPointer(), data->GetDataDensePointer(),
        (float) params->C, (float) params->gamma, numSVs, data->GetDimVects(),
        (float) params->eps);

    params->rho = -beta;
    //printf("Train done. Calulate Vector counts \n");

    SAFE_CALL(CalculateSupperVectorCounts());

    //ExtractSupportVectors(dataStruct);

    return SUCCESS;
}

int CuSvmModel::StoreModel(char *model_file_name, SVM_MODEL_FILE_TYPE type) {
    return StoreModelGeneric(model_file_name, type);

    //unsigned int i,
    //  j,
    //  width,
    //  nofVects;
    //float alpha,
    //  value,
    //  alpha_mult;
    //FILE *fid;
    //struct cuSVM_data *dataStruct = ((CuSvmData *) data)->GetDataStruct();

    //FILE_SAFE_OPEN(fid, model_file_name, "w")

    //nofVects = model->nof_vectors;
    //width = model->wof_vectors;

    ///* Print header. */
    //fprintf(fid, "svm_type c_svc\nkernel_type rbf\ngamma %g\nnr_class 2\ntotal_sv %d\n",
    //  params->gamma, model->nof_vectors);

    ///* Print labels and counts. */
    //if (model->alphas[0] < 0.f) { // If the first label is negative
    //  fprintf(fid, "rho %g\nlabel -1 1\nnr_sv %d %d\nSV\n",
    //      -params->rho, params->nsv_class1, params->nsv_class2);
    //  alpha_mult = -1.f;
    //} else {
    //  fprintf(fid, "rho %g\nlabel 1 -1\nnr_sv %d %d\nSV\n",
    //      params->rho, params->nsv_class2, params->nsv_class1);
    //  alpha_mult = 1.f;
    //}

    ///* Print Support Vectors */
    //for (i = 0; i < nofVects; i++) {
    //  alpha = alpha_mult * model->alphas[i];
    //  fprintf(fid, "%g ", alpha);

    //  for (j = 0; j < width; j++) {
    //      value = model->vectors[i * width + j];
    //      if (value != 0.0F) {
    //          if (value == 1.0F) {
    //              fprintf(fid, "%d:1 ", j + 1);
    //          } else {
    //              fprintf(fid, "%d:%g ", j + 1, value);
    //          }
    //      }
    //  }

    //  fprintf(fid, "\n");
    //}

    //fclose(fid);

    //return SUCCESS;
}

//void CuSvmModel::ExtractSupportVectors(struct cuSVM_data * data) {
//  unsigned int i,
//      j,
//      k,
//      width,
//      height;
//
//  MEM_SAFE_ALLOC(model, struct cuSVM_model, 1)
//
//  params->nsv_class1 = 0;
//  params->nsv_class2 = 0;
//
//  height = data->nof_vectors;
//
//  /* Count support vectors for each class. */
//  for (i = 0; i < height; i++) {
//      if (alphas[i] != 0.0) {
//          alphas[i] *= data->ref_results[i];
//          if (alphas[i] < 0.0) {
//              params->nsv_class1++;   /* class label -1 */
//          } else {
//              params->nsv_class2++;  /* class label +1 */
//          }
//      }
//  }
//
//  width = data->wof_vectors;
//
//  model->nof_vectors = params->nsv_class1 + params->nsv_class2;
//  model->wof_vectors = width;
//
//  model->alphas = NULL;
//  model->vectors = NULL;
//
//  MEM_SAFE_ALLOC(model->alphas, float, model->nof_vectors)
//  MEM_SAFE_ALLOC(model->vectors, float, model->nof_vectors * width)
//
//  /* Store support vectors and alphas. */
//  for (i = 0, j = 0; i < height; i++) {
//      if (alphas[i] != 0.0) {
//          for (k = 0; k < width; k++) {
//              /* Transpose training data. */
//              model->vectors[j * width + k] = data->vectors[k * height + i];
//          }
//          model->alphas[j++] = alphas[i];
//      }
//  }
//}
