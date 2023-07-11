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

#ifndef _SVM_TEMPLATE
#define _SVM_TEMPLATE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#if defined WIN32 || defined WIN64
    #define MALLOC_ALIGNED(pointer, ptype, memsize, memalign) pointer=(ptype*)_aligned_malloc(memsize, memalign)
#else //LINUX
    #define MALLOC_ALIGNED(pointer, ptype, memsize, memalign) posix_memalign((void**)&pointer, memalign, memsize)
    #define _aligned_free free
#endif

struct svm_params {
	/* Training algorithm parameters: */
	double eps;	/* Stopping criteria. */
	double C;

	/* Kernel parameters: */
	int kernel_type;
	int degree;
	double gamma;
	double coef0;
	double nu;
	double p;

	/* Output parameters: */
	double rho; /* beta */
	unsigned int nsv_class1; /* Number of support vectors for the class with label -1. */
	unsigned int nsv_class2; /* Number of support vectors for the class with label +1. */
	int argc; 
	const char *argv[];
};

struct svm_trainingInfo {
	unsigned int numIters;
	float elTime1; //el time without init, deinit and CPU-GPU data transfer
	float elTime2; //total of pure kernels time
};

struct csr {
	unsigned int nnz;
	unsigned int numRows;
	unsigned int numCols;
	float *values;
	unsigned int *colInd;
	unsigned int *rowOffsets;
};

#define SUPPORTED_FORMAT_DENSE 1
#define SUPPORTED_FORMAT_CSR 2
#define SUPPORTED_FORMAT_BINARY 4
struct svm_memory_dataformat {
	unsigned int supported_types; //bit mask: &1 - dense, &2 - CSR, &4 - binary
	bool transposed;
	bool allocate_write_combined;
	bool allocate_pinned;
	bool labelsInFloat;
	unsigned int dimAlignment;
	unsigned int vectAlignment;
};

enum SVM_DATA_TYPE {UNKNOWN, DENSE, SPARSE, BINARY};
enum SVM_FILE_TYPE {LIBSVM_TXT, LASVM_BINARY};
enum SVM_MODEL_FILE_TYPE {M_LIBSVM_TXT};

void malloc_general(svm_memory_dataformat *format, void **x, size_t size);

class SvmData {
private:
protected:
	SVM_DATA_TYPE type;
	unsigned int numVects;
	unsigned int numVects_aligned;
	unsigned int dimVects;
	unsigned int dimVects_aligned;
	unsigned int numClasses;
	float *data_dense;
	int *class_labels;
	int *vector_labels;
	bool allocatedByCudaHost;
	bool transposed;
	bool labelsInFloat;
	bool invertLabels;
	csr *data_csr; //pointer to sparse data representation

	int load_libsvm_data_dense(FILE * &fid, SVM_DATA_TYPE data_type, svm_memory_dataformat *req_data_format);
	int load_libsvm_data_sparse(FILE * &fid, SVM_DATA_TYPE data_type, svm_memory_dataformat *req_data_format);
	int load_lasvm_binary_data(FILE * &fid, svm_memory_dataformat *req_data_format);
	int ConvertDataToDense();
	int ConvertDataToCSR();

public:
	SvmData();
	virtual ~SvmData();
	int Load(const char *filename, SVM_FILE_TYPE file_type, SVM_DATA_TYPE data_type, struct svm_memory_dataformat *req_data_format);
	virtual int Load(const char *filename, SVM_FILE_TYPE file_type, SVM_DATA_TYPE data_type) = 0; //requested data format settings need to be overloaded, than Load() is called 
	int Delete();
	unsigned int GetNumClasses() {return numClasses;}
	unsigned int GetNumVects() {return numVects;}
    unsigned int GetNumVectsAligned() {return numVects_aligned;}
	unsigned int GetDimVects() {return dimVects;}
    unsigned int GetDimVectsAligned() {return dimVects_aligned;}
	SVM_DATA_TYPE GetDataType() {return type;}
	float * GetDataDensePointer() {return data_dense;}
	csr * GetDataSparsePointer() {return data_csr;}
	float GetValue(unsigned int iVect, unsigned int iDim) {return transposed? data_dense[iDim * numVects_aligned + iVect] : data_dense[iVect * dimVects_aligned + iDim];}
	int * GetVectorLabelsPointer() {return vector_labels;}
	int * GetClassLabelsPointer() {return class_labels;}
	bool GetLabelsInFloat() {return labelsInFloat;}

	friend class SvmModel;
};

class SvmModel {
private:
protected:
	float *alphas;
	SvmData *data; //pointer to exiting external SvmData object - it is not own memory
	struct svm_params * params;
	bool allocatedByCudaHost;

	int StoreModel_LIBSVM_TXT(const char *model_file_name);
	int StoreModelGeneric(const char *model_file_name, SVM_MODEL_FILE_TYPE type);
	int LoadModel_LIBSVM_TXT(char *model_file_name, SVM_DATA_TYPE data_type, struct svm_memory_dataformat *req_data_format);
	int LoadModelGeneric(char *model_file_name, SVM_MODEL_FILE_TYPE type, SVM_DATA_TYPE data_type, struct svm_memory_dataformat *req_data_format);
	int CalculateSupperVectorCounts();

public:
	SvmModel();
	virtual ~SvmModel();
	int Delete();
	virtual int Train(SvmData *data, struct svm_params * params, struct svm_trainingInfo *trainingInfo) = 0;
	virtual int Predict(SvmData *testData, const char * file_out);
	virtual int StoreModel(const char *model_file_name, SVM_MODEL_FILE_TYPE type) = 0;
	virtual int LoadModel(char *model_file_name, SVM_MODEL_FILE_TYPE type, SVM_DATA_TYPE data_type);
};

class Utils {
public:
	static int StoreResults(char *filename, int *results, unsigned int numResults);
};

class GenericSvmData : public SvmData {
public:
	int Load(char *filename, SVM_FILE_TYPE file_type, SVM_DATA_TYPE data_type);
};

#endif //_SVM_TEMPLATE
