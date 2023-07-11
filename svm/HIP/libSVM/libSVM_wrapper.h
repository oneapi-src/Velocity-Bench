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

#ifndef _LIBSVM_WRAPPER
#define _LIBSVM_WRAPPER

#include "svm_template.h"
#include "svm.h"

using namespace libsvm;

class SvmData;
class SvmModel;

class LibSvmData : public SvmData {
private:
protected:
	struct svm_problem *prob;

	void ConvertFromDenseData();
	void ConvertFromCSRData();
public:
	LibSvmData();
	~LibSvmData();

	int Load(char *filename, SVM_FILE_TYPE file_type, SVM_DATA_TYPE data_type);
	int Delete();
	struct svm_problem * GetProb();
};

class LibSvmModel : public SvmModel {
private:
	float * alphas;
	void ConvertParameters(struct svm_params * par_src, struct svm_parameter * &par_dst);

protected:
	struct svm_params * params;
	struct svm_model * model;

public:
	LibSvmModel();
	~LibSvmModel();

	int Train(SvmData * data, struct svm_params * params, struct svm_trainingInfo *trainingInfo);
	int StoreModel(char * model_file_name, SVM_MODEL_FILE_TYPE type);
	int Delete();
};

#endif //_LIBSVM_WRAPPER
