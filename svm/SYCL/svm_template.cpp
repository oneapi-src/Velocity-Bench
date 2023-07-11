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

#include "svm_template.h"
#include "utils.h"
#ifndef __NO_CUDA
//#include "cuda_runtime_api.h"
#endif
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <stdexcept>

using namespace libsvm;

///////////////////////////////////////////////////////////////
// Utils

static std::vector<std::string> & split(const std::string & s, char delim, std::vector<std::string> & tokens) {
    std::istringstream ss(s);
    std::string token;
    while (std::getline(ss, token, delim)) {
        tokens.push_back(token);
    }
    return tokens;
}

static std::vector<std::string> split(const std::string & s, char delim) {
    std::vector<std::string> tokens;
    split(s, delim, tokens);
    return tokens;
}

int Utils::StoreResults(char *filename, int *results, unsigned int numResults) {
	unsigned int i;
	FILE *fid;

	if ((fid = fopen(filename, "w")) == NULL) {
		printf("Cannot open file %s\n", filename);
		return -1;
	}
	for (i = 0; i < numResults; i++) {
		fprintf(fid, "%d\n", results[i]); 
	}
	fclose(fid);
	return 0;
}

///////////////////////////////////////////////////////////////
// SvmData

SvmData::SvmData() {
	type = DENSE;
	numClasses = 0;
	numVects = 0;
	dimVects = 0;
	numVects_aligned = 0;
	dimVects_aligned = 0;
	data_dense = NULL;
	data_csr = NULL;
	class_labels = NULL;
	vector_labels = NULL;
	allocatedByCudaHost = false;
	transposed = false;
	labelsInFloat = false;
	invertLabels = false;
}

SvmData::~SvmData() {
	Delete();
}

int SvmData::Delete() {
	type = DENSE;
	numClasses = 0;
	numVects = 0;
	dimVects = 0;
	numVects_aligned = 0;
	dimVects_aligned = 0;
	//if(allocatedByCudaHost) {

/*	
#ifdef __NO_CUDA
		fprintf(stderr, "Error: __NO_CUDA deffined\n");
#else
		if(data_dense != NULL) cudaFreeHost(data_dense);
		if(class_labels != NULL) cudaFreeHost(class_labels);
		if(vector_labels != NULL) cudaFreeHost(vector_labels);
		if(data_csr != NULL) {
			cudaFreeHost(data_csr->values);
			cudaFreeHost(data_csr->colInd);
			cudaFreeHost(data_csr->rowOffsets);
			free(data_csr);
		}
#endif */
//	} else {
		if(data_dense != NULL) free(data_dense);
		if(class_labels != NULL) free(class_labels);
		if(vector_labels != NULL) free(vector_labels);
		if(data_csr != NULL) {
			free(data_csr->values);
			free(data_csr->colInd);
			free(data_csr->rowOffsets);
			free(data_csr);
		}
//	}
	data_dense = NULL;
	data_csr = NULL;
	class_labels = NULL;
	vector_labels = NULL;
	allocatedByCudaHost = false;
	transposed = false;
	labelsInFloat = false;
	invertLabels = false;
	return SUCCESS;
}

int SvmData::Load(const char *filename, SVM_FILE_TYPE file_type, SVM_DATA_TYPE data_type, struct svm_memory_dataformat *req_data_format) {
	FILE *fid;

	Delete();

	float * labels_tmp = NULL;

	if(file_type == LASVM_BINARY) {
		FILE_SAFE_OPEN(fid, filename, "rb")
	} else {
		FILE_SAFE_OPEN(fid, filename, "r")
	}

		/* Read data from file. */
		switch(file_type) {
		case LIBSVM_TXT:
			if((req_data_format->supported_types & SUPPORTED_FORMAT_CSR) && (data_type == UNKNOWN || data_type == SPARSE)) {
				load_libsvm_data_sparse(fid, data_type, req_data_format);
			} else {
				load_libsvm_data_dense(fid, data_type, req_data_format);
			}
			break;
		case LASVM_BINARY:
			load_lasvm_binary_data(fid, req_data_format);
			break;
		default:
			printf("Format of the data file not supported or the setting is wrong\n");
			return FAILURE;
	}

	/* Close streams. */
	fclose(fid);

	//if(data_type == SPARSE && (req_data_format->supported_types & SPARSE) == 0) ConvertDataToDense(req_data_format);
	//if(data_type == DENSE && (req_data_format->supported_types & DENSE) == 0) ConvertDataToCSR(req_data_format);

    if ((type == SPARSE && !(req_data_format->supported_types & SUPPORTED_FORMAT_CSR)) || (type == DENSE && !(req_data_format->supported_types & SUPPORTED_FORMAT_DENSE)))
    {
        printf("Unsupported data type\n");
        return FAILURE;
    }

	return SUCCESS;
} //SvmData::Load()

int SvmData::ConvertDataToDense() {

	return 0;
} //SvmData::ConvertDataToDense

int SvmData::ConvertDataToCSR() {

	return 0;
} //SvmData::ConvertDataToCSR

int SvmData::load_libsvm_data_dense(FILE * &fid, SVM_DATA_TYPE data_type, svm_memory_dataformat *req_data_format) {

	Delete();
	bool useBuffer = true;

	char *buf,
		*sbuf = NULL,
		*sbuf_tmp,
		*line = NULL,
		*line_end;
	size_t i,
		it = 0,
		j =0,
		start_pos,
		sbuf_size = 64<<20, //intial size 64MB
		read_chars,
		new_len,
		line_len;
	long long line_size;
	int ret;

	if ((start_pos = ftell(fid)) == EOF) {
		printf("File is not openned!\n");
		exit(EXIT_FAILURE);
	}

	ret = SUCCESS;

	if(req_data_format->labelsInFloat && sizeof(int) != sizeof(float)) REPORT_ERROR("4byte-int platform assumed");

	if (useBuffer) {
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
					if (fseek(fid, start_pos, SEEK_SET) !=0)
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
				printf("Buffering input text file (%ld B).\n", sbuf_size);
				break;
			}
		}
	}

	if (!useBuffer || sbuf_size == 0) {
		/* Count lines and elements. */
		line = (char *) malloc((line_len = LINE_LENGTH) * sizeof(char));
		while (1) {
			line_size = parse_line(fid, line, line_end, line_len);
			if (line_size < 1) {
				if (line_size == 0) {
					/* Empty line. */
					exit_input_error(numVects + 1);
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
				buf = std::min(next_string_spec_colon(buf), next_eol(buf));
				if ((*buf == '\n') || (*buf == 0)) {
					break;
				}
				ret |= *(buf++) ^ ':';
				buf = next_string_spec_space(buf);
				if (*buf == '\n' || *buf == 0) {
					break;
				}
				ret |= *buf ^ ' ';
				buf++;
			}

			if (ret != SUCCESS) {
				exit_input_error(numVects + 1);
			}

			i = strtoi_reverse(last_string_spec_colon(buf));

			if (*buf == '\n') {
				buf++;
			}

			if (i > dimVects) {
				dimVects = i;
			}
			numVects++;
		}
		if (fseek(fid, start_pos, SEEK_SET) != 0)
			exit(EXIT_FAILURE);

		numVects_aligned = ALIGN_UP(numVects, req_data_format->vectAlignment);
		dimVects_aligned = ALIGN_UP(dimVects, req_data_format->dimAlignment);
		malloc_general(req_data_format, (void **) & vector_labels, sizeof(float) * numVects_aligned);
		malloc_general(req_data_format, (void **) & data_dense, sizeof(float) * dimVects_aligned * numVects_aligned);
		memset(data_dense, 0, sizeof(float) * dimVects_aligned * numVects_aligned);

		i = 0;
		while (1) {
			if (parse_line(fid, line, line_end, line_len) == -1) {
				/* End of file. */
				break;
			}
			/* Read alpha. */
			vector_labels[i] = strtol(line, &buf, 10);

			while (buf < line_end) {
				/* Read index. */
				buf = std::min(next_string_spec_colon(buf), next_eol(buf));
				if ((*buf == '\n') || (*buf == 0)) {
					break;
				}
				j = strtoi_reverse(buf - 1) - 1;
				buf++;

				/* Read value. */
				if (req_data_format->transposed) data_dense[j * numVects_aligned + i] = strtof_fast(buf, &buf);
				else data_dense[i * dimVects_aligned + j] = strtof_fast(buf, &buf);

				if (*buf == '\n' || *buf == 0) {
					break;
				}
				buf++;
			}
			i++;
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
				buf = std::min(next_string_spec_colon(buf), next_eol(buf));
				if (*buf == '\n' || *buf == 0) {
					break;
				}
				ret |= *(buf++) ^ ':';
				buf = next_string_spec_space(buf);
				if (*buf == '\n' || *buf == 0) {
					break;
				}
				ret |= *buf ^ ' ';
				buf++;
			}

			if (ret != SUCCESS) {
				exit_input_error(numVects + 1);
			}

			i = strtoi_reverse(last_string_spec_colon(buf));

			if (*buf == '\n') {
				buf++;
			}

			if (i > dimVects) {
				dimVects = i;
			}
			numVects++;
		}

		if (fseek(fid, start_pos, SEEK_SET) != 0)
			exit(EXIT_FAILURE);

		numVects_aligned = ALIGN_UP(numVects, req_data_format->vectAlignment);
		dimVects_aligned = ALIGN_UP(dimVects, req_data_format->dimAlignment);
		malloc_general(req_data_format, (void **) & vector_labels, sizeof(float) * numVects_aligned);
		malloc_general(req_data_format, (void **) & data_dense, sizeof(float) * dimVects_aligned * numVects_aligned);
		memset(data_dense, 0, sizeof(float) * dimVects_aligned * numVects_aligned);

		i = 0;
		buf = sbuf;
		while (*buf != 0) {
			/* Read alpha. */
			vector_labels[i] = strtol(buf, &buf, 10);
			buf++;

			while ((*buf != '\n') && (*buf != 0)) {
				/* Read index. */
				buf = std::min(next_string_spec_colon(buf), next_eol(buf));
				if ((*buf == '\n') || (*buf == 0)) {
					break;
				}
				j = strtoi_reverse(buf - 1) - 1;
				buf++;

				/* Read value. */
				if (req_data_format->transposed) data_dense[j * numVects_aligned + i] = strtof_fast(buf, &buf);
				else data_dense[i * dimVects_aligned + j] = strtof_fast(buf, &buf);

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

		/* Free memory. */
		free(sbuf);
	}

	allocatedByCudaHost = req_data_format->allocate_pinned || req_data_format->allocate_write_combined;
	this->type = DENSE;
	transposed = req_data_format->transposed;

	//make class labels
	int max_idx = -2;
	for(unsigned int i=0; i < numVects; i++) {
		if(max_idx < vector_labels[i]) max_idx = vector_labels[i];
	}
	class_labels = (int *) malloc(sizeof(int) * (max_idx + 2));
	for(int i=0; i < max_idx + 2; i++) class_labels[i] = -2;
	for(unsigned int i=0; i < numVects; i++) {
		int ci = vector_labels[i];
		if(ci < -1) REPORT_ERROR("Class index lower than -1");
		if(class_labels[ci+1] == -2) class_labels[ci+1] = ci;
	}
	numClasses = 0;
	for(int i=0; i < max_idx + 2; i++) {
		if(class_labels[i] != -2) {
			class_labels[numClasses] = class_labels[i];
			numClasses++;
		}
	}

	//store if the first label is negative: to LibSVM compatibility of stored model:
	invertLabels = !(vector_labels[0] == 1);

	//if float labels required:
	if(req_data_format->labelsInFloat) {
		labelsInFloat = true;
		float *p = (float *)vector_labels;
		for(unsigned int i=0; i < numVects; i++) p[i] = (float) vector_labels[i];
	}

	return 0;
} //SvmData::load_libsvm_data_dense

int SvmData::load_libsvm_data_sparse(FILE * &fid, SVM_DATA_TYPE data_type, svm_memory_dataformat *req_data_format) {

	Delete();
	bool useBuffer = true;

	char *buf,
		*sbuf = NULL,
		*sbuf_tmp,
		*line = NULL,
		*line_end;
	size_t i,
		it = 0,
		j =0,
		start_pos,
		sbuf_size = 64<<20, //intial size 64MB
		read_chars,
		new_len,
		line_len;
	long long line_size;
	int ret;

	if ((start_pos = ftell(fid)) == EOF) {
		printf("File is not openned!\n");
		exit(EXIT_FAILURE);
	}

	ret = SUCCESS;

	if(req_data_format->labelsInFloat && sizeof(int) != sizeof(float)) REPORT_ERROR("4byte-int platform assumed");

	this->data_csr = new csr;
	data_csr->nnz = 0;
	data_csr->numCols = 0;
	data_csr->numRows = 0;

	if (useBuffer) {
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
					if (fseek(fid, start_pos, SEEK_SET) !=  0)
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

	if (!useBuffer || sbuf_size == 0) {
		/* Count lines and elements. */
		line = (char *) malloc((line_len = LINE_LENGTH) * sizeof(char));
		while (1) {
			line_size = parse_line(fid, line, line_end, line_len);
			if (line_size < 1) {
				if (line_size == 0) {
					/* Empty line. */
					exit_input_error(numVects + 1);
				} else {
					/* End of file. */
					break;
				}
			}
			buf = line;

			//sparse data: simply count ':' and \n chars
			while(*buf != '\n' && *buf != 0 && buf < line_end) {
				if(*buf == ':') data_csr->nnz++;
				buf++;
			}
			numVects++;

			if (*buf == '\n') {
				buf++;
			}
			numVects++;
		}
		if (fseek(fid, start_pos, SEEK_SET) != 0)
			exit(EXIT_FAILURE);

		data_csr->numRows = numVects;
		numVects_aligned = ALIGN_UP(numVects, req_data_format->vectAlignment);

		if (req_data_format->allocate_write_combined) {
			malloc_host_WC((void **) &vector_labels, sizeof(float) * numVects_aligned);
			malloc_host_WC((void **) data_csr->values, sizeof(float) * data_csr->nnz);
			malloc_host_WC((void **) data_csr->colInd, sizeof(int) * data_csr->nnz);
			malloc_host_WC((void **) data_csr->rowOffsets, sizeof(int) * (numVects_aligned+1));
		} else {
			if (req_data_format->allocate_pinned) {
				malloc_host_PINNED((void **) & vector_labels, sizeof(float) * numVects_aligned);
				malloc_host_PINNED((void **) data_csr->values, sizeof(float) * data_csr->nnz);
				malloc_host_PINNED((void **) data_csr->colInd, sizeof(int) * data_csr->nnz);
				malloc_host_PINNED((void **) data_csr->rowOffsets, sizeof(int) * (numVects_aligned+1));
			} else {
				vector_labels = (int *) malloc(sizeof(float) * numVects_aligned);
				data_csr->values = (float *) malloc(sizeof(float) * data_csr->nnz);
				data_csr->colInd = (unsigned int *) malloc(sizeof(int) * data_csr->nnz);
				data_csr->rowOffsets = (unsigned int *) malloc(sizeof(int) * (numVects_aligned+1));
			}
		}

		i = 0;
		data_csr->rowOffsets[0] = 0;
		dimVects = 0;
		unsigned int offset = 0;
		while (1) {
			if (parse_line(fid, line, line_end, line_len) == -1) {
				/* End of file. */
				break;
			}
			/* Read alpha. */
			vector_labels[i] = strtol(line, &buf, 10);

			while (buf < line_end) {
				/* Read index. */
				buf = std::min(next_string_spec_colon(buf), next_eol(buf));
				if ((*buf == '\n') || (*buf == 0)) {
					break;
				}
				j = strtoi_reverse(buf - 1) - 1;
				buf++;

				/* Read value. */
				data_csr->values[offset] = strtof_fast(buf, &buf);
				data_csr->colInd[offset] = j;
				offset++;

				if (*buf == '\n' || *buf == 0) {
					break;
				}
				buf++;
			}
			if(dimVects <= j) dimVects = j+1;//maximum observed dimension
			i++;
			data_csr->rowOffsets[i] = offset;
		}
		i++;
		for(int ii=i; ii < numVects_aligned+1; ii++) data_csr->rowOffsets[ii] = offset; //fill the padded area

		dimVects_aligned = ALIGN_UP(dimVects, req_data_format->dimAlignment);
		data_csr->numCols = dimVects;
		/* Free memory. */
		free(line);
	} else {
		/* Count lines and elements. */
		buf = sbuf;
		while (*buf != 0) {
			//sparse data: simply count ':' and \n chars
			while(*buf != '\n' && *buf != 0) {
				if(*buf == ':') data_csr->nnz++;
				buf++;
			}
			numVects++;
			buf++;
		}
		data_csr->numRows = numVects;
		numVects_aligned = ALIGN_UP(numVects, req_data_format->vectAlignment);

		if (fseek(fid, start_pos, SEEK_SET) != 0)
			exit(EXIT_FAILURE);
		
		if (req_data_format->allocate_write_combined) {
			malloc_host_WC((void **) &vector_labels, sizeof(float) * numVects_aligned);
			malloc_host_WC((void **) data_csr->values, sizeof(float) * data_csr->nnz);
			malloc_host_WC((void **) data_csr->colInd, sizeof(int) * data_csr->nnz);
			malloc_host_WC((void **) data_csr->rowOffsets, sizeof(int) * (numVects_aligned+1));
		} else {
			if (req_data_format->allocate_pinned) {
				malloc_host_PINNED((void **) & vector_labels, sizeof(float) * numVects_aligned);
				malloc_host_PINNED((void **) data_csr->values, sizeof(float) * data_csr->nnz);
				malloc_host_PINNED((void **) data_csr->colInd, sizeof(int) * data_csr->nnz);
				malloc_host_PINNED((void **) data_csr->rowOffsets, sizeof(int) * (numVects_aligned+1));
			} else {
				vector_labels = (int *) malloc(sizeof(float) * numVects_aligned);
				data_csr->values = (float *) malloc(sizeof(float) * data_csr->nnz);
				data_csr->colInd = (unsigned int *) malloc(sizeof(int) * data_csr->nnz);
				data_csr->rowOffsets = (unsigned int *) malloc(sizeof(int) * (numVects_aligned+1));
			}
		}

		i = 0;
		data_csr->rowOffsets[0] = 0;
		unsigned int offset = 0;
		dimVects = 0; //maximum observed dimension
		buf = sbuf;
		while (*buf != 0) {
			/* Read alpha. */
			vector_labels[i] = strtol(buf, &buf, 10);
			buf++;

			while ((*buf != '\n') && (*buf != 0)) {
				/* Read index. */
				buf = std::min(next_string_spec_colon(buf), next_eol(buf));
				if ((*buf == '\n') || (*buf == 0)) {
					break;
				}
				j = strtoi_reverse(buf - 1) - 1;
				buf++;

				/* Read value. */
				data_csr->values[offset] = strtof_fast(buf, &buf);
				data_csr->colInd[offset] = j;
				offset++;

				if ((*buf == '\n') || (*buf == 0)) {
					break;
				}
				buf++;
			}
			if (*buf == '\n') {
				buf++;
			}
			if(dimVects <= j) dimVects = j+1;//maximum observed dimension
			i++;
			data_csr->rowOffsets[i] = offset;
		}
		i++;
		for(int ii=i; ii < numVects_aligned+1; ii++) data_csr->rowOffsets[ii] = offset; //fill the padded area

		dimVects_aligned = ALIGN_UP(dimVects, req_data_format->dimAlignment);
		data_csr->numCols = dimVects;
		/* Free memory. */
		free(sbuf);
	}
	printf("NNZ: %d\n%% NNZ: %.3lf\nAvg. NNZ per row: %.3lf\n", data_csr->nnz, 100. * data_csr->nnz / (numVects * dimVects), data_csr->nnz / (double)numVects);

	allocatedByCudaHost = req_data_format->allocate_pinned || req_data_format->allocate_write_combined;
	this->type = SPARSE;
	if(req_data_format->transposed) REPORT_WARNING("Warning: CSR data format cannot be transposed")
	transposed = false;

	//make class labels
	int max_idx = -2;
	for(unsigned int i=0; i < numVects; i++) {
		if(max_idx < vector_labels[i]) max_idx = vector_labels[i];
	}
	class_labels = (int *) malloc(sizeof(int) * (max_idx + 2));
	for(int i=0; i < max_idx + 2; i++) class_labels[i] = -2;
	for(unsigned int i=0; i < numVects; i++) {
		int ci = vector_labels[i];
		if(ci < -1) REPORT_ERROR("Class index lower than -1");
		if(class_labels[ci+1] == -2) class_labels[ci+1] = ci;
	}
	numClasses = 0;
	for(int i=0; i < max_idx + 2; i++) {
		if(class_labels[i] != -2) {
			class_labels[numClasses] = class_labels[i];
			numClasses++;
		}
	}

	//store if the first label is negative: to LibSVM compatibility of stored model:
	invertLabels = !(vector_labels[0] == 1);

	//if float labels required:
	if(req_data_format->labelsInFloat) {
		labelsInFloat = true;
		float *p = (float *)vector_labels;
		for(unsigned int i=0; i < numVects; i++) p[i] = (float) vector_labels[i];
	}

	return 0;
} //load_libsvm_data_sparse

int SvmData::load_lasvm_binary_data(FILE * &fid, svm_memory_dataformat *req_data_format) {
	Delete();

	unsigned int buf[2];
	if (fread(&buf, sizeof(int), 2, fid) != (sizeof(int) * 2))
		exit(EXIT_FAILURE);

	this->numVects = buf[0];
	if (this->numVects == 0)
		exit(EXIT_FAILURE);
	
	
	
	numVects_aligned = ALIGN_UP(numVects, req_data_format->vectAlignment);
	malloc_general(req_data_format, (void**)&vector_labels, sizeof(float) * numVects_aligned);
	dimVects = buf[1];

	if(dimVects > 0) { //dense
		dimVects_aligned = ALIGN_UP(dimVects, req_data_format->dimAlignment);
		malloc_general(req_data_format, (void**)&data_dense, sizeof(float) * numVects_aligned * dimVects_aligned);
		memset(data_dense, 0, sizeof(float) * numVects_aligned * dimVects_aligned);
		for(unsigned int i = 0; i < numVects; i++) {
			int label = 0;
			if (fread(&label, sizeof(int), 1, fid) != sizeof(int))
				exit(EXIT_FAILURE);

			vector_labels[i] = label;
			if(req_data_format->transposed) {
				float value;
				for(unsigned int k = 0; k < dimVects; k++) {
					if (fread(&value, sizeof(float), 1, fid) != sizeof(float))
						exit(EXIT_FAILURE);

					data_dense[k*numVects_aligned + i] = value;
				}
			} else {
				if (fread(data_dense + i*dimVects_aligned, sizeof(float), dimVects, fid) != (sizeof(float) * dimVects))
					exit(EXIT_FAILURE);

			}
		}
		this->type = DENSE;
		transposed = req_data_format->transposed;
	} else { //sparse
		this->type = SPARSE;
		if(req_data_format->transposed) REPORT_WARNING("Warning: CSR data format cannot be transposed")
		transposed = false;
		data_csr = new csr;
		data_csr->numRows = numVects;
		malloc_general(req_data_format, (void**)&(data_csr->rowOffsets), sizeof(int) * (numVects_aligned + 1));
		data_csr->nnz = 0;
		data_csr->rowOffsets[0] = 0;
		for(unsigned int i = 0; i < numVects; i++) {
			int label = 0;
			if (fread(&label, sizeof(int), 1, fid) != sizeof(int))
				exit(EXIT_FAILURE);
			vector_labels[i] = label;
			//count nnz:
			unsigned int count = 0;
			if (fread(&count, sizeof(int), 1, fid) != sizeof(int))
				exit(EXIT_FAILURE);

			data_csr->nnz += count;
			data_csr->rowOffsets[i+1] = data_csr->nnz;
			if (fseek(fid, count * (sizeof(int) + sizeof(float)), SEEK_CUR) != 0)
				exit(EXIT_FAILURE);

			//float foo_f;
			//int foo_i[14];
			//skip values in the first pass, fseek doesn work properly
			//for(unsigned int foo=0; foo < count; foo++) {
			//	fread(foo_i, sizeof(int), 14, fid);
			//}
			//for(unsigned int foo=0; foo < count; foo++) {
			//	fread(&foo_f, sizeof(float), 1, fid);
			//}
		}
		malloc_general(req_data_format, (void**)&(data_csr->values), sizeof(float) * data_csr->nnz);
		malloc_general(req_data_format, (void**)&(data_csr->colInd), sizeof(int) * data_csr->nnz);
		
		if (fseek(fid, 2*sizeof(int), SEEK_SET) != 0)
			exit(EXIT_FAILURE);

		for(unsigned int i = 0; i < numVects; i++) {
			int label = 0;
			if (fread(&label, sizeof(int), 1, fid) != sizeof(int))
				exit(EXIT_FAILURE);
			
			unsigned int count = 0;

			if (fread(&count, sizeof(int), 1, fid) != sizeof(int))
				exit(EXIT_FAILURE);

			if (fread(data_csr->colInd + data_csr->rowOffsets[i], sizeof(int), count, fid) != sizeof(int))
				exit(EXIT_FAILURE);

			if (fread(data_csr->values + data_csr->rowOffsets[i], sizeof(float), count, fid) != sizeof(float))
				exit(EXIT_FAILURE);
			
			
			unsigned int maxDim = 1 + data_csr->colInd[data_csr->rowOffsets[i+1]-1];
			if(dimVects < maxDim) dimVects = maxDim;
		}
		dimVects_aligned = ALIGN_UP(dimVects, req_data_format->dimAlignment);
		data_csr->numCols = dimVects;

		printf("NNZ: %d\n%% NNZ: %.3lf\nAvg. NNZ per row: %.3lf\n", data_csr->nnz, 100. * data_csr->nnz / (numVects * dimVects), data_csr->nnz / (double)numVects);
		
	}

	//make class labels
	int max_idx = -2;
	for(unsigned int i=0; i < numVects; i++) {
		if(max_idx < vector_labels[i]) max_idx = vector_labels[i];
	}
	class_labels = (int *) malloc(sizeof(int) * (max_idx + 2));
	for(int i=0; i < max_idx + 2; i++) class_labels[i] = -2;
	for(unsigned int i=0; i < numVects; i++) {
		int ci = vector_labels[i];
		if(ci < -1) REPORT_ERROR("Class index lower than -1");
		if(class_labels[ci+1] == -2) class_labels[ci+1] = ci;
	}
	numClasses = 0;
	for(int i=0; i < max_idx + 2; i++) {
		if(class_labels[i] != -2) {
			class_labels[numClasses] = class_labels[i];
			numClasses++;
		}
	}

	//store if the first label is negative: to LibSVM compatibility of stored model:
	invertLabels = !(vector_labels[0] == 1);

	//if float labels required:
	if(req_data_format->labelsInFloat) {
		labelsInFloat = true;
		float *p = (float *)vector_labels;
		for(unsigned int i=0; i < numVects; i++) p[i] = (float) vector_labels[i];
	}

	return 0;
} //load_lasvm_binary_data

///////////////////////////////////////////////////////////////
// SvmModel
SvmModel::SvmModel() {
	alphas = NULL;
	data = NULL;
	params = NULL;
	allocatedByCudaHost = false;
}

SvmModel::~SvmModel() {
	Delete();
}

int SvmModel::Delete() {
	//don't delete data & params - they are only poiters - they need to be Destroyet themself externally
	data = NULL;
	params = NULL;
//#ifndef __NO_CUDA
//	if(allocatedByCudaHost)	cudaFree(alphas);
//	else free(alphas);
//#else
	free(alphas);
//#endif
	alphas = NULL;
	return SUCCESS;
}

int SvmModel::StoreModelGeneric(const char *model_file_name, SVM_MODEL_FILE_TYPE type) {

	switch(type) {
		case M_LIBSVM_TXT:
			SAFE_CALL(StoreModel_LIBSVM_TXT(model_file_name));
			break;
		default:
			REPORT_ERROR("Unsupported model storage format");
	}

	return SUCCESS;
}

int SvmModel::StoreModel_LIBSVM_TXT(const char *model_file_name) {
	//unsigned int i,
	//	j,
	//	width,
	//	height;
	//float alpha,
	//	value,
	//	alpha_mult;
	FILE *fid;

	if (alphas == NULL || data == NULL || params == NULL) {
		return FAILURE;
	}
	if(data->data_dense == NULL && data->data_csr == NULL) {
		return FAILURE;
	}

	FILE_SAFE_OPEN(fid, model_file_name, "w");

	unsigned int height = data->GetNumVects();
	unsigned int width =  data->GetDimVects();

	/* Print header. */
	fprintf(fid, "svm_type c_svc\nkernel_type %s\n", kernel_type_table[params->kernel_type]);
	switch (params->kernel_type) {
	case POLY:
		fprintf(fid, "degree %d\n", params->degree);
		break;
	case SIGMOID:
		fprintf(fid, "coef0 %g\n", params->coef0);
		break;
	case RBF:
		fprintf(fid, "gamma %g\n", params->gamma);
		break;
	}
	fprintf(fid, "nr_class %d\ntotal_sv %d\n", data->numClasses, params->nsv_class1 + params->nsv_class2);

	//float alpha_mult = (data->class_labels[0] > data->class_labels[1])? -1.0f : 1.0f;
	unsigned int classIds[2] = {0, 1};
	//store labels and counts - to LibSVM compatible
	if(data->invertLabels) {
		fprintf(fid, "rho %g\nlabel %d %d\nnr_sv %d %d\nSV\n", -params->rho, data->class_labels[0], data->class_labels[1], params->nsv_class1, params->nsv_class2);
		classIds[0] = 1;
		classIds[1] = 0;
	} else {
		fprintf(fid, "rho %g\nlabel %d %d\nnr_sv %d %d\nSV\n", params->rho, data->class_labels[1], data->class_labels[0], params->nsv_class2, params->nsv_class1);
	}

	//store positive Support Vectors
	csr *data_csr = data->GetDataSparsePointer();
	for (unsigned int i = 0; i < height; i++) {
		if(alphas[i] > 0.0f) {
			if(data->labelsInFloat && ((float*)data->vector_labels)[i] != (float) data->class_labels[classIds[1]]) continue;
			if(!data->labelsInFloat && data->vector_labels[i] != data->class_labels[classIds[1]]) continue;
			float a = alphas[i];
			fprintf(fid, "%.16g ", a);

			if(data->data_dense != NULL) {
				for (unsigned int j = 0; j < width; j++) {
					float value = data->GetValue(i, j);
					if (value != 0.0F) {
						if (value == 1.0F) {
							fprintf(fid, "%d:1 ", j + 1);
						} else {
							fprintf(fid, "%d:%g ", j + 1, value);
						}
					}
				}
			} else { //CSR data
				for (unsigned int j = data_csr->rowOffsets[i]; j < data_csr->rowOffsets[i+1]; j++) {
					float value = data_csr->values[j];
					if (value == 1.0F) {
						fprintf(fid, "%d:1 ", data_csr->colInd[j] + 1);
					} else {
						fprintf(fid, "%d:%g ", data_csr->colInd[j] + 1, value);
					}
				}
			}

			fprintf(fid, "\n");
		}
	}

	//store negative Support Vectors
	for (unsigned int i = 0; i < height; i++) {
		if(alphas[i] > 0.0f) {
			if(data->labelsInFloat && ((float*)data->vector_labels)[i] != (float) data->class_labels[classIds[0]]) continue;
			if(!data->labelsInFloat && data->vector_labels[i] != data->class_labels[classIds[0]]) continue;
			float a = alphas[i];
			fprintf(fid, "%.16g ", -a);

			if(data->data_dense != NULL) {
				for (unsigned int j = 0; j < width; j++) {
					float value = data->GetValue(i, j);
					if (value != 0.0F) {
						if (value == 1.0F) {
							fprintf(fid, "%d:1 ", j + 1);
						} else {
							fprintf(fid, "%d:%g ", j + 1, value);
						}
					}
				}
			} else { //CSR data
				for (unsigned int j = data_csr->rowOffsets[i]; j < data_csr->rowOffsets[i+1]; j++) {
					float value = data_csr->values[j];
					if (value == 1.0F) {
						fprintf(fid, "%d:1 ", data_csr->colInd[j] + 1);
					} else {
						fprintf(fid, "%d:%g ", data_csr->colInd[j] + 1, value);
					}
				}
			}

			fprintf(fid, "\n");
		}
	}

	fclose(fid);
	return SUCCESS;
} //StoreModel

int SvmModel::LoadModelGeneric(char *model_file_name, SVM_MODEL_FILE_TYPE type, SVM_DATA_TYPE data_type, struct svm_memory_dataformat *req_data_format) {

    switch(type) {
        case M_LIBSVM_TXT:
            SAFE_CALL(LoadModel_LIBSVM_TXT(model_file_name, data_type, req_data_format));
            break;
        default:
            REPORT_ERROR("Unsupported model storage format");
    }

    return SUCCESS;
}

int SvmModel::LoadModel_LIBSVM_TXT(char *model_file_name, SVM_DATA_TYPE data_type, struct svm_memory_dataformat *req_data_format) {

	if (data == NULL || params == NULL) {
		return FAILURE;
	}
    bool sparse = (req_data_format->supported_types & SUPPORTED_FORMAT_CSR) && (data_type == UNKNOWN || data_type == SPARSE);
    data->type = sparse ? SPARSE : DENSE;

    //std::ifstream fin(model_file_name, std::ios::in | std::ios::binary);
	std::ifstream fin(model_file_name, std::ios::in);
    if (!fin)
    {
        std::cerr << "Failed to open model file '" << model_file_name << "' for reading\n";
        return FAILURE;
    }
    std::string line;
    //read header
    while (std::getline(fin, line))
    {
		if (line.compare("SV") == 0)  //header end
            break;
        std::vector<std::string> tokens;
        split(line, ' ', tokens);

        if (tokens[0].compare("svm_type") == 0) {
			if (tokens[1].compare("c_svc") != 0)
                throw std::runtime_error("Unsupported SVM type");
        }
        else if (tokens[0].compare("kernel_type") == 0) {
            int kernel_type = 0;
            while (kernel_type_table[kernel_type]) {
                if (tokens[1].compare(kernel_type_table[kernel_type]) == 0) {
                    params->kernel_type = kernel_type;
                    break;
                }
                kernel_type++;
            }
            if (!kernel_type_table[kernel_type])
                throw std::runtime_error("Unsupported/wrong kernel type");
        }
        else if (tokens[0].compare("degree") == 0) {
            if (params->kernel_type != POLY)
                throw std::runtime_error("Parameter 'degree' can be set only for polynomial kernels");
            params->degree = std::stoi(tokens[1]);
        }
        else if (tokens[0].compare("coef0") == 0) {
            if (params->kernel_type != SIGMOID)
                throw std::runtime_error("Parameter 'coef0' can be set only for sigmoid kernels");
            params->coef0 = std::stod(tokens[1]);
        }
        else if (tokens[0].compare("gamma") == 0) {
            if (params->kernel_type != RBF)
                throw std::runtime_error("Parameter 'gamma' can be set only for RBF kernels");
            params->gamma = std::stod(tokens[1]);
        }
        else if (tokens[0].compare("nr_class") == 0) {
            data->numClasses = std::stoi(tokens[1]);
            data->class_labels = (int *)malloc(data->numClasses * sizeof(int));
        }
        else if (tokens[0].compare("total_sv") == 0) {
            //ignore
        }
        else if (tokens[0].compare("rho") == 0) {
            params->rho = std::stod(tokens[1]);
        }
        else if (tokens[0].compare("label") == 0) {
            data->class_labels[1] = std::stoi(tokens[1]);
            data->class_labels[0] = std::stoi(tokens[2]);
        }
        else if (tokens[0].compare("nr_sv") == 0) {
            params->nsv_class2 = std::stoi(tokens[1]);
            params->nsv_class1 = std::stoi(tokens[2]);
        }
    }
    //get data dim
    int nnz = 0;
    std::streampos spos = fin.tellg();
    while (std::getline(fin, line)) {
        std::vector<std::string> tokens;
        split(line, ' ', tokens);
        for (int i = 1; i < tokens.size(); i++) {
            int d = std::stoi(tokens[i]);
            if (d > data->dimVects)
                data->dimVects = d;
        }
        nnz += tokens.size() - 1;
    }
    fin.clear();
    fin.seekg(spos);
    data->numVects = params->nsv_class1 + params->nsv_class2;
    data->numVects_aligned = ALIGN_UP(data->numVects, req_data_format->vectAlignment);
    data->dimVects_aligned = ALIGN_UP(data->dimVects, req_data_format->dimAlignment);

    malloc_general(req_data_format, (void **) & alphas, sizeof(float) * data->numVects_aligned);
    if (sparse) {
        data->data_csr = new csr;
        data->data_csr->nnz = nnz;
        data->data_csr->numCols = data->dimVects;
        data->data_csr->numRows = data->numVects;
        malloc_general(req_data_format, (void **) &data->data_csr->values, sizeof(float) * nnz);
        malloc_general(req_data_format, (void **) &data->data_csr->colInd, sizeof(unsigned int) * nnz);
        malloc_general(req_data_format, (void **) &data->data_csr->rowOffsets, sizeof(unsigned int) * (data->numVects_aligned + 1));
        data->data_csr->rowOffsets[0] = 0;
    }
    else {
        malloc_general(req_data_format, (void **) &data->data_dense, sizeof(float) * data->dimVects_aligned * data->numVects_aligned);
        memset(data->data_dense, 0, sizeof(float) * data->dimVects_aligned * data->numVects_aligned);
    }
    //read SV
    int k = 0;
    int offset = 0;
    while (std::getline(fin, line)) {
        std::vector<std::string> tokens;
        split(line, ' ', tokens);
        alphas[k] = std::stof(tokens[0]);
        for (int i = 1; i < tokens.size(); i++) {
            size_t colon;
            int d = std::stoi(tokens[i], &colon) - 1;
            float v = std::strtof(tokens[i].c_str() + colon + 1, 0);
            if (sparse) {
                data->data_csr->values[offset] = v;
                data->data_csr->colInd[offset] = d;
                offset++;
            }
            else {
                if (req_data_format->transposed)
                    data->data_dense[data->numVects_aligned * d + k] = v;
                else
                    data->data_dense[data->dimVects_aligned * k + d] = v;
            }
        }
        k++;
        if (sparse)
            data->data_csr->rowOffsets[k] = offset;
    }
    if (sparse)
        for (k++ ; k < data->numVects_aligned + 1; k++)
            data->data_csr->rowOffsets[k] = offset;

    return SUCCESS;
}

int SvmModel::CalculateSupperVectorCounts() {
	if(alphas == NULL || params == NULL || data == NULL) return FAILURE;

	params->nsv_class1 = 0;
	params->nsv_class2 = 0;
	int *labels = data->GetVectorLabelsPointer();
	int *class_labels = data->GetClassLabelsPointer();
	for(unsigned int i=0; i < data->GetNumVects(); i++) {
		if(alphas[i] > 0) {
			if(data->GetLabelsInFloat()) {
				 if(((float*)labels)[i] == (float) class_labels[0]) params->nsv_class1++;
											                   else params->nsv_class2++;
			} else {
				 if(labels[i] == class_labels[0]) params->nsv_class1++;
										     else params->nsv_class2++;
			}
		}
	}
	return SUCCESS;
}

int SvmModel::Predict(SvmData *testData, const char * file_out)
{
    return FAILURE;
}

int SvmModel::LoadModel(char *model_file_name, SVM_MODEL_FILE_TYPE type, SVM_DATA_TYPE data_type)
{
    return FAILURE;
}

void malloc_general(svm_memory_dataformat *req_data_format, void **x, size_t size) {
	if (req_data_format->allocate_write_combined) {
			malloc_host_WC(x, size);
	} else {
		if (req_data_format->allocate_pinned) {
			malloc_host_PINNED(x, size);
		} else {
			*x = malloc(size);
		}
	}
} //malloc_general


int GenericSvmData::Load(char *filename, SVM_FILE_TYPE file_type, SVM_DATA_TYPE data_type) {

	svm_memory_dataformat req_data_format;
	req_data_format.allocate_pinned = false;
	req_data_format.allocate_write_combined = false;
	req_data_format.dimAlignment = 1;
	req_data_format.vectAlignment = 1;
	req_data_format.transposed = false;
	req_data_format.labelsInFloat = false;
	req_data_format.supported_types = SUPPORTED_FORMAT_DENSE | SUPPORTED_FORMAT_CSR;

	SAFE_CALL(SvmData::Load(filename, file_type, data_type, &req_data_format));

	return SUCCESS;
}
