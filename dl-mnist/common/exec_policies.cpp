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


#include "exec_policies.h"

Exec_setup full_exec_setup {
    .no_of_epochs = 3,
    .no_of_minibatches = 6000,
    .no_of_images_per_minibatch = 10,
};

Exec_setup min0_exec_setup {
    .no_of_epochs = 1,
    .no_of_minibatches = 1,
    .no_of_images_per_minibatch = 1,
};

Exec_setup min1_exec_setup {
    .no_of_epochs = 1,
    .no_of_minibatches = 1,
    .no_of_images_per_minibatch = 3,
};

Exec_setup min2_exec_setup {
    .no_of_epochs = 1,
    .no_of_minibatches = 3,
    .no_of_images_per_minibatch = 3,
};


// https://arxiv.org/pdf/2008.10400v2.pdf
// AN ENSEMBLE OF SIMPLE CONVOLUTIONAL NEURAL NETWORK MODELS FOR MNIST DIGIT RECOGNITION
Exec_setup mnist_paper_1_exec_setup {
    .no_of_epochs = 1,
    .no_of_minibatches = 1,
    .no_of_images_per_minibatch = 120,
};