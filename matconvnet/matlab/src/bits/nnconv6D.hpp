// @file nnconv.cu
// @brief Convolution block
// @author Andrea Vedaldi
// @author Max Jaderberg

/*
Copyright (C) 2014 Andrea Vedaldi and Max Jaderberg
Copyright (C) 2015-16 Andrea Vedaldi.

All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef __vl__nnconv6D__
#define __vl__nnconv6D__

#include "data.hpp"

namespace vl {

  vl::ErrorCode
  nnconv6D_forward(vl::Context& context,
                 vl::Tensor output, double outputMult,
                 vl::Tensor data, double dataMult,
                 vl::Tensor filters,
                 vl::Tensor biases,
                 int strideY, int strideX,
                 int strideAnY, int strideAnX,
                 int padTop, int padBottom,
                 int padLeft, int padRight,
                 int padAnTop, int padAnBottom,
                 int padAnLeft, int padAnRight) ;

  vl::ErrorCode
  nnconv6D_backward(vl::Context& context,
                  vl::Tensor derData,
                  vl::Tensor derFilters,
                  vl::Tensor derBiases,
                  vl::Tensor data,
                  vl::Tensor filters,
                  vl::Tensor derOutput,
                  int strideY, int strideX,
                  int strideAnY, int strideAnX,
                  int padTop, int padBottom,
                  int padLeft, int padRight,
                  int padAnTop, int padAnBottom,
                  int padAnLeft, int padAnRight) ;

}


#endif /* defined(__vl__nnconv6D__) */
