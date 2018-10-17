// @file nnconv_blas.hpp
// @brief Convolution block CuDNN-based implementation.
// @author Andrea Vedaldi

/*
Copyright (C) 2015-16 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef __vl__nnconv6D_cudnn__
#define __vl__nnconv6D_cudnn__

#include "../data.hpp"
#include "cudnn.h"

namespace vl { namespace impl {

  template<vl::DataType dataType>
  struct nnconv6D_cudnn
  {
    static vl::ErrorCode
    forward(Context& context,
            Tensor output, double outputMult,
            Tensor data, double dataMult,
            Tensor filters,
            Tensor biases,
            int strideX, int strideY,
            int strideAnY, int strideAnX,
            int padTop, int padBottom,
            int padLeft, int padRight,
            int padAnTop, int padAnBottom,
            int padAnLeft, int padAnRight) ;

    static vl::ErrorCode
    backward(Context& context,
             Tensor derData,
             Tensor derFilters,
             Tensor derBiases,
             Tensor data,
             Tensor filters,
             Tensor derOutput,
             int strideX, int strideY,
             int strideAnY, int strideAnX,
             int padTop, int padBottom,
             int padLeft, int padRight,
             int padAnTop, int padAnBottom,
             int padAnLeft, int padAnRight) ;
  } ;

} }
#endif /* defined(__vl__nnconv6D_cudnn__) */
