// @file nnconv_cudnn.cu
// @brief Convolution block CuDNN-based implementation.
// @author Andrea Vedaldi

/*
Copyright (C) 2015-16 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#if !defined(ENABLE_GPU) | !defined(ENABLE_CUDNN)
#error "nnconv6D_cudnn.hpp cannot be compiled without GPU and CUDNN support."
#endif

#include "nnconv6D_cudnn.hpp"
#include "cudnnhelper.hpp"
#include "../datacu.hpp"
#include <assert.h>
#include <algorithm>
#include <iostream>

using namespace vl ;

#define CHECK(x) \
{ \
cudnnError = x ; \
if (cudnnError != CUDNN_STATUS_SUCCESS) { \
error = context.setError(context.getCudaHelper().catchCudnnError(cudnnError, \
STRINGIZE(__FILE__) ":" STRINGIZE(__LINE__))) ; \
goto done ; \
} }

/* ---------------------------------------------------------------- */
/*                                             nnconv_forward_cudnn */
/* ---------------------------------------------------------------- */

namespace vl { namespace impl {

  template<vl::DataType dataType>
  vl::ErrorCode
  vl::impl::nnconv6D_cudnn<dataType>::forward(Context& context,
                                            Tensor output, double outputMult,
                                            Tensor data, double dataMult,
                                            Tensor filters,
                                            Tensor biases,
                                            int strideY, int strideX,
                                            int strideAnY, int strideAnX,
                                            int padTop, int padBottom,
                                            int padLeft, int padRight,
                                            int padAnTop, int padAnBottom,
                                            int padAnLeft, int padAnRight)
  {
    assert(output) ;
    assert(data) ;
    assert(filters) ;

    typedef typename DataTypeTraits<dataType>::type type ;

    cudnnTensorDescriptor_t outputDesc, biasesDesc, dataDesc ;
    cudnnFilterDescriptor_t filtersDesc ;
    cudnnConvolutionDescriptor_t convDesc ;
    bool outputDescInitialized = false ;
    bool biasesDescInitialized = false ;
    bool dataDescInitialized = false ;
    bool filtersDescInitialized = false ;
    bool convDescInitialized = false ;

    void* workSpace = NULL ;

    int numGroups = data.getDimension(4) / filters.getDimension(4) ;
    int numFiltersPerGroup = filters.getSize() / numGroups ;

    if (padLeft != padRight) return vl::VLE_Unsupported ;
    if (padTop != padBottom) return vl::VLE_Unsupported ;
    if (padAnLeft != padAnRight) return vl::VLE_Unsupported ;
    if (padAnTop != padAnBottom) return vl::VLE_Unsupported ;
    if (filters.getDimension(0) > data.getDimension(0)) return vl::VLE_Unsupported ;
    if (filters.getDimension(1) > data.getDimension(1)) return vl::VLE_Unsupported ;
    if (filters.getDimension(2) > data.getDimension(2)) return vl::VLE_Unsupported ;
    if (filters.getDimension(3) > data.getDimension(3)) return vl::VLE_Unsupported ;

    cudnnStatus_t cudnnError = CUDNN_STATUS_SUCCESS ;
    vl::ErrorCode error = vl::VLE_Success ;
    cudnnHandle_t handle ;

    // Get CuDNN
    CHECK(context.getCudaHelper().getCudnnHandle(&handle)) ;

    // Get tensor descripotrs
    {
    CHECK(cudnnCreateTensorDescriptor(&outputDesc)) ;
    outputDescInitialized = true ;
    
    int out_n = output.getDimension(5);
    int out_c = output.getDimension(4);
    int out_w = output.getDimension(0);
    int out_h = output.getDimension(1);
    int out_an_w = output.getDimension(2);
    int out_an_h = output.getDimension(3);
    int out_dims [6] = {out_n, out_c, out_w, out_h, out_an_w, out_an_h};
    int out_strides [6] = {out_c*out_w*out_h*out_an_w*out_an_h, out_w*out_h*out_an_w*out_an_h,
                       out_h*out_an_w*out_an_h, out_an_w*out_an_h, out_an_h, 1};

    CHECK(cudnnSetTensorNdDescriptor(outputDesc,
                                       DataTypeToCudnn<dataType>::id ,
                                       6 ,
                                       out_dims,
                                       out_strides)) ;
    }
    
    {
    CHECK(cudnnCreateTensorDescriptor(&dataDesc)) ;
    dataDescInitialized = true ;

    int data_n = data.getDimension(5);
    int data_c = data.getDimension(4);
    int data_w = data.getDimension(0);
    int data_h = data.getDimension(1);
    int data_an_w = data.getDimension(2);
    int data_an_h = data.getDimension(3);
    int data_dims [6] = {data_n, data_c, data_w, data_h, data_an_w, data_an_h};
    int data_strides [6] = {data_c*data_w*data_h*data_an_w*data_an_h, data_w*data_h*data_an_w*data_an_h,
                       data_h*data_an_w*data_an_h, data_an_w*data_an_h, data_an_h, 1};

    CHECK(cudnnSetTensorNdDescriptor(dataDesc,
                                       DataTypeToCudnn<dataType>::id,
                                       6,
                                       data_dims,
                                       data_strides)) ;
    }

    {
    CHECK(cudnnCreateFilterDescriptor(&filtersDesc)) ;
    filtersDescInitialized = true ;

    int filter_n = filters.getDimension(5);
    int filter_c = filters.getDimension(4);
    int filter_w = filters.getDimension(0);
    int filter_h = filters.getDimension(1);
    int filter_an_w = filters.getDimension(2);
    int filter_an_h = filters.getDimension(3);
    int filter_dims [6] = {filter_n, filter_c, filter_w, filter_h, filter_an_w, filter_an_h};    

    CHECK(cudnnSetFilterNdDescriptor(filtersDesc,
                                     DataTypeToCudnn<dataType>::id,
                                     IF_CUDNN_GE5(CUDNN_TENSOR_NCHW COMMA)
                                     6,
                                     filter_dims)) ;
    }

    if (biases) {
      CHECK(cudnnCreateTensorDescriptor(&biasesDesc)) ;
      biasesDescInitialized = true ;

      int bias_c = biases.getNumElements() / numGroups;
      int bias_dims [6] = {1,bias_c,1,1,1,1};
      int bias_strides [6] = { bias_c, 1,1,1,1,1};
      
      CHECK(cudnnSetTensorNdDescriptor(biasesDesc,
                                       DataTypeToCudnn<dataType>::id ,
                                       6,
                                       bias_dims,
                                       bias_strides)) ;
    }

    // Get convolution descriptor
    {
    CHECK(cudnnCreateConvolutionDescriptor(&convDesc)) ;
    convDescInitialized = true ;

    int conv_pad [4] = {padLeft, padTop,padAnLeft, padAnTop};
    int conv_stride [4] = {strideX, strideY, strideAnX, strideAnY};
    int conv_up [4] = {1,1,1,1};
    CHECK(cudnnSetConvolutionNdDescriptor(convDesc,
                                          4,
                                          conv_pad,
                                          conv_stride,
                                          conv_up, // upscale
                                          CUDNN_CROSS_CORRELATION,
                                          DataTypeToCudnn<dataType>::id)) ;
    }
    // Sanity check

#if 1
    {
      int output_dims [6];
      cudnnGetConvolutionNdForwardOutputDim(convDesc,
                                            dataDesc,
                                            filtersDesc,
                                            6,
                                            output_dims) ;
      bool sane =
      output.getDimension(5) == output_dims[0] &&
      numFiltersPerGroup == output_dims[1] &&
      output.getDimension(0) == output_dims[2] &&
      output.getDimension(1) == output_dims[3] &&
      output.getDimension(2) == output_dims[4] &&
      output.getDimension(3) == output_dims[5] ;
      assert(sane) ;
    }
#endif
    context.getCudaHelper().getCudnnEnabled();
    context.getCudaHelper().cudnnConvolutionFwdWorkSpaceUsed = 0 ;
    context.getCudaHelper().cudnnConvolutionBwdFilterWorkSpaceUsed = 0 ;
    context.getCudaHelper().cudnnConvolutionBwdDataWorkSpaceUsed = 0 ;
    
    
    if (!context.getCudaHelper().cudnnConvolutionFwdSpecificAlgo) {
      // Determine algorithm automatically
      CHECK(cudnnGetConvolutionForwardAlgorithm(handle,
                                                dataDesc,
                                                filtersDesc,
                                                convDesc,
                                                outputDesc,
                                                context.getCudaHelper().cudnnConvolutionFwdPreference,
                                                context.getCudaHelper().cudnnConvolutionFwdWorkSpaceLimit,
                                                &context.getCudaHelper().cudnnConvolutionFwdAlgo)) ;
    }

    // Get workspace size
    CHECK(cudnnGetConvolutionForwardWorkspaceSize(handle,
                                                  dataDesc,
                                                  filtersDesc,
                                                  convDesc,
                                                  outputDesc,
                                                  context.getCudaHelper().cudnnConvolutionFwdAlgo,
                                                  &context.getCudaHelper().cudnnConvolutionFwdWorkSpaceUsed)) ;

    // Get workspace
    if (context.getCudaHelper().cudnnConvolutionFwdWorkSpaceUsed > 0) {
      workSpace = context.getWorkspace(vl::VLDT_GPU, context.getCudaHelper().cudnnConvolutionFwdWorkSpaceUsed) ;
      if (workSpace == NULL) {
        error = context.getLastError() ;
        goto done ;
      }
    }

    // Perform convolution for each filter group
    for (int g = 0  ; g < numGroups ; ++g) {
      ptrdiff_t dataGrpOffset = (data.getHeight() * data.getWidth() * filters.getDepth()) *  g ;
      ptrdiff_t filtersGrpOffset = (filters.getHeight() * filters.getWidth() * filters.getDepth()) * numFiltersPerGroup * g ;
      ptrdiff_t outputGrpOffset = (output.getHeight() * output.getWidth() * numFiltersPerGroup) * g ;
      ptrdiff_t biasesGrpOffset = numFiltersPerGroup * g ;

      type alpha = dataMult ;
      type beta = outputMult ;
      CHECK(cudnnConvolutionForward(handle,
                                    &alpha,
                                    dataDesc, (type const*)data.getMemory() + dataGrpOffset,
                                    filtersDesc, (type const*)filters.getMemory() + filtersGrpOffset,
                                    convDesc,
                                    context.getCudaHelper().cudnnConvolutionFwdAlgo,
                                    workSpace, context.getCudaHelper().cudnnConvolutionFwdWorkSpaceUsed,
                                    &beta,
                                    outputDesc, (type*)output.getMemory() + outputGrpOffset)) ;

      if (biases) {
        type alpha = 1.0f ;
        type beta = 1.0f ;
#if (CUDNN_VERSION < 4000)
        CHECK(cudnnAddTensor(handle,
                             CUDNN_ADD_SAME_C,
                             &alpha,
                             biasesDesc, (type const*)biases.getMemory() + biasesGrpOffset,
                             &beta,
                             outputDesc, (type*)output.getMemory() + outputGrpOffset)) ;
#else
        CHECK(cudnnAddTensor(handle,
                             &alpha,
                             biasesDesc, (type const*)biases.getMemory() + biasesGrpOffset,
                             &beta,
                             outputDesc, (type*)output.getMemory() + outputGrpOffset)) ;
#endif
      }
    }

    /* cleanup */
  done:
    if (convDescInitialized) { cudnnDestroyConvolutionDescriptor(convDesc) ; }
    if (filtersDescInitialized) { cudnnDestroyFilterDescriptor(filtersDesc) ; }
    if (dataDescInitialized) { cudnnDestroyTensorDescriptor(dataDesc) ; }
    if (biasesDescInitialized) { cudnnDestroyTensorDescriptor(biasesDesc) ; }
    if (outputDescInitialized) { cudnnDestroyTensorDescriptor(outputDesc) ; }
    return context.passError(error, __func__) ;
  }

  /* ---------------------------------------------------------------- */
  /*                                            nnconv_backward_cudnn */
  /* ---------------------------------------------------------------- */

  template<vl::DataType dataType>
  vl::ErrorCode
  vl::impl::nnconv6D_cudnn<dataType>::backward(Context& context,
                                             Tensor derData,
                                             Tensor derFilters,
                                             Tensor derBiases,
                                             Tensor data,
                                             Tensor filters,
                                             Tensor derOutput,
                                             int strideY, int strideX,
                                             int strideAnY, int strideAnX,
                                             int padTop, int padBottom,
                                             int padLeft, int padRight,
                                             int padAnTop, int padAnBottom,
                                             int padAnLeft, int padAnRight)
  {
    typedef typename DataTypeTraits<dataType>::type type ;

    /* no derDataDesc needed as same as dataDesc */
    cudnnTensorDescriptor_t dataDesc, derBiasesDesc, derOutputDesc ;
    cudnnFilterDescriptor_t filtersDesc ;
    cudnnConvolutionDescriptor_t convDesc ;
    bool dataDescInitialized = false ;
    bool derBiasesDescInitialized = false ;
    bool derOutputDescInitialized = false ;
    bool filtersDescInitialized = false ;
    bool convDescInitialized = false ;

#if (CUDNN_VERSION >= 3000)
    void* workSpace = NULL ;
    size_t workSpaceSize = 0 ;
#endif

    ptrdiff_t numGroups = 1 ;
    ptrdiff_t numFiltersPerGroup = 0 ;
    ptrdiff_t filtersVolume = 0 ;

    if (padLeft != padRight) return vl::VLE_Unsupported ;
    if (padTop != padBottom) return vl::VLE_Unsupported ;
    if (padAnLeft != padAnRight) return vl::VLE_Unsupported ;
    if (padAnTop != padAnBottom) return vl::VLE_Unsupported ;

    cudnnStatus_t cudnnError = CUDNN_STATUS_SUCCESS ;
    vl::ErrorCode error = vl::VLE_Success ;
    cudnnHandle_t handle ;

    // Get CuDNN
    CHECK(context.getCudaHelper().getCudnnHandle(&handle)) ;

    // Get the dimensions of the tensrors involved
    // If derData is specified (hence comptued as output), use this
    // tensor as a basis to compute such dimensions, otherwise use derFilters.

    if (derData) {
      assert(filters) ;
      numGroups = derData.getDimension(4) / filters.getDimension(4) ;
      numFiltersPerGroup = filters.getDimension(5) / numGroups ;
      filtersVolume = filters.getDimension(0) * filters.getDimension(1) * filters.getDimension(2) 
                      * filters.getDimension(3) * filters.getDimension(4) ;

      
      CHECK(cudnnCreateTensorDescriptor(&dataDesc)) ;
      dataDescInitialized = true ;

      int der_data_n = derData.getDimension(5);
      int der_data_c = derData.getDimension(4);
      int der_data_w = derData.getDimension(0);
      int der_data_h = derData.getDimension(1);
      int der_data_an_w = derData.getDimension(2);
      int der_data_an_h = derData.getDimension(3);
      int der_data_dims [6] = {der_data_n, der_data_c, der_data_w, der_data_h, der_data_an_w, der_data_an_h};
      int der_data_strides [6] = {der_data_c*der_data_w*der_data_h*der_data_an_w*der_data_an_h, der_data_w*der_data_h*der_data_an_w*der_data_an_h,
                       der_data_h*der_data_an_w*der_data_an_h, der_data_an_w*der_data_an_h, der_data_an_h, 1};

      CHECK(cudnnSetTensorNdDescriptor(dataDesc,
                                         DataTypeToCudnn<dataType>::id ,
                                         6,
                                         der_data_dims,
                                         der_data_strides)) ;

      

      CHECK(cudnnCreateFilterDescriptor(&filtersDesc)) ;
      filtersDescInitialized = true ;

      int filter_n = filters.getDimension(5);
      int filter_c = filters.getDimension(4);
      int filter_w = filters.getDimension(0);
      int filter_h = filters.getDimension(1);
      int filter_an_w = filters.getDimension(2);
      int filter_an_h = filters.getDimension(3);
      int filter_dims [6] = {filter_n, filter_c, filter_w, filter_h, filter_an_w, filter_an_h};

      CHECK(cudnnSetFilterNdDescriptor(filtersDesc,
                                       DataTypeToCudnn<dataType>::id ,
                                       IF_CUDNN_GE5(CUDNN_TENSOR_NCHW COMMA)
                                       6,
                                       filter_dims)) ;
    } else if (derFilters) {
      assert(data) ;
      numGroups = data.getDimension(4) / derFilters.getDimension(4) ;
      numFiltersPerGroup = derFilters.getDimension(5) / numGroups ;
      filtersVolume = derFilters.getDimension(0) * derFilters.getDimension(1) * derFilters.getDimension(2) 
                      * derFilters.getDimension(3) * derFilters.getDimension(4) ;

      CHECK(cudnnCreateTensorDescriptor(&dataDesc)) ;
      dataDescInitialized = true ;
      
      int data_n = data.getDimension(5);
      int data_c = data.getDimension(4);
      int data_w = data.getDimension(0);
      int data_h = data.getDimension(1);
      int data_an_w = data.getDimension(2);
      int data_an_h = data.getDimension(3);
      int data_dims [6] = {data_n, data_c, data_w, data_h, data_an_w, data_an_h};
      int data_strides [6] = {data_c*data_w*data_h*data_an_w*data_an_h, data_w*data_h*data_an_w*data_an_h,
                       data_h*data_an_w*data_an_h, data_an_w*data_an_h, data_an_h, 1};

      CHECK(cudnnSetTensorNdDescriptor(dataDesc,
                                         DataTypeToCudnn<dataType>::id ,
                                         6,
                                         data_dims,
                                         data_strides)) ;

      CHECK(cudnnCreateFilterDescriptor(&filtersDesc)) ;
      filtersDescInitialized = true ;

      int derFilter_n = derFilters.getDimension(5);
      int derFilter_c = derFilters.getDimension(4);
      int derFilter_w = derFilters.getDimension(0);
      int derFilter_h = derFilters.getDimension(1);
      int derFilter_an_w = derFilters.getDimension(2);
      int derFilter_an_h = derFilters.getDimension(3);
      int derFilter_dims [6] = {derFilter_n, derFilter_c, derFilter_w, derFilter_h, derFilter_an_w, derFilter_an_h};

      CHECK(cudnnSetFilterNdDescriptor(filtersDesc,
                                       DataTypeToCudnn<dataType>::id ,
                                       IF_CUDNN_GE5(CUDNN_TENSOR_NCHW COMMA)
                                       6,
                                       derFilter_dims)) ;
    }

    {       
    CHECK(cudnnCreateConvolutionDescriptor(&convDesc)) ;
    convDescInitialized = true ;
    int conv_pad [4] = {padLeft, padTop,padAnLeft, padAnTop};
    int conv_stride [4] = {strideX, strideY, strideAnX, strideAnY};
    int conv_up [4] = {1,1,1,1};
    CHECK(cudnnSetConvolutionNdDescriptor(convDesc,
                                          4,
                                          conv_pad,
                                          conv_stride,
                                          conv_up, // upscale
                                          CUDNN_CROSS_CORRELATION,
                                          DataTypeToCudnn<dataType>::id)) ;
    }

    // Must have derOutput for all derivatives
    {
    assert(derOutput) ;
    CHECK(cudnnCreateTensorDescriptor(&derOutputDesc)) ;
    derOutputDescInitialized = true ;

    int der_out_n = derOutput.getDimension(5);
    int der_out_c = derOutput.getDimension(4);
    int der_out_w = derOutput.getDimension(0);
    int der_out_h = derOutput.getDimension(1);
    int der_out_an_w = derOutput.getDimension(2);
    int der_out_an_h = derOutput.getDimension(3);
    int der_out_dims [6] = {der_out_n, der_out_c, der_out_w, der_out_h, der_out_an_w, der_out_an_h};
    int der_out_strides [6] = {der_out_c*der_out_w*der_out_h*der_out_an_w*der_out_an_h, der_out_w*der_out_h*der_out_an_w*der_out_an_h,
                       der_out_h*der_out_an_w*der_out_an_h, der_out_an_w*der_out_an_h, der_out_an_h, 1};

    CHECK(cudnnSetTensorNdDescriptor(derOutputDesc,
                                       DataTypeToCudnn<dataType>::id ,
                                       6, 
                                       der_out_dims,
                                       der_out_strides)) ;
    }
    // for derivatives w.r.t. bias
    if (derBiases) {
      CHECK(cudnnCreateTensorDescriptor(&derBiasesDesc)) ;
      derBiasesDescInitialized = true ;

      int der_bias_c = derBiases.getNumElements() / numGroups;
      int der_bias_dims [6] = {1,der_bias_c,1,1,1,1};
      int der_bias_strides [6] = { der_bias_c, 1,1,1,1,1};
      
      CHECK(cudnnSetTensorNdDescriptor(derBiasesDesc,
                                       DataTypeToCudnn<dataType>::id ,
                                       6,
                                       der_bias_dims,
                                       der_bias_strides)) ;
    }


    context.getCudaHelper().cudnnConvolutionFwdWorkSpaceUsed = 0 ;
    context.getCudaHelper().cudnnConvolutionBwdFilterWorkSpaceUsed = 0 ;
    context.getCudaHelper().cudnnConvolutionBwdDataWorkSpaceUsed = 0 ;

#if (CUDNN_VERSION >= 3000)

    if (derFilters) {
      // Get filter derivatives algorithm
      CHECK(cudnnGetConvolutionBackwardFilterAlgorithm
            (handle,
             dataDesc,
             derOutputDesc,
             convDesc,
             filtersDesc,
             context.getCudaHelper().cudnnConvolutionBwdFilterPreference,
             context.getCudaHelper().cudnnConvolutionBwdFilterWorkSpaceLimit,
             &context.getCudaHelper().cudnnConvolutionBwdFilterAlgo)) ;

      // Get workspace size
      CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize
            (handle,
             dataDesc,
             derOutputDesc,
             convDesc,
             filtersDesc,
             context.getCudaHelper().cudnnConvolutionBwdFilterAlgo,
             &context.getCudaHelper().cudnnConvolutionBwdFilterWorkSpaceUsed)) ;
      workSpaceSize = std::max(workSpaceSize, context.getCudaHelper().cudnnConvolutionBwdFilterWorkSpaceUsed) ;
    }

    if (derData) {
      // Get data derivatives
      CHECK(cudnnGetConvolutionBackwardDataAlgorithm
            (handle,
             filtersDesc,
             derOutputDesc,
             convDesc,
             dataDesc,
             context.getCudaHelper().cudnnConvolutionBwdDataPreference,
             context.getCudaHelper().cudnnConvolutionBwdDataWorkSpaceLimit,
             &context.getCudaHelper().cudnnConvolutionBwdDataAlgo)) ;

      // Get workspace size
      CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize
            (handle,
             filtersDesc,
             derOutputDesc,
             convDesc,
             dataDesc,
             context.getCudaHelper().cudnnConvolutionBwdDataAlgo,
             &context.getCudaHelper().cudnnConvolutionBwdDataWorkSpaceUsed)) ;
      workSpaceSize = std::max(workSpaceSize, context.getCudaHelper().cudnnConvolutionBwdDataWorkSpaceUsed) ;
    }

    // Get workspace
    if (workSpaceSize > 0) {
      workSpace = context.getWorkspace(vl::VLDT_GPU, workSpaceSize) ;
      if (workSpace == NULL) {
        error = context.getLastError() ;
        goto done ;
      }
    }
#endif

    // Perform backward convolution for each filter group
    for (int g = 0  ; g < numGroups ; ++g) {
      ptrdiff_t filtersGrpOffset = filtersVolume * numFiltersPerGroup  * g ;
      ptrdiff_t derOutputGrpOffset = (derOutput.getHeight() * derOutput.getWidth() * numFiltersPerGroup) * g ;

      if (derBiases) {
        ptrdiff_t derBiasesGrpOffset = numFiltersPerGroup * g ;
        type alpha = 1 ;
        type beta = 0 ;
        CHECK(cudnnConvolutionBackwardBias
              (handle,
               &alpha,
               derOutputDesc, (type const*)derOutput.getMemory() + derOutputGrpOffset,
               &beta,
               derBiasesDesc, (type*)derBiases.getMemory() + derBiasesGrpOffset)) ;
      }

      if (derFilters) {
        ptrdiff_t dataGrpOffset = (data.getHeight() * data.getWidth() * derFilters.getDepth()) *  g ;
        type alpha = 1 ;
        type beta = 0 ;
#if (CUDNN_VERSION >= 3000)
        CHECK(
              IF_CUDNN_GE4(cudnnConvolutionBackwardFilter)
              IF_CUDNN_GE3_LT4(cudnnConvolutionBackwardFilter_v3)
              (handle,
               &alpha,
               dataDesc, (type const*)data.getMemory() + dataGrpOffset,
               derOutputDesc, (type const*)derOutput.getMemory() + derOutputGrpOffset,
               convDesc,
               context.getCudaHelper().cudnnConvolutionBwdFilterAlgo,
               workSpace, workSpaceSize,
               &beta,
               filtersDesc, (type*)derFilters.getMemory() + filtersGrpOffset)) ;
#else
        CHECK(cudnnConvolutionBackwardFilter
              (handle,
               &alpha,
               dataDesc, (type const*)data.getMemory() + dataGrpOffset,
               derOutputDesc, (type const*)derOutput.getMemory() + derOutputGrpOffset,
               convDesc,
               &beta,
               filtersDesc, (type*)derFilters.getMemory() + filtersGrpOffset)) ;
#endif
      }

      if (derData) {
        ptrdiff_t dataGrpOffset = (derData.getHeight() * derData.getWidth() * filters.getDepth()) *  g ;
        type alpha = 1 ;
        type beta = 0 ;

#if (CUDNN_VERSION >= 3000)
        CHECK(
              IF_CUDNN_GE4(cudnnConvolutionBackwardData)
              IF_CUDNN_GE3_LT4(cudnnConvolutionBackwardData_v3)
              (handle,
               &alpha,
               filtersDesc, (type const*)filters.getMemory() + filtersGrpOffset,
               derOutputDesc, (type const*)derOutput.getMemory() + derOutputGrpOffset,
               convDesc,
               context.getCudaHelper().cudnnConvolutionBwdDataAlgo,
               workSpace, workSpaceSize,
               &beta,
               dataDesc, (type*)derData.getMemory() + dataGrpOffset)) ;
#else
        CHECK(cudnnConvolutionBackwardData
              (handle,
               &alpha,
               filtersDesc, filters.getMemory() + filtersGrpOffset,
               derOutputDesc, derOutput.getMemory() + derOutputGrpOffset,
               convDesc,
               &beta,
               dataDesc, derData.getMemory() + dataGrpOffset)) ;
#endif
      }
    }

  done:
    if (convDescInitialized) { cudnnDestroyConvolutionDescriptor(convDesc) ; }
    if (filtersDescInitialized) { cudnnDestroyFilterDescriptor(filtersDesc) ; }
    if (derOutputDescInitialized) { cudnnDestroyTensorDescriptor(derOutputDesc) ; }
    if (derBiasesDescInitialized) { cudnnDestroyTensorDescriptor(derBiasesDesc) ; }
    if (dataDescInitialized) { cudnnDestroyTensorDescriptor(dataDesc) ; }
    return context.passError(error, __func__) ;
  }

} }

// Instantiations
template struct vl::impl::nnconv6D_cudnn<vl::VLDT_Float> ;

#ifdef ENABLE_DOUBLE
template struct vl::impl::nnconv6D_cudnn<vl::VLDT_Double> ;
#endif



