// @file im2row_gpu.cu
// @brief Stack image patches as matrix rows (GPU)
// @author Andrea Vedaldi

/*
Copyright (C) 2014-15 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "im2row6D.hpp"
#include "../datacu.hpp"
#include <iostream>

using namespace vl ;

/* ---------------------------------------------------------------- */
/*                                                           im2row */
/* ---------------------------------------------------------------- */

template <typename T> __global__ void
im2row_forward_kernel(T* stacked,
                      T const* data,
                       const int numPatchesX,
                       const int numPatchesY,
                       const int numPatchesXAn,
                       const int numPatchesYAn,
                       const int numPatchSlices,
                       const int width, const int height,
                       const int widthAn, const int heightAn,
                       const int windowWidth, const int windowHeight,
                       const int windowWidthAn, const int windowHeightAn,
                       const int strideX, const int strideY,
                       const int strideXAn, const int strideYAn,
                       const int padLeft, const int padTop,
                       const int padLeftAn, const int padTopAn)
{
  /* each kernel copies the pixels in an image patch for one channel */
  int index = threadIdx.x + blockIdx.x * blockDim.x ;
  if (index < numPatchSlices) {
    /*
     get the patch slice (x,y, xan, yan, z) to copy
     */
    int x = index ;
    int y = x / numPatchesX ;
    int xAn = y / numPatchesY ;
    int yAn = xAn / numPatchesXAn ;
    int z = yAn / numPatchesYAn ;
    x %= numPatchesX ;
    y %= numPatchesY ;
    xAn %= numPatchesXAn ;
    yAn %= numPatchesYAn ;

    /*
     pick the top-left corer of the patch slice in the input image
     */
    int x_data = x * strideX - padLeft ;
    int y_data = y * strideY - padTop ;
    int x_dataAn = xAn * strideXAn - padLeftAn ;
    int y_dataAn = yAn * strideYAn - padTopAn ;
    data += (((z * heightAn + y_dataAn) * widthAn + x_dataAn ) * height + y_data ) * width + x_data;

    /*
     pick the column of the stacked image which contains this patch,
     and move down along the column at the beginning of the patch slice
     */
    int patchSliceOffset = (windowWidth*windowHeight*windowWidthAn*windowHeightAn) * z ;
    stacked += (((numPatchesYAn * patchSliceOffset + yAn) * numPatchesXAn + xAn ) * numPatchesY + y)
                * numPatchesX + x;

    /*
     copy the patch slice
     */
    for (int s = 0 ; s < windowHeightAn ; s += 1) {
      for (int t = 0 ; t < windowWidthAn ; t += 1) {
        for (int v = 0 ; v < windowHeight ; v += 1) {
          for (int u = 0 ; u < windowWidth ; u += 1) {
            if (y_data + v >= 0 &&
                y_data + v < height &&
                x_data + u >= 0 &&
                x_data + u < width &&
                y_dataAn + s >= 0 &&
                y_dataAn + s < heightAn &&
                x_dataAn + t >= 0 &&
                x_dataAn + t < widthAn) {
              *stacked = data[s * width * height * widthAn + t * width * height + v * width + u] ;
            } else {
              *stacked = 0 ;
            }
            stacked += (numPatchesX*numPatchesY*numPatchesXAn*numPatchesYAn) ;
          }
        }
      }
    }
  }
}

/* ---------------------------------------------------------------- */
/*                                           im2row backward kernel */
/* ---------------------------------------------------------------- */

// The next two functions assume b > 0.
__forceinline__ __device__
int floordiv6D(int a, int b)
{
  int q = a/b ;
  if (a >= 0 || a == q*b) return q ;
  return q - 1 ;
}

__forceinline__ __device__
int ceildiv6D(int a, int b)
{
  int q = a/b ;
  if (a <= 0 || a == q*b) return q ;
  return q + 1 ;
}


int floordiv6D_cpu(int a, int b)
{
  int q = a/b ;
  if (a >= 0 || a == q*b) return q ;
  return q - 1 ;
}

int ceildiv6D_cpu(int a, int b)
{
  int q = a/b ;
  if (a <= 0 || a == q*b) return q ;
  return q + 1 ;
}


template <typename T> __global__ void
im2row_backward_kernel(T* data,
                        T const* stacked,
                       const int numPatchesX,
                       const int numPatchesY,
                       const int numPatchesXAn,
                       const int numPatchesYAn,
                       const int dataVolume,
                       const int width,
                       const int height,
                       const int widthAn,
                       const int heightAn,
                       const int depth,
                       const int windowWidth,
                       const int windowHeight,
                       const int windowWidthAn,
                       const int windowHeightAn,
                       const int strideX,
                       const int strideY,
                       const int strideXAn,
                       const int strideYAn,
                       const int padLeft,
                       const int padTop,
                       const int padLeftAn,
                       const int padTopAn,
                       const int gcdx, const int gcdy,
                       const int xbar, const int ybar,
                       const int ubar, const int vbar,
                       const int gcdxAn, const int gcdyAn,
                       const int xbarAn, const int ybarAn,
                       const int ubarAn, const int vbarAn)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < dataVolume)
  {
    T accumulator = 0 ;
    /*
     The goal of this kernel is to accumulate data[index]=data[x_data,y_data]
     all elements of the patch matrix that received copies of data[index] in the forward
     pass. To do this, we need to find which patches (x,y) that contain
     copies of this pixel and the relative offsets (u,v) within each such
     patch.

     First, we find which patches (x,y) contain copies of pixel (x_data,y_data)
     in the input tensor. The input tensor coordiante (x_data,y_data) of
     pixel  (u,v) in patch (x,y) are related by equations:

       x_data = x * strideX + u * dilateX - padLeft,
       y_data = y * strideY + v * dilateY - padTop.

     Now we find all values of (x,y) that can be generated by this equation.
     These gives us the patches (x,y) that must be summed. We have:

       strideX * x + dilateX * u = x_data + padLeft.

     where x and u are integers. This is a linear Diophantine equation.
     Rewrite it as:

       ax + bu = c, where

       a = strideX,
       b = dilateY,
       c = x_data + padLeft.

     This equation has a solution only if the greatest common divisor
     g = gcd(a,b) of a and b divides c as well. In this case,
     let (x0,u0) be a solution (i.e. a x0 + b u0 = c); all other solutions
     are in the form

       x_k = x0 + Dx * k,  Dx = b/g,
       u_k = u0 - Du * k,  Du = a/g.

     Next, we look for the values of k such that x_k and u_k are within
     bounds:

       1) 0 <= x_k <= Pw - 1
       2) 0 <= u_k <= Ww - 1

     Thus

       0) recall: gcd(a,b) must divide c
       1) ceil(- x0/Dx) <= k <= floor((Iw - 1 - x0)/Dx)
       2) ceil((u0 - Ww + 1)/Du) <= k <= floor(u0/Du)

     Thus we need to look for the k in the interval

       k_min = ceil(max(-x0/Dx, (u0 - Ww + 1)/Du)),
       k_max = floor(min((Pw - 1 - x0)/Dx,u0/Du).

     Toghether with (*) and the corresponding equations for y,
     this produces a list of patches (x_k,y_p) that contains
     pixel (x_data,y_data) (the list can be empty).

     Furthermore, x_data is mapped to a specific pixel in
     patch x_k whose coordiante is u_k, also given above.
     */

    int x_data = index ;
    int y_data = x_data / width ;
    int x_dataAn = y_data / height ;
    int y_dataAn = x_dataAn / widthAn ;
    int z = y_dataAn / heightAn ;
    x_data %= width ;
    y_data %= height ;
    x_dataAn %= widthAn ;
    y_dataAn %= heightAn ;

    int cx = x_data + padLeft ;
    int cy = y_data + padTop ;
    int cxAn = x_dataAn + padLeftAn ;
    int cyAn = y_dataAn + padTopAn ;
    int qx = cx / gcdx ;
    int qy = cy / gcdy ;
    int qxAn = cxAn / gcdxAn ;
    int qyAn = cyAn / gcdyAn ;

    if (cx != gcdx * qx || cy != gcdy * qy || cxAn != gcdxAn * qxAn || cyAn != gcdyAn * qyAn) { data[index] = 0 ; return ; }

    int x0 = xbar * qx ;
    int u0 = ubar * qx ;
    int y0 = ybar * qy ;
    int v0 = vbar * qy ;
    int x0An = xbarAn * qxAn ;
    int u0An = ubarAn * qxAn ;
    int y0An = ybarAn * qyAn ;
    int v0An = vbarAn * qyAn ;

    int Dx = 1 / gcdx ;
    int Du = strideX / gcdx ;
    int Dy = 1 / gcdy ;
    int Dv = strideY / gcdy ;
    int DxAn = 1 / gcdxAn ;
    int DuAn = strideXAn / gcdxAn ;
    int DyAn = 1 / gcdyAn ;
    int DvAn = strideYAn / gcdyAn ; 
    
    int kmin1 = ceildiv6D(-x0,Dx) ;
    int kmax1 = floordiv6D(numPatchesX - 1 - x0,Dx) ;
    int kmin2 = ceildiv6D(u0 - windowWidth + 1,Du) ;
    int kmax2 = floordiv6D(u0,Du) ;
    int kmin = max(kmin1,kmin2) ;
    int kmax = min(kmax1,kmax2) ;

    int qmin1 = ceildiv6D(-y0,Dy) ;
    int qmax1 = floordiv6D(numPatchesY - 1 - y0,Dy) ;
    int qmin2 = ceildiv6D(v0 - windowHeight + 1,Dv) ;
    int qmax2 = floordiv6D(v0,Dv) ;
    int qmin = max(qmin1,qmin2) ;
    int qmax = min(qmax1,qmax2) ;

    int kmin1An = ceildiv6D(-x0An,DxAn) ;
    int kmax1An = floordiv6D(numPatchesXAn - 1 - x0An,DxAn) ;
    int kmin2An = ceildiv6D(u0An - windowWidthAn + 1,DuAn) ;
    int kmax2An = floordiv6D(u0An,DuAn) ;
    int kminAn = max(kmin1An,kmin2An) ;
    int kmaxAn = min(kmax1An,kmax2An) ;

    int qmin1An = ceildiv6D(-y0An,DyAn) ;
    int qmax1An = floordiv6D(numPatchesYAn - 1 - y0An,DyAn) ;
    int qmin2An = ceildiv6D(v0An - windowHeightAn + 1,DvAn) ;
    int qmax2An = floordiv6D(v0An,DvAn) ;
    int qminAn = max(qmin1An,qmin2An) ;
    int qmaxAn = min(qmax1An,qmax2An) ;

    /*
     Now we have kmin <= k <= kmax, qmin <= q <= qmax and

     x_k = x0 + Dx * k,     u_k = u0 - Du * k,
     y_q = y0 + Dy * q,     v_q = v0 - Dv * q.

     Thus for each (k,q) in the allowable range, we visit
     patch (x_k,y_q) and pixel (u_k,v_q) within it.

     (x_k,y_q) tells us which row of the patch matix to look for, and
     (u_k,v_q) tells us which column. Linearizing all this:

     pm_row(k,q) = y_q * numPatchesX + x_k,
     pm_col(k,q) = ((z * windowHeight) + v_q) * windowWidth + u_k.

     This is further linearized into an index:

     pm_index(k,q) = (numPatchesX*numPatchesY) * pm_col(k,q) + pm_row(k,q)

     Substituting everything

     pm_row(k,q)
     = (y0 + Dy * q) * numPatchesX + x0 + Dx * k
     = (numPatchesX * Dy) * q + Dx * k + (y0 * numPatchesX + x0)
     = rqc * q + rkc * k + roc

     pm_col(k,q)
     = ((z * windowHeight) + v0 - Dv * q) * windowWidth + u0 - Du * k
     = - (windowWidth * Dv) * q - (Du) * k + (windowHeight * windowWidth * z + v0 * windowWidth + u0)
     = cqc * q + ckc * k + coc ;

     pm_index(k,q)
     = (numPatchesX*numPatchesY) * (cqc * q + ckc * k + coc) + rqc * q + rkc * k + roc
     = (numPatchesX*numPatchesY * cqc + rqc) * q + (numPatchesX*numPatchesY * ckc + rkc) * k + (numPatchesX*numPatchesY * coc + roc)
     = iqc * q + ikc * k + ioc

     */
    
    int rqcAn = DyAn * numPatchesXAn * numPatchesX * numPatchesY ;
    int rkcAn = DxAn * numPatchesX * numPatchesY ;
    int rqc = numPatchesX * Dy ;
    int rkc = Dx ;
    int roc = y0An * numPatchesXAn * numPatchesX * numPatchesY + x0An * numPatchesX * numPatchesY + numPatchesX * y0 + x0 ;
    
    int cqcAn =  - windowWidthAn * windowHeight * windowWidth * DvAn;
    int ckcAn = - windowHeight * windowWidth * DuAn;
    int cqc = - windowWidth * Dv ;
    int ckc = - Du ;
    int coc = (( windowWidthAn * (windowHeightAn * z + v0An) + u0An ) * windowHeight + v0) * windowWidth + u0;

    int np = numPatchesX * numPatchesY * numPatchesXAn * numPatchesYAn ;
    int iqcAn = np * cqcAn + rqcAn ;
    int ikcAn = np * ckcAn + rkcAn ;
    int iqc = np * cqc + rqc ;
    int ikc = np * ckc + rkc ;
    int ioc = np * coc + roc ;

    stacked += ioc ;
    for (int qAn = qminAn ; qAn <= qmaxAn ; ++ qAn) {
      for (int kAn = kminAn ; kAn <= kmaxAn ; ++ kAn) {
        for (int q = qmin ; q <= qmax ; ++ q) {
          for (int k = kmin ; k <= kmax ; ++ k) {
            accumulator += stacked[iqcAn * qAn + ikcAn * kAn + iqc * q + ikc * k] ;
          }
        }
      }
    }
    data[index] = accumulator;
  }
}

namespace vl { namespace impl {

  template<typename type>
  struct im2row6D<vl::VLDT_GPU, type>
  {

    /* ------------------------------------------------------------ */
    /*                                                      forward */
    /* ------------------------------------------------------------ */

    static vl::ErrorCode
    forward(Context & context,
            type* stacked,
            type const* data,
            size_t width,
            size_t height,
            size_t widthAn,
            size_t heightAn,
            size_t depth,
            size_t windowWidth,
            size_t windowHeight,
            size_t windowWidthAn,
            size_t windowHeightAn,
            size_t strideX,
            size_t strideY,
            size_t strideXAn,
            size_t strideYAn,
            size_t padLeft,
            size_t padRight,
            size_t padTop,
            size_t padBottom,
            size_t padLeftAn,
            size_t padRightAn,
            size_t padTopAn,
            size_t padBottomAn)
    {
      /* Each kernel instance copies a feature dimension of a patch */


      int numPatchesX = (width + (padLeft + padRight) - windowWidth)/strideX + 1 ;
      int numPatchesY = (height + (padTop + padBottom) - windowHeight)/strideY + 1 ;
      int numPatchesXAn = (widthAn + (padLeftAn + padRightAn) - windowWidthAn)/strideXAn + 1 ;
      int numPatchesYAn = (heightAn + (padTopAn + padBottomAn) - windowHeightAn)/strideYAn + 1 ;
      int numPatchSlices = numPatchesX * numPatchesY * numPatchesXAn * numPatchesYAn * depth ;

      im2row_forward_kernel<type>
      <<< divideAndRoundUp(numPatchSlices, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
      (stacked,
       data,
       numPatchesX,
       numPatchesY,
       numPatchesXAn,
       numPatchesYAn,
       numPatchSlices,
       width, height,
       widthAn, heightAn,
       windowWidth, windowHeight,
       windowWidthAn, windowHeightAn,
       strideX, strideY,
       strideXAn, strideYAn,
       padLeft, padTop,
       padLeftAn, padTopAn) ;

      return context.setError(context.getCudaHelper().catchCudaError(__func__)) ;
    }

    /* ------------------------------------------------------------ */
    /*                                                     backward */
    /* ------------------------------------------------------------ */

    static vl::ErrorCode
    backward(Context & context,
             type* data,
             type const* stacked,
            size_t width,
            size_t height,
            size_t widthAn,
            size_t heightAn,
            size_t depth,
            size_t windowWidth,
            size_t windowHeight,
            size_t windowWidthAn,
            size_t windowHeightAn,
            size_t strideX,
            size_t strideY,
            size_t strideXAn,
            size_t strideYAn,
            size_t padLeft,
            size_t padRight,
            size_t padTop,
            size_t padBottom,
            size_t padLeftAn,
            size_t padRightAn,
            size_t padTopAn,
            size_t padBottomAn)
    {
      /*
       Each kernel integrates all contributions to a particular element
       of data.
       */

      int numPatchesX = (width + (padLeft + padRight) - windowWidth)/strideX + 1 ;
      int numPatchesY = (height + (padTop + padBottom) - windowHeight)/strideY + 1 ;
      int numPatchesXAn = (widthAn + (padLeftAn + padRightAn) - windowWidthAn)/strideXAn + 1 ;
      int numPatchesYAn = (heightAn + (padTopAn + padBottomAn) - windowHeightAn)/strideYAn + 1 ;
      int dataVolume = width * height * widthAn * heightAn * depth ;

      int xbar ;
      int ubar ;
      int gcdx = vl::gcd(strideX, 1, xbar, ubar) ;

      int ybar ;
      int vbar ;
      int gcdy = vl::gcd(strideY, 1, ybar, vbar) ;

      int xbarAn ;
      int ubarAn ;
      int gcdxAn = vl::gcd(strideXAn, 1, xbarAn, ubarAn) ;

      int ybarAn ;
      int vbarAn ;
      int gcdyAn = vl::gcd(strideYAn, 1, ybarAn, vbarAn) ;
      
      im2row_backward_kernel<type>
      <<< divideAndRoundUp(dataVolume, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
      (data,
       stacked,
       numPatchesX,
       numPatchesY,
       numPatchesXAn,
       numPatchesYAn,
       dataVolume,
       width, height, widthAn, heightAn, depth,
       windowWidth, windowHeight,
       windowWidthAn, windowHeightAn,
       strideX, strideY,
       strideXAn, strideYAn,
       padLeft, padTop,
       padLeftAn, padTopAn,
       gcdx, gcdy, xbar, ybar, ubar, vbar,
       gcdxAn, gcdyAn, xbarAn, ybarAn, ubarAn, vbarAn) ;

      return context.setError(context.getCudaHelper().catchCudaError(__func__)) ;
    }

  } ;

} }

// Instantiations
template struct vl::impl::im2row6D<vl::VLDT_GPU, float> ;

#ifdef ENABLE_DOUBLE
template struct vl::impl::im2row6D<vl::VLDT_GPU, double> ;
#endif
