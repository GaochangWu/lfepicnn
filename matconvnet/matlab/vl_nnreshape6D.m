function y = vl_nnreshape6D(x, dim, sp2an, dzdy)
% VL_NNRESHAPE Feature reshaping
%   Y = VL_NNRESHAPE(X, DIMS) reshapes the input data X to have
%   the dimensions specified by DIMS. X is a SINGLE array of 
%   dimension H x W x D x N where (H,W) are the height and width of 
%   the map stack, D is the image depth (number of feature channels) 
%   and N the number of of images in the stack. DIMS is a 1 x 3 array
%   of integers describing the dimensions that Y will take (batch 
%   size is preserved). In addition to positive integers, the 
%   following can also be specified in the style of caffe:
%
%   Interpretation of DIMS elements:
%   -1 := work it out from other dims
%    0 := copy dimension from X
%
%   NOTE: At most one dimension can be worked out from the others.
%
%   DZDX = VL_NNRESHAPE(X, DIMS, DZDY) computes the derivatives of the
%   block projected onto DZDY. DZDX and DZDY have the same dimensions
%   as X and Y respectively.

%assert(sum(dims == -1) <= 1, 'at most one dim can be computed from the others') ;
assert(length(dim) == 1, 'dims should have only one elements') ;
dim = dim(1);
%sz = size(x) ;

%copyDims = find(dims == 0) ;
%if copyDims
%    dims(copyDims) = sz(copyDims) ;
%end
%
%targetDim = find(dims == -1) ;
%if targetDim
%    idx = [1 2 3] ;
%    idx(targetDim) = [] ;
%    dims(targetDim) = prod(sz(1:3)) / prod(dims(idx)) ;
%end

%dims = horzcat(dims, size(x,4)) ;

if sp2an
    if nargin < 4 || isempty(dzdy)
        [h,w,c,n] = size(x);
        x_tmp = permute(x, [1,2,4,3]);
        y = reshape(x_tmp, [h, w, dim, dim, c, 1]);
    else
        [h, w, dim1, dim2, c, n] = size(dzdy);
        der_tmp = reshape(dzdy, [h, w, dim1*dim2, c]) ;
        y = permute(der_tmp, [1,2,4,3]);
    end
else
    if nargin < 4 || isempty(dzdy)
        [h, w, dim1, dim2, c, n] = size(x);
        x_tmp = reshape(x, [h, w, dim1*dim2, c]);
        y = permute(x_tmp, [1,2,4,3]);
    else
        [h,w,c,n] = size(dzdy);
        der_tmp = permute(dzdy, [1,2,4,3]);
        y = reshape(der_tmp, [h, w, dim, dim, c, 1]);
    end
end
