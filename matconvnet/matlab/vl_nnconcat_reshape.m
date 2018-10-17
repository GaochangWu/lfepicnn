function y = vl_nnconcat_reshape(inputs, dim, dzdy, varargin)
%VL_NNCONCAT CNN concatenate multiple inputs.
%  Y = VL_NNCONCAT(INPUTS, DIM) concatenates the inputs in the cell
%  array INPUTS along dimension DIM generating an output Y.
%
%  DZDINPUTS = VL_NNCONCAT(INPUTS, DIM, DZDY) computes the derivatives
%  of the block projected onto DZDY. DZDINPUTS has one element for
%  each element of INPUTS, each of which is an array that has the same
%  dimensions of the corresponding array in INPUTS.

% Copyright (C) 2015 Karel Lenc and Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

opts.inputSizes = [] ;
opts = vl_argparse(opts, varargin, 'nonrecursive') ;

if nargin < 2, dim = 5; end;
if nargin < 3, dzdy = []; end;

if isempty(dzdy)
  inputs{2} = reshape(inputs{2}, [size(inputs{2}, 1),size(inputs{2}, 2),1,1,9,1]);
  y = cat(5, inputs{:});
  y = y(:,:,:,:,[41,1:2,42,3:4,43,5:18,44,19:20,45,21:22,46,23:36,47,37:38,48,39:40,49],:);
  y = reshape(y, [size(inputs{2}, 1),size(inputs{2}, 2),7,7,1,1]);
  %disp(size(y));
else
  %if isempty(opts.inputSizes)
  %  opts.inputSizes = cellfun(@(inp) [size(inp,1),size(inp,2),size(inp,3),size(inp,4)], inputs, 'UniformOutput', false) ;
  %end
  [h,w,~] = size(dzdy);
  dzdy = reshape(dzdy, [h,w,1,1,49,1]);
  dzdy = dzdy(:,:,:,:,[2:3,5:6,8:21,23:24,26:27,29:42,44:45,47:48,1,4,7,22,25,28,43,46,49],:);
  opts.inputSizes{1} = [h,w,1,1,40,1];
  opts.inputSizes{2} = [h,w,1,1,9,1];
  start = 1 ;
  y = cell(1, numel(opts.inputSizes)) ;
  s.type = '()' ;
  s.subs = {':', ':', ':', ':',':',':'} ;
  for i = 1:numel(opts.inputSizes)
    stop = start + opts.inputSizes{i}(dim) ;
    s.subs{dim} = start:stop-1 ;
    y{i} = subsref(dzdy,s) ;
    start = stop ;
  end
  y{1} = reshape(y{1}, [h,w,1,1,40,1]);
end
