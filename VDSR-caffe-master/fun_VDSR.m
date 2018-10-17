function im_h=fun_VDSR(im_l,use_cascade,up_scale)
% The input image 'im_l' should be uint8 format, and the output 'im_h' is also uint8 format.

% run matconvnet/matlab/vl_setupnn;
% addpath('F:\Work\VDSR-caffe-master\Test\utils')

load('.\VDSR-caffe-master\Test\VDSR_Official.mat');
%load('.\VDSR-caffe-master\Test\VDSR_Adam.mat');

use_gpu = 0;

if use_gpu
    for i = 1:20
        model.weight{i} = gpuArray(model.weight{i});
        model.bias{i} = gpuArray(model.bias{i});
    end
end

im_l  = im2double(im_l);

[~,~,C] = size(im_l);
if C == 3
    im_l_ycbcr = rgb2ycbcr(im_l);
else
    im_l_ycbcr = im_l;
end
im_l_y = im_l_ycbcr(:,:,1);
if use_gpu
    im_l_y = gpuArray(im_l_y);
end

im_h_y = VDSR_Matconvnet(im_l_y, model,up_scale,use_cascade);

if use_gpu
    im_h_y = gather(im_h_y);
end
im_h_y = im_h_y * 255;
im_h_ycbcr = imresize(im_l_ycbcr,up_scale,'bicubic');
if C == 3
    im_h_ycbcr(:,:,1) = im_h_y / 255.0;
    im_h  = ycbcr2rgb(im_h_ycbcr) * 255.0;
else
    im_h = im_h_y;
end
im_h = uint8(im_h);
