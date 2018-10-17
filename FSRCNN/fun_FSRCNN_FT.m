function im_h=fun_FSRCNN_FT(data)
load('.\FSRCNN\FSRCNN_modelFT.mat');

[H,W,C]=size(data);
curEPI=double(data)/255;

if C == 3
    im_l_ycbcr = rgb2ycbcr(curEPI);
else
    im_l_ycbcr = zeros(H,W,C);
    im_l_ycbcr(:,:,1) = curEPI;
    im_l_ycbcr(:,:,2) = curEPI;
    im_l_ycbcr(:,:,3) = curEPI;
end
im_l_y = im_l_ycbcr(:,:,1);
im_l_ycbcr = imresize(im_l_ycbcr,[(H-1)*3+1,W],'bicubic');

im_l_y=single(im_l_y);
k=1;
% Feature extraction: conv layer
layer = vl_nnconv(im_l_y, weight{k}, bias{k}, 'pad', 2, 'stride', 1, 'NoCudnn');
layer = max(layer,0)+ prelu(k)*min(layer,0);
k=k+1;
% Shrinking: conv layer
layer = vl_nnconv(layer, weight{k}, bias{k}, 'pad', 0, 'stride', 1, 'NoCudnn');
layer = max(layer,0)+ prelu(k)*min(layer,0) ;
k=k+1;
% Mapping: conv layer
for i = 1:4
    layer = vl_nnconv(layer, weight{k}, bias{k}, 'pad', 1, 'stride', 1, 'NoCudnn');
    layer = max(layer,0)+ prelu(k)*min(layer,0) ;
    k=k+1;
end
% Expanding: conv layer
layer = vl_nnconv(layer, weight{k}, bias{k}, 'pad', 0, 'stride', 1, 'NoCudnn');
layer = max(layer,0)+ prelu(k)*min(layer,0) ;
k=k+1;
% Deconvolution: deconv layer
layer = vl_nnconvt(layer, weight{k}, bias{k}, 'Upsample', [3 1], 'Crop', [round((size(weight{k},1)-3)/2)+1, round((size(weight{k},2)-3)/2)+1, 4, 4]);

im_l_ycbcr(:,:,1) = double(layer);
im_h = ycbcr2rgb(im_l_ycbcr);
im_h = im_h*255;
