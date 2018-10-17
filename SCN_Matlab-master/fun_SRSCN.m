function im_h=fun_SRSCN(im,up_scale)
load('.\SCN_Matlab-master\weights_srnet_x2_52.mat');
% if max(im(:))>=20
    im = double(im)/255;
% else
%     im = double(im);
% end

[H,W,C] = size(im);
if C == 3
    im_l_ycbcr = rgb2ycbcr(im);
else
    im_l_ycbcr = zeros(H,W,C);
    im_l_ycbcr(:,:,1) = im;
    im_l_ycbcr(:,:,2) = im;
    im_l_ycbcr(:,:,3) = im;
end
im_l_y = im_l_ycbcr(:,:,1) * 255;

im_h_y = SCN(im_l_y,model,up_scale);


im_h_ycbcr = imresize(im_l_ycbcr,up_scale,'bicubic');
im_h_ycbcr(:,:,1) = im_h_y / 255.0;
im_h  = ycbcr2rgb(im_h_ycbcr) * 255.0;

