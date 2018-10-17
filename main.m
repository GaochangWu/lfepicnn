clear;clc;close all;
% Read Lytro data, please download the data at
% http://lightfields.stanford.edu/
sceneName='occlusions_16_eslf';
toneCoef = 1.5;
LFDimOut = 7; % Output resolution in angular dimension
LFDimIn = 3; % Input resolution in angular dimension

addpath('./utils');
fileName=['.\Data\',sceneName];
sRes = 14;
tRes = 14;
angExtractionStart = 5;
down_scale = (LFDimOut-1)/(LFDimIn-1);
sizeBlurKernel = 3;
% Extract sub-aperture images
fprintf(['Working on scene ',sceneName,'. Loading light field ...']);
[LFGT, LFIn] = fun_loadLytroLF(fileName, sRes, tRes, angExtractionStart, LFDimOut, LFDimIn, toneCoef);
fprintf(['Done!\n']);
[H,W,C,~,~]=size(LFGT);
figure(1);imshow(mean(mean(im2double(LFIn),4),5));
%% "Blur-restoration-deblur" framework
LFOut = fun_BlurRestoreDeblur(LFIn, LFDimOut, sizeBlurKernel);
%% Display
for i=1:LFDimOut
    if mod(i,2)==1
        displaySequence=1:LFDimOut;
    else
        displaySequence=LFDimOut:-1:1;
    end
    for j=displaySequence
        figure(1);imshow(LFOut(:,:,:,i,j));
    end
end
%% Evaluation
MakeDir(['./Results/',sceneName,'/images']);
fid = fopen(['./Results/',sceneName,'/Log.txt'], 'wt');
LFDimOut=size(LFOut,4);
idxOut = [1:down_scale:LFDimOut];
borderCut = 0;
PSNR=0;
SSIM=0;
k=0;
for i=1:LFDimOut
    for j=1:LFDimOut
            k=k+1;
            [psnr(i,j),ssim(i,j)]=compute_psnr(LFOut(:,:,:,i,j), LFGT(:,:,:,i,j), borderCut);
            PSNR=PSNR+psnr(i,j);
            SSIM=SSIM+ssim(i,j);
%         imwrite(LFOut(:,:,:,i,j),['./Results/',sceneName,'/images/',sprintf('%02d',i),'_',sprintf('%02d',j),'.png']);
    end
end
fprintf('Mean PSNR and SSIM on all the sub-aperture images are: %2.2f, %0.4f\n', PSNR/k, SSIM/k);
fprintf(fid, 'Mean PSNR and SSIM on all the sub-aperture images are: %2.2f, %0.4f\n', PSNR/k, SSIM/k);