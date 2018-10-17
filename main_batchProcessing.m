clear;clc;close all;
% Read Lytro data, please download the data at
% http://lightfields.stanford.edu/
LFDimIn = 3;
LFDimOut = 7;
toneCoef = 2.25;

addpath('./utils');
sceneFile='.\Data';
im_list = dir(strcat(sceneFile,'/*.png'));
for n = 1:length(im_list)
    sceneName = im_list(n).name;
    sceneName = sceneName(1:end-4);
    sRes = 14;
    tRes = 14;
    angExtractionStart = 5;
    sizeBlurKernel = 3;
    % Extract sub-aperture images
    fprintf(['Working on scene ',sceneName,'(%0d of %0d). Loading light field ...'], n, length(im_list));
    [LFGT, LFIn] = fun_loadLytroLF(strcat(sceneFile,'/',sceneName), sRes, tRes, angExtractionStart, LFDimOut, LFDimIn, toneCoef);
    fprintf(['Done!\n']);
    [H,W,C,~,~]=size(LFGT);
    figure(1);imshow(mean(mean(im2double(LFIn),4),5));
%% "Blur-restoration-deblur" framework
    LFOut = fun_BlurRestoreDeblur(LFIn, LFDimOut, sizeBlurKernel);
%% Evaluation
    MakeDir(['./Results/',sceneName,'/images']);
    fid = fopen(['./Results/',sceneName,'/Log.txt'], 'wt');
    LFDimOut=size(LFOut,4);
    down_scale = fix( (LFDimOut-1)/(LFDimIn-1) );
    idxOut = [1:down_scale:LFDimOut];
    borderCut = 22;
    PSNR=0;
    SSIM=0;
    k=0;
    for i=1:LFDimOut
        for j=1:LFDimOut
            if ismember(i,idxOut) && ismember(j,idxOut)
            else
                k=k+1;
                [psnr(i,j),ssim(i,j)]=compute_psnr(LFOut(:,:,:,i,j), LFGT(:,:,:,i,j), borderCut);
                PSNR=PSNR+psnr(i,j);
                SSIM=SSIM+ssim(i,j);
            end
    %         imwrite(LFOut(:,:,:,i,j),['./Results/',sceneName,'/images/',sprintf('%02d',i),'_',sprintf('%02d',j),'.png']);
        end
    end
    fprintf('Mean PSNR and SSIM on synthetic views are: %2.2f, %0.4f\n', PSNR/k, SSIM/k);
    fprintf(fid, 'Mean PSNR and SSIM on synthetic views are: %2.2f, %0.4f\n', PSNR/k, SSIM/k);
end