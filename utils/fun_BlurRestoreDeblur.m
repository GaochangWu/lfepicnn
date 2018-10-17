function LFOut = fun_BlurRestoreDeblur(LFIn, LFDimOut, sizeBlurKernel)
[H,W,C,LFDimIn,~]=size(LFIn);
%% Parameters setting
upScale=fix((LFDimOut-1)/(LFDimIn-1));
if sizeBlurKernel>1
    load BlurKernel
    blurKernel=imresize(kernel,[3,sizeBlurKernel],'bicubic');
    blurKernel=blurKernel/(sum(blurKernel(:)));
else
    blurKernel=[0,1,0];
end
% Parameters for deblur
weight_ring = 1;
lambda_tv = 0.001;
lambda_l0 = 1e-4;
LFOut=uint8(zeros(H,W,C,LFDimOut,LFDimOut));
%% Iteration 1: Row Interpolation
fprintf('Begin iteration 1: Row Interpolation.\n');
for row=1:LFDimIn
    SliceIn=LFIn(:,:,:,row,:);
    SliceIn=permute(SliceIn,[1,2,3,5,4]);
    
    % "Blur-Restoration" step
    SliceOut=fun_InterpKernel(SliceIn,LFDimOut,LFDimIn,blurKernel);
    
    % "Deblur" step
    parfor i=1:LFDimOut
        im=double(SliceOut(:,:,:,i))/255;
        if sizeBlurKernel>1
            im = ringing_artifacts_removal(im, blurKernel, lambda_tv, lambda_l0, weight_ring);
        end
        SliceOut(:,:,:,i)=uint8(im*255);
    end
    SliceOut=permute(SliceOut,[1,2,3,5,4]);
    LFOut(:,:,:,(row-1)*upScale+1,:)=SliceOut;
    fprintf('  Row %d has been processed.\n',row);
end
%% Iteration 1: Colume Interpolation
fprintf('Begin iteration 1: Colume Interpolation.\n');
for colume=1:LFDimIn
    SliceIn=LFIn(:,:,:,:,colume);
    SliceIn=permute(SliceIn,[2,1,3,4]);
    
    % "Blur-Restoration" step
    SliceOut=fun_InterpKernel(SliceIn,LFDimOut,LFDimIn,blurKernel);
    
    % "Deblur" step
    parfor i=1:LFDimOut
        im=double(SliceOut(:,:,:,i))/255;
        if sizeBlurKernel>1
            im = ringing_artifacts_removal(im, blurKernel, lambda_tv, lambda_l0, weight_ring);
        end
        SliceOut(:,:,:,i)=uint8(im*255);
    end
    SliceOut=permute(SliceOut,[2,1,3,4]);
    LFOut(:,:,:,:,(colume-1)*upScale+1)=SliceOut;
    fprintf('  Colume %d has been processed.\n',colume);
end
%% Iteration 2: Row Interpolation
fprintf('Begin iteration 2: Row Interpolation.\n');
idxInterp=1:LFDimOut;
idxInterp=setdiff(idxInterp,([1:LFDimIn]-1)*upScale+1);
for row=idxInterp
    SliceIn=LFOut(:,:,:,row,1:upScale:LFDimOut);
    SliceIn=permute(SliceIn,[1,2,3,5,4]);
    
    % "Blur-Restoration" step
    SliceOut=fun_InterpKernel(SliceIn,LFDimOut,LFDimIn,blurKernel);
    
    % "Deblur" step
    parfor i=1:LFDimOut
        im=double(SliceOut(:,:,:,i))/255;
        if sizeBlurKernel>1
            im = ringing_artifacts_removal(im, blurKernel, lambda_tv, lambda_l0, weight_ring);
        end
        SliceOut(:,:,:,i)=uint8(im*255);
    end
    SliceOut=permute(SliceOut,[1,2,3,5,4]);
    LFOut(:,:,:,row,:)=SliceOut;
    fprintf('  Row %d has been processed.\n',row);
end