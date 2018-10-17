function SliceOut=fun_InterpKernel(SliceIn,LFDimOut,LFDimIn,blurKernel)
[H,W,C,~]=size(SliceIn);
upScale=fix(LFDimOut/LFDimIn)+1;

% "Blur" step
SliceBlured=zeros(H,W,C,LFDimIn);
sizeKernel=size(blurKernel,2);
for i=1:LFDimIn
    im=SliceIn(:,:,:,i);
    if sizeKernel>1
        im=imfilter(im,blurKernel,'conv','circular');
    end
    SliceBlured(:,:,:,i)=im;
end

% "Restoration" step
SliceOut=zeros(H,LFDimOut,W,C);
parfor i=1:H
    EPIIn=SliceBlured(i,:,:,:);
    EPIIn=permute(uint8(EPIIn),[4,2,3,1]);
    
%---------Select a restoration kernel----------
for k=1:ceil(log(upScale)/log(3))
%%% EPICNN
%     addpath('./EPICNN');
%     EPIIn=fun_EPICNN(EPIIn);
%%% FSRCNN
    addpath('./FSRCNN');
    EPIIn=fun_FSRCNN_FT(EPIIn);
%%% SRCNN
%     addpath('./SRCNN');
%     EPIIn=fun_SRCNN(EPIIn);
%%% SCN
%     addpath('./SCN_Matlab-master');
%     EPIIn=fun_SRSCN(EPIIn,upScale);
%%% VDSR
%     addpath('./VDSR-caffe-master');
%     EPIIn=fun_VDSR(EPIIn,0,upScale);
%%% Sparse coding
%     addpath(genpath('./ScSR'));
%     EPIIn=fun_SRSC(EPIIn,upScale);
end
%----------------------------------------------
    EPIOut=imresize(EPIIn,[LFDimOut,W],'bicubic');
    SliceOut(i,:,:,:)=uint8(EPIOut);
end
SliceOut=permute(SliceOut,[1,3,4,2]);