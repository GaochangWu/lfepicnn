function [FullLF, inputLF] = fun_loadLytroLF(scenePath,sRes,tRes,angExtractionStart,angResOut,angResIn,toneCoef)
scale = round( (angResOut-1)/(angResIn-1) );
numImgsX = sRes;
numImgsY = tRes;
try
    inputImg = im2double(imread([scenePath,'.jpg']));
catch
    inputImg = im2double(imread([scenePath,'.png']));
end
inputImg = fun_adjustTone(inputImg,toneCoef);
h = size(inputImg, 1) / numImgsY;
w = size(inputImg, 2) / numImgsX;
FullLF = zeros(h, w, 3, numImgsY, numImgsX);
for ax = 1 : numImgsX
    for ay = 1 : numImgsY
        FullLF(:, :, :, ay, ax) = inputImg(ay:numImgsY:end, ax:numImgsX:end, :);
    end
end
FullLF=imresize(uint8(FullLF*255),1);

FullLF = FullLF(:, :, :, angExtractionStart:angExtractionStart+angResOut-1, angExtractionStart:angExtractionStart+angResOut-1);
inputLF = FullLF(:, :, :, [1:scale:angResOut], [1:scale:angResOut]);