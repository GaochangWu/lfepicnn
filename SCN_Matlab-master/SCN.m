function im_h_y = SCN(im_l_y, model,scale)

model_scale = 2;
patch_size = 5;
border_size = 6;
scale_y = 1.1;

iter_all = ceil(log(scale)/log(model_scale));
[lh,lw] = size(im_l_y);
for iter = 1:iter_all
    im_y = imresize(im_l_y,scale,'bicubic');
    im_y = padarray(im_y,[border_size,border_size],'symmetric');
    convfea = extrconvfea(im_y, model.conv);                               % extract convolution feature
    im_mean = extrconvfea(im_y,model.mean2);
    diffms = extrconvfea(im_y,model.diffms);
    [h,w,c] = size(convfea);
    convfea = reshape(convfea,h*w,c);
    convfea_norm = sqrt(sum(convfea.^2,2));
    convfea = convfea./repmat(convfea_norm,1,size(convfea,2));
    wd = convfea*model.wd;
    z0 = ShlU(wd,1);                                                       % formula: sign(a)(|a|-theta)
    z = ShlU(z0*model.usd1+wd,1);                                          % usd1: S
    
    hPatch  = z*model.ud;                                                  % ud: Dx
    hNorm = sqrt(sum(hPatch.^2,2));
    diffms = reshape(diffms,h*w,size(diffms,3));
    mNorm = sqrt(sum(diffms.^2,2));
    hPatch = hPatch./repmat(hNorm,1,size(hPatch,2)).*repmat(mNorm,1,size(diffms,2))*scale_y;
    hPatch = hPatch.*repmat(model.addp',size(hPatch,1),1);
    hPatch = reshape(hPatch,h,w,size(hPatch,2));
    im_h_y = im_mean;
    [h,w] = size(im_h_y);
    cnt = 1;
    for ii = patch_size:-1:1
        for jj = patch_size:-1:1
            im_h_y = im_h_y + hPatch(ii:ii+h-1,jj:jj+w-1,cnt);
            cnt = cnt + 1;
        end
    end
    im_l_y = im_h_y;
end

if size(im_h_y,1) > lh * scale
   im_h_y = imresize(im_h_y,[lh * scale,lw * scale],'bicubic');
end
end