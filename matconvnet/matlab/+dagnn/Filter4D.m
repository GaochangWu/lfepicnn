classdef Filter4D < dagnn.Layer
  properties
    pad = [0 0 0 0]
    stride = [1 1]
    padAngular = [0 0 0 0]
    strideAngular = [1 1]
  end
  methods
    function set.pad(obj, pad)
      if numel(pad) == 1
        obj.pad = [pad pad pad pad] ;
      elseif numel(pad) == 2
        obj.pad = pad([1 1 2 2]) ;
      else
        obj.pad = pad ;
      end
    end

    function set.stride(obj, stride)
      if numel(stride) == 1
        obj.stride = [stride stride] ;
      else
        obj.stride = stride ;
      end
    end
    
    function set.padAngular(obj, padAngular)
      if numel(padAngular) == 1
        obj.padAngular = [padAngular padAngular padAngular padAngular] ;
      elseif numel(padAngular) == 2
        obj.padAngular = padAngular([1 1 2 2]) ;
      else
        obj.padAngular = padAngular ;
      end
    end

    function set.strideAngular(obj, strideAngular)
      if numel(strideAngular) == 1
        obj.strideAngular = [strideAngular strideAngular] ;
      else
        obj.strideAngular = strideAngular ;
      end
    end

    function kernelSize = getKernelSize(obj)
      kernelSize = [1 1 1 1] ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      ke = obj.getKernelSize() ;
      outputSizes{1} = [...
        fix((inputSizes{1}(1) + obj.pad(1) + obj.pad(2) - ke(1)) / obj.stride(1)) + 1, ...
        fix((inputSizes{1}(2) + obj.pad(3) + obj.pad(4) - ke(2)) / obj.stride(2)) + 1, ...
        fix((inputSizes{1}(3) + obj.padAngular(1) + obj.padAngular(2) - ke(3)) / obj.strideAngular(1)) + 1, ...
        fix((inputSizes{1}(4) + obj.padAngular(3) + obj.padAngular(4) - ke(4)) / obj.strideAngular(2)) + 1, ...
        1, ...
        inputSizes{1}(6)] ;
    end

    function rfs = getReceptiveFields(obj)
      ke = obj.getKernelSize() ;
      y1 = 1 - obj.pad(1) ;
      y2 = 1 - obj.pad(1) + ke(1) - 1 ;
      x1 = 1 - obj.pad(3) ;
      x2 = 1 - obj.pad(3) + ke(2) - 1 ;
      h = y2 - y1 + 1 ;
      w = x2 - x1 + 1 ;
      
      y1An = 1 - obj.padpadAngular(1) ;
      y2An = 1 - obj.padpadAngular(1) + ke(3) - 1 ;
      x1An = 1 - obj.padpadAngular(3) ;
      x2An = 1 - obj.padpadAngular(3) + ke(4) - 1 ;
      hAn = y2An - y1An + 1 ;
      wAn = x2An - x1An + 1 ;
      
      rfs.size = [h, w, hAn, wAn] ;
      rfs.stride = obj.stride ;
      rfs.strideAngular = obj.strideAngular ;
      rfs.offset = [y1+y2, x1+x2, y1An+y2An, x1An+x2An]/2 ;
    end
  end
end
