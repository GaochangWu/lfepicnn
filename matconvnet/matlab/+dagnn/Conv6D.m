classdef Conv6D < dagnn.Filter4D
  properties
    size = [0 0 0 0 0 0]
    hasBias = true
    opts = {'nocuDNN'}

  end

  methods
    function outputs = forward(obj, inputs, params)
      
      if ~obj.hasBias, params{2} = [] ; end
      %disp(size(inputs{1}));
      outputs{1} = vl_nnconv6D(...
        inputs{1}, params{1}, params{2}, ...
        'pad', obj.pad, ...
        'padAngular', obj.padAngular, ...
        'stride', obj.stride, ...
        'strideAngular', obj.strideAngular, ...
        obj.opts{:}) ;
      %disp(size(outputs{1}));
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      if ~obj.hasBias, params{2} = [] ; end
      [derInputs{1}, derParams{1}, derParams{2}] = vl_nnconv6D(...
        inputs{1}, params{1}, params{2}, derOutputs{1}, ...
        'pad', obj.pad, ...
        'padAngular', obj.padAngular, ...
        'stride', obj.stride, ...
        'strideAngular', obj.strideAngular, ...
        obj.opts{:}) ;
    end

    function kernelSize = getKernelSize(obj)
      kernelSize = obj.size(1:4) ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      outputSizes = getOutputSizes@dagnn.Filter(obj, inputSizes) ;
      outputSizes{1}(5) = obj.size(6) ;
    end

    function params = initParams(obj)
      % Xavier improved
      sc = sqrt(2 / prod(obj.size(1:5))) ;
      %sc = sqrt(2 / prod(obj.size([1 2 4]))) ;
      params{1} = randn(obj.size,'single') * sc ;
      if obj.hasBias
        params{2} = zeros(obj.size(6),1,'single') ;
      end
    end

    function set.size(obj, ksize)
      % make sure that ksize has 6 dimensions
      ksize = [ksize(:)' 1 1 1 1 1 1] ;
      obj.size = ksize(1:6) ;
    end

    function obj = Conv6D(varargin)
      obj.load(varargin) ;
      % normalize field by implicitly calling setters defined in
      % dagnn.Filter and here
      obj.size = obj.size ;
      obj.stride = obj.stride ;
      obj.pad = obj.pad ;
      obj.strideAngular = obj.strideAngular;
      obj.padAngular = obj.padAngular ;
    end
  end
end
