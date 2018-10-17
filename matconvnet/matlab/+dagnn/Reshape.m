classdef Reshape < dagnn.Layer
    %RESHAPE Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        dims = []
        sp2an = true
        h = 32
        w = 32
    end
    
    methods
        function outputs = forward(obj, inputs, params)
            outputs{1} = vl_nnreshape(inputs{1}, obj.dims, obj.sp2an, obj.h, obj.w);
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            derInputs{1} = vl_nnreshape(inputs, obj.dims, obj.sp2an, obj.h, obj.w, derOutputs{1});
            derParams = {} ;
        end
        
        function obj = Reshape(varargin)
          obj.load(varargin);
        end
    end
    
end

