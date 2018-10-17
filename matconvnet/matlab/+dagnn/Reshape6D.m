classdef Reshape6D < dagnn.Layer
    %RESHAPE Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        dims = []
        sp2an = true
    end
    
    methods
        function outputs = forward(obj, inputs, params)
            outputs{1} = vl_nnreshape6D(inputs{1}, obj.dims, obj.sp2an);
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            derInputs{1} = vl_nnreshape6D(inputs, obj.dims, obj.sp2an, derOutputs{1});
            derParams = {} ;
        end
        
        function obj = Reshape6D(varargin)
          obj.load(varargin);
        end
    end
    
end

