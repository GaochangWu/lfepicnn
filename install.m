addpath(genpath('./matconvnet'));
try
    cmd = 'cp -a vllab_dag_loss.m matconvnet/matlab/+dagnn';
catch err
    cmd = 'copy vllab_dag_loss.m matconvnet/matlab/+dagnn';
end
fprintf('%s\n', cmd);
system(cmd);

vl_compilenn('enableGPu', false, 'enableCudnn', false);
