
%% SGD
%%% Input：
%       |x0| is the initial point, |N| is the number of loss, |pgfun| is used to
%       compute the stochastic gradient, |fun| is used to compute the
%       function values and full gradient, |opts| contains the hyperparameters
function [x, out] = SGD(x0, N, pgfun, fun, opts)
%%% Parameters
%       |opts.maxit| is the maximum number of iterations, |opts.alpha| is
%       the (initial) step size, |opts.lambda| is the regularization
%       parameter, |opts.step_type| is the type of step size ('fixed',
%       'diminishing', 'hybrid'), |opts.batchsize| is the mini-batch size
%       in the stochastic algorithms 
if ~isfield(opts, 'maxit'); opts.maxit = 5000; end
if ~isfield(opts, 'alpha'); opts.alpha = 1e-2; end
if ~isfield(opts, 'lambda'); opts.lambda = 0.1; end
if ~isfield(opts, 'step_type'); opts.step_type = 'fixed'; end
if ~isfield(opts, 'batchsize'); opts.batchsize = 10; end
if ~isfield(opts, 'verbose'); opts.verbose = 1; end


%%
%
x = x0;
out = struct();

[f_sgd, g_sgd, acc_sgd] = fun(x);
out.fvec = f_sgd;    
out.nrmG = norm(g_sgd, 2);
out.acc = acc_sgd;
out.epoch = 0;
out.cpu = 0;

[f_sgd_iter, g_sgd_iter, acc_sgd_iter] = fun(x);
out.fvec_iter = f_sgd_iter;
out.nrmG_iter = norm(g_sgd_iter, 2);
out.acc_iter = acc_sgd_iter;
out.iter = 0;
out.cpu_iter = 0;

lambda = opts.lambda;

count = 1;
iter = 1;
%% Update of SGD
cpu_start = cputime; 
for k = 1:opts.maxit
    
    idx = randi(N, opts.batchsize, 1);
    g = pgfun(x, idx) + lambda*sign(x)./((1 + abs(x)).^2);
    
    alpha = set_step(k, opts);
    x = x - alpha*g;

    % --------------------- iteration --------------------- %
    [f_sgd_iter, g_sgd_iter, acc_sgd_iter] = fun(x);
    out.fvec_iter = [out.fvec_iter; f_sgd_iter];
    out.nrmG_iter = [out.nrmG_iter; norm(g_sgd_iter, 2)];
    out.acc_iter = [out.acc_iter; acc_sgd_iter];
    out.iter = [out.iter; iter];
    cpu_end_iter = cputime;
    out.cpu_iter = [out.cpu_iter; cpu_end_iter - cpu_start];
    iter = iter +1;

    % ----------------------- epoch ----------------------- %
    if k*opts.batchsize/N >= count
        [f_sgd, g_sgd, acc_sgd] = fun(x);
        out.fvec = [out.fvec; f_sgd];
        out.acc = [out.acc; acc_sgd];
        out.nrmG = [out.nrmG; norm(g_sgd, 2)];
        out.epoch = [out.epoch; k*opts.batchsize/N];
        cpu_end = cputime;
        out.cpu = [out.cpu; cpu_end - cpu_start];
        count = count + 1;
    end
end
end

%% Step size
% |opts.step_type| is the type of step size ('fixed', 'diminishing' or 'hybrid')
function a = set_step(k, opts)
type = opts.step_type;
zeta = 0.1;
thres_dimin = 100;
if strcmp(type, 'fixed')
    a = opts.alpha;
elseif strcmp(type, 'diminishing')
    a = opts.alpha/(1 + opts.alpha*zeta*k);
elseif strcmp(type, 'hybrid')
    if k < thres_dimin
        a = opts.alpha/(1 + 40*zeta*k);
    else
        a = opts.alpha/(1 + 40*zeta*thres_dimin);
    end
else
    error('unsupported type.');
end
end
