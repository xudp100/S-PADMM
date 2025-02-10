
%% STOC-ADMM 
%%% Input：
%       |x0| is the initial point, |N| is the number of loss, |pgfun| is used to
%       compute the stochastic gradient, |fun| is used to compute the
%       function values and full gradient, |opts| contains the hyperparameters
function [x, out] = STOC_ADMM(x0, N, pgfun, fun, opts)
%%% Parameters
%       |opts.maxit| is the maximum number of iterations, |opts.alpha| is
%       the (initial) step size, |opts.rho| is the penalty parameter,
%       |opts.omega| is the coefficient of approximation matrix, |opts.lambda|
%       is the regularization parameter, |opts.step_type| is the type of
%       step size ('fixed', 'diminishing', 'hybrid'), |opts.batchsize| is
%       the mini-batch size in the stochastic algorithms
if ~isfield(opts, 'maxit'); opts.maxit = 5000; end
if ~isfield(opts, 'alpha'); opts.alpha = 1; end   
if ~isfield(opts, 'rho'); opts.rho = 0.9; end    
if ~isfield(opts, 'omega'); opts.omega = 1; end  
if ~isfield(opts, 'lambda'); opts.lambda = 0.1; end
if ~isfield(opts, 'step_type'); opts.step_type = 'fixed'; end
if ~isfield(opts, 'batchsize'); opts.batchsize = 10; end
if ~isfield(opts, 'verbose'); opts.verbose = 1; end

%%
%
x = x0;
out = struct();

[f_sadmm, g_sadmm, acc_sadmm] = fun(x);
out.fvec = f_sadmm;
out.nrmG = norm(g_sadmm, 2);
out.acc = acc_sadmm;
out.epoch = 0;
out.cpu = 0;

[f_sadmm_iter, g_sadmm_iter, acc_sadmm_iter] = fun(x);
out.fvec_iter = f_sadmm_iter;
out.nrmG_iter = norm(g_sadmm_iter, 2);
out.acc_iter = acc_sadmm_iter;
out.iter = 0;
out.cpu_iter = 0;

omega = opts.omega;
rho = opts.rho;
lambda = opts.lambda;

z = zeros(size(x));

count = 1;
iter = 1;
%% Update of STOC-ADMM
cpu_start = cputime;
for k = 1:opts.maxit
    
    % stochastic gradient 
    idx = randi(N, opts.batchsize, 1);
    g = pgfun(x,idx);  
    
    % update y
    [y, ~] = yfun(lambda, z, x, rho);
    
    % update x
    alpha = set_step(k, opts);
    h = omega*x - alpha*g - alpha*z + rho*alpha*y;
    x = h/(omega + rho*alpha);
    
    % update z
    z = z + rho*(x - y);
    
    % --------------------- iteration --------------------- %
    [f_sadmm_iter, g_sadmm_iter, acc_sadmm_iter] = fun(x);
    out.fvec_iter = [out.fvec_iter; f_sadmm_iter];
    out.nrmG_iter = [out.nrmG_iter; norm(g_sadmm_iter, 2)];
    out.acc_iter = [out.acc_iter; acc_sadmm_iter];
    out.iter = [out.iter; iter];
    cpu_end_iter = cputime;
    out.cpu_iter = [out.cpu_iter; cpu_end_iter - cpu_start];
    iter = iter + 1;

    % ----------------------- epoch ----------------------- %
    if k*opts.batchsize/N >= count
        [f_sadmm, g_sadmm, acc_sadmm] = fun(x);
        out.fvec = [out.fvec; f_sadmm];
        out.nrmG = [out.nrmG; norm(g_sadmm, 2)];
        out.acc = [out.acc; acc_sadmm];
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

%% Solve the subproblem of variable y
function [y_min, y_fval] = yfun(lambda, zt, xt, rho)

    fy = @(y) objectiveFunctiony(y, lambda, zt, xt, rho);
    y0 = rand(size(xt));    % random initialization
    % y0 = yt;    % fixed initialization
    % y0 = zeros(size(yt));    % fixed initialization

    options = optimoptions('fminunc', 'Algorithm', 'quasi-newton', ...
                           'MaxIterations', 5, ...  
                           'SpecifyObjectiveGradient', true,...  
                           'Display', 'off');  
    [y_min, y_fval] = fminunc(fy, y0, options);
end

function [fy, gy] = objectiveFunctiony(y, lambda, zt, xt, rho)

    fy = lambda*sum(sum(abs(y)./(1 + abs(y)))) - sum(sum(zt.*y)) + 1/2*rho*norm(y - xt, 'fro').^2;

    if nargout > 1
        gy = lambda*sign(y)./(1 + abs(y)).^2 - zt + rho*(y - xt);  
    end
end
