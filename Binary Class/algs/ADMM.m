
%% ADMM 
%%% Input：
%       |x0| is the initial point, |N| is the number of loss, |A| is the
%       sample matrix, |b| is the label, |fun| is used to compute the
%       function values and full gradient, |opts| contains the
%       hyperparameters
function [x,out] = ADMM(x0, N, A, b, fun,opts)
%%% Parameters
%       |opts.maxit| is the maximum number of iterations, |opts.rho| is the
%       penalty parameter, |opts.lambda| is the regularization parameter,
%       |opts.batchsize| is the mini-batch size in the stochastic
%       algorithms
if ~isfield(opts, 'maxit'); opts.maxit = 5000; end
if ~isfield(opts, 'rho'); opts.rho = 1; end 
if ~isfield(opts, 'lambda'); opts.lambda = 0.1; end
if ~isfield(opts, 'batchsize'); opts.batchsize = 10; end
if ~isfield(opts, 'verbose'); opts.verbose = 1; end

%%
%
x = x0;
out = struct();

[f_admm,  g_admm, acc_admm] = fun(x);
out.fvec = f_admm;
out.nrmG = norm(g_admm, 2);
out.acc = acc_admm;
out.epoch = 0;
out.cpu = 0;

[f_admm_iter, g_admm_iter, acc_admm_iter] = fun(x);
out.fvec_iter = f_admm_iter;
out.nrmG_iter = norm(g_admm_iter,2);
out.acc_iter = acc_admm_iter;
out.iter = 0;
out.cpu_iter = 0;

lambda = opts.lambda;
rho = opts.rho;   

z = zeros(size(x));    

count = 1;
iter = 1;
%% Update of ADMM
cpu_start = cputime;
for k = 1:opts.maxit
    
    % update y
    [y, ~] = yfun(lambda, z, x, rho);  

    % update x
    [x, ~] = xfun(N, b, A, z, y, rho); 

    % update z
    z = z + rho*(x - y);   
    
    % --------------------- iteration --------------------- %
    [f_admm_iter, g_admm_iter, acc_admm_iter] = fun(x);
    out.fvec_iter = [out.fvec_iter; f_admm_iter];
    out.nrmG_iter = [out.nrmG_iter; norm(g_admm_iter, 2)];
    out.acc_iter = [out.acc_iter; acc_admm_iter];
    out.iter = [out.iter; iter];
    cpu_end_iter = cputime;
    out.cpu_iter = [out.cpu_iter; cpu_end_iter - cpu_start];
    iter = iter + 1;

    % ----------------------- epoch ----------------------- %
    if k*opts.batchsize/N >= count
        [f_admm, g_admm, acc_admm] = fun(x);
        out.fvec = [out.fvec; f_admm];
        out.nrmG = [out.nrmG; norm(g_admm, 2)];
        out.acc = [out.acc; acc_admm];
        out.epoch = [out.epoch; k*opts.batchsize/N];
        cpu_end = cputime;
        out.cpu = [out.cpu; cpu_end - cpu_start];
        count = count + 1;
    end
end
end

%% Solve the subproblem of variable y
function [y_min, y_fval] = yfun(lambda, zt, xt, rho)
    
    fy = @(y) objectiveFunctiony(y, lambda, zt, xt, rho);
    y0 = rand(size(xt));    % random initialization
    % y0 = xt;    % fixed initialization
    % y0 = zeros(size(xt));    % fixed initialization 

    options = optimoptions('fminunc', 'Algorithm', 'quasi-newton', ...
                           'MaxIterations', 5, ...   
                           'SpecifyObjectiveGradient', true,...   
                           'Display', 'off');   
    [y_min, y_fval] = fminunc(fy, y0, options);
end

function [fy, gy] = objectiveFunctiony(y, lambda, zt, xt, rho)

    fy = lambda*sum(abs(y)./(1 + abs(y))) - sum(zt.*y) + 1/2*rho*norm(y - xt)^2;

    if nargout > 1
        gy = lambda*sign(y)./(1 + abs(y)).^2 - zt + rho*(y - xt);    % gradient  
    end
end

%% Solve the subproblem of variable x
function [x_min, x_fval] = xfun(N, b, A, zt, yt, rho)

    fx = @(x) objectiveFunctionx(x, N, b, A, zt, yt, rho);
    x0 = rand(size(zt));    % random initialization
    % x0 = zeros(size(zt));    % fixed initialization

    options = optimoptions('fminunc', 'Algorithm', 'quasi-newton', ...
                           'MaxIterations', 5, ...  
                           'SpecifyObjectiveGradient', true,...  
                           'Display', 'off');  
    [x_min, x_fval] = fminunc(fx, x0, options);
end

function [fx, gx] = objectiveFunctionx(x, N, b, A, zt, yt, rho)

    fx = sum(1./(1 + exp(b.*(A*x))))/N + sum(zt.*x) + 1/2*rho*norm(x - yt)^2;

    if nargout > 1
        b_expba = b.*exp(b.*(A*x));
        exp_square = (1 + exp(b.*(A*x))).^2;
        gx = A'* (-b_expba./exp_square)/N + zt + rho*(x - yt);    % gradient
    end
end
