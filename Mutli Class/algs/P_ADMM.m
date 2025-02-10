
%% PADMM 
%%% Input：
%       |x0| is the initial point, |N| is the number of losses, |A| is the
%       sample matrix, |b| is the label, |K| is the number of classes, 
%       |fun| is used to compute the function values and full gradient, 
%       |opts| contains the hyperparameters
function [x, out] = P_ADMM(x0, N, A, b, K, fun, opts)
%%% Parameters
%       |opts.maxit| is the maximum number of iterations, |opts.rho| is the
%       penalty parameter, |opts.gamma| is the coefficient of proximal matrix, 
%       |opts.lambda| is the regularization parameter, |opts.batchsize| is
%       the mini-batch size in the stochastic algorithms
if ~isfield(opts, 'maxit'); opts.maxit = 5000; end
if ~isfield(opts, 'rho'); opts.rho = 1; end   
if ~isfield(opts, 'gamma'); opts.gamma = 2; end
if ~isfield(opts, 'lambda'); opts.lambda = 0.1; end
if ~isfield(opts, 'batchsize'); opts.batchsize = 10; end
if ~isfield(opts, 'verbose'); opts.verbose = 1; end

%%
% 
x = x0;
out = struct();

[f_padmm, g_padmm, acc_padmm] = fun(x);
out.fvec = f_padmm;
out.nrmG = norm(g_padmm, 2);
out.acc = acc_padmm;
out.epoch = 0;
out.cpu = 0;

[f_padmm_iter, g_padmm_iter, acc_padmm_iter] = fun(x);
out.fvec_iter = f_padmm_iter;
out.nrmG_iter = norm(g_padmm_iter, 2);
out.acc_iter = acc_padmm_iter;
out.iter = 0;
out.cpu_iter = 0;

lambda = opts.lambda;
rho = opts.rho;    
gamma = opts.gamma;   
B_hot =full(ind2vec(b',K))' ;

y = zeros(size(x));    
z = zeros(size(x));   

count = 1;
iter = 1;
%% Update of P-ADMM
cpu_start = cputime;
for k = 1:opts.maxit
    
    % update y
    [y, ~] = yfun(lambda, z, x, y, rho,gamma);

    % update x
    [x, ~] = xfun(N, B_hot, A, z, y, rho);

    % update z
    z = z + rho*(x - y);
    
    % --------------------- iteration --------------------- %
    [f_padmm_iter,g_padmm_iter,acc_padmm_iter] = fun(x);
    out.fvec_iter = [out.fvec_iter; f_padmm_iter];
    out.nrmG_iter = [out.nrmG_iter; norm(g_padmm_iter,2)];
    out.acc_iter = [out.acc_iter; acc_padmm_iter];
    out.iter = [out.iter; iter];
    cpu_end_iter = cputime;
    out.cpu_iter = [out.cpu_iter; cpu_end_iter - cpu_start];
    iter = iter +1;

    % ----------------------- epoch ----------------------- %
    if k*opts.batchsize/N >= count
       [f_padmm, g_padmm, acc_padmm] = fun(x);
       out.fvec = [out.fvec; f_padmm];
       out.nrmG = [out.nrmG; norm(g_padmm, 2)];
       out.acc = [out.acc; acc_padmm];
       out.epoch = [out.epoch; k*opts.batchsize/N];
       cpu_end = cputime;
       out.cpu = [out.cpu; cpu_end - cpu_start];
       count = count + 1;
    end
end
end


%% Solve the subproblem of variable y
function [y_min, y_fval] = yfun(lambda, zt, xt, yt, rho, gamma)

    fy = @(y) objectiveFunctiony(y, lambda, zt, xt, yt, rho, gamma);
    y0 = rand(size(xt));    % random initialization
    % y0 = yt;    % fixed initialization
    % y0 = zeros(size(yt));    % fixed initialization   

    options = optimoptions('fminunc', 'Algorithm', 'quasi-newton', ...
                           'MaxIterations', 5, ...  
                           'SpecifyObjectiveGradient', true,... 
                           'Display', 'off'); 
    [y_min, y_fval] = fminunc(fy, y0, options);
end

function [fy, gy] = objectiveFunctiony(y, lambda, zt, xt, yt, rho, gamma)

    fy = lambda*sum(sum(abs(y)./(1 + abs(y)))) - sum(sum(zt.*y)) + 1/2*rho*norm(y - xt, 'fro').^2 + 1/2*gamma*norm(y - yt, 'fro')^2;

    if nargout > 1
        gy = lambda*sign(y)./(1 + abs(y)).^2 - zt + rho * (y - xt) + gamma*(y-yt);    % gradient 
    end
end

%% Solve the subproblem of variable x
function [x_min, x_fval] = xfun(N, B_hot, A, zt, yt1, rho)

    fx = @(x) objectiveFunctionx(x, N, B_hot, A, zt, yt1, rho);
    x0 = rand(size(zt));    % random initialization
    % x0 = zeros(size(zt));    % fixed initialization  

    options = optimoptions('fminunc', 'Algorithm', 'quasi-newton', ...
                           'MaxIterations', 5, ...  
                           'SpecifyObjectiveGradient', true,...  
                           'Display', 'off');  
    [x_min, x_fval] = fminunc(fx, x0, options);
end

function [fx, gx] = objectiveFunctionx(x, N, B_hot, A, zt, yt, rho)

    fx = -(1/N)*sum(sum(B_hot.*log(softmax((A*x)')'))) + sum(sum(zt.*x)) + 1/2*rho*norm(x-yt, 'fro')^2;

    if nargout > 1
        B_pred = softmax((A*x)')';
        diff = B_pred - B_hot;
        gx = A'*diff/N + zt + rho*(x - yt);    % gradient  
    end
end


