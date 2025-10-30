%% S-PADMM 
function [x, out] = SPADMM(x0, L, N, pgfun, fun, opts)

if ~isfield(opts, 'maxit'); opts.maxit = 5000; end
if ~isfield(opts, 'alpha'); opts.alpha = 1; end  
if ~isfield(opts, 'rho'); opts.rho = 1; end     
if ~isfield(opts, 'gamma'); opts.gamma = 2; end
if ~isfield(opts, 'lambda'); opts.lambda = 0.1; end
if ~isfield(opts, 'step_type'); opts.step_type = 'fixed'; end
if ~isfield(opts, 'batchsize'); opts.batchsize = 10; end
if ~isfield(opts, 'verbose'); opts.verbose = 1; end
if ~isfield(opts, 's'); opts.verbose = 1; end
if ~isfield(opts, 'epsilon'); opts.s = 1e-08; end
%%
% 
x = x0;
out = struct();

[f_spadmm, acc_spadmm] = fun(x);
out.fvec = f_spadmm;
out.acc = acc_spadmm;
out.epoch = 0;
out.cpu = 0;

[f_spadmm_iter, acc_spadmm_iter] = fun(x);
out.fvec_iter = f_spadmm_iter;
out.acc_iter = acc_spadmm_iter;
out.iter = 0;
out.cpu_iter = 0;

rho = opts.rho;
lambda = opts.lambda;
s = opts.s;
epsilon= opts.epsilon;
gamma = opts.gamma;

y = zeros(size(L*x));
z = zeros(size(L*x));

count = 1;
iter = 1;
%% Update of S-PADMM
cpu_start = cputime;
for k = 1:opts.maxit 

    % stochastic gradient
    idx = randi(N, opts.batchsize, 1);
    g = pgfun(x, idx);    
    
    % update y
    s_k = (s/k)+epsilon;
    tau = lambda/((s_k) + rho);
    y_hat = (z + rho*L*x + (s_k)*y)/((s_k) + rho);
    y = soft_threshold(y_hat, tau);
    
    % update x
    eta = set_step(k, opts);
    h = (eta/gamma)*(g + L'*z + rho*L'*(L*x - y));
    x = x - h;
    
    % update z
    z = z + rho*(L*x - y); 

   % % --------------------- iteration --------------------- %
    [f_spadmm_iter, acc_spadmm_iter] = fun(x);
    out.fvec_iter = [out.fvec_iter; f_spadmm_iter];
    out.acc_iter = [out.acc_iter; acc_spadmm_iter];
    out.iter = [out.iter; iter];
    cpu_end_iter = cputime;
    out.cpu_iter = [out.cpu_iter; cpu_end_iter - cpu_start];

    iter = iter + 1;

    % ----------------------- epoch ----------------------- %
    if k*opts.batchsize/N >= count
        [f_spadmm, acc_spadmm] = fun(x);
        out.fvec = [out.fvec; f_spadmm];
        out.acc = [out.acc; acc_spadmm];
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
