clear;clc;clear;
%% 
lambda = 0.00001;    % regularization parameter
batchsize = 256; 
rho = 1;    
gamma = 2;    
omega = 1; 

test('dataset/covtype', 'dataset/covtype_test', batchsize, rho, gamma, lambda, omega);   
%%
function [out] = research(alpha, rho, gamma, lambda, omega, batchsize, dataset_train, dataset_test, stoalg, lr_loss, pgfun)

% random seed
seed = 1;
ss = RandStream('mt19937ar', 'Seed', seed);
RandStream.setGlobalStream(ss);

% load datasets
[b,A] = libsvmread(dataset_train);    % training set
[N,m] = size(A);
[b_test, A_test] = libsvmread(dataset_test);    % test set

% display dataset information
disp(['Training set: ', dataset_train, ', Test set: ', dataset_test,]);
disp(['Training samples: ', num2str(size(A, 1)),', Test samples: ', num2str(size(A_test, 1)), ', Features: ', num2str(size(A, 2))]);
disp(['Labels: ', num2str(unique(b)')]);

% initial point
x0 = randn(m, 1);    

opts = struct();
opts.batchsize = batchsize;    
opts.maxit = 100*N/opts.batchsize;   
opts.q = 100;
opts.momentum = 0.9;
opts.alpha = alpha;    
opts.rho = rho;    
opts.gamma = gamma;
opts.lambda = lambda; 
opts.omega = omega;   
opts.step_type = 'fixed';    % type of step size 'fixed', 'diminishing' or 'hybrid'

fun = @(x)lr_loss(x, A, b, A_test, b_test, lambda);  

% optimization algorithms
if strcmp(stoalg, 'SGD')
    stofun = @SGD;
elseif strcmp(stoalg, 'SGDM')
    stofun = @SGDM;
elseif strcmp(stoalg, 'AdaGrad')
    stofun = @AdaGrad;
elseif strcmp(stoalg, 'ADMM')
    stofun = @ADMM;
elseif strcmp(stoalg, 'P_ADMM')
    stofun = @P_ADMM;
elseif strcmp(stoalg, 'STOC_ADMM')
    stofun = @STOC_ADMM;
elseif strcmp(stoalg, 'SPADMM')
    stofun = @SPADMM;
elseif strcmp(stoalg, 'SPIDER_ADMM')
    stofun = @SPIDER_ADMM;      
end
%%%
if strcmp(stoalg, 'ADMM') || strcmp(stoalg, 'P_ADMM')
   [~, out] = stofun(x0, N, A, b, fun, opts);
else
   [~, out] = stofun(x0, N, @(x, ind)pgfun(x, ind, A, b), fun, opts);
end
end   

%% 
function test(dataset_train, dataset_test, batchsize, rho, gamma, lambda, omega)
    
    out1 = research(0.01, rho, gamma, lambda, omega, batchsize, dataset_train, dataset_test, 'SGD', @lr_loss, @pgfun);
    save('results\out1.mat', 'out1');
    out2 = research(0.01, rho, gamma, lambda, omega, batchsize, dataset_train, dataset_test,'SGDM', @lr_loss, @pgfun);
    save('results\out2.mat', 'out2');
    out3 = research(0.01, rho, gamma, lambda, omega, batchsize, dataset_train, dataset_test, 'AdaGrad', @lr_loss, @pgfun);
    save('results\out3.mat', 'out3');
    out4 = research(1, rho, gamma, lambda, omega, batchsize, dataset_train, dataset_test, 'ADMM', @lr_loss, @pgfun);
    save ('results\out4.mat', 'out4');
    out5 = research(1, rho, gamma, lambda, omega, batchsize, dataset_train, dataset_test, 'P_ADMM', @lr_loss, @pgfun);
    save ('results\out5.mat', 'out5');
    out6 = research(1, rho, gamma, lambda, omega, batchsize, dataset_train, dataset_test, 'STOC_ADMM', @lr_loss, @pgfun);
    save ('results\out6.mat', 'out6');
    out7 = research(1, rho, gamma, lambda, omega, batchsize, dataset_train, dataset_test, 'SPADMM', @lr_loss, @pgfun);
    save ('results\out7.mat', 'out7');
    out8 = research(1, rho, gamma, lambda, omega, batchsize, dataset_train, dataset_test, 'SPIDER_ADMM', @lr_loss, @pgfun);
    save ('results\out8.mat', 'out8');
 
end


%% Function values & Full gradient & Accuracy
function [f,g,acc] = lr_loss(x, A, b, A_test, b_test, lambda)
  [N,~] = size(A);
  exp_ba = 1 + exp(b.*(A*x));
  loss = sum(1./exp_ba)/N; 
  % 
  f = loss + lambda*sum(abs(x)./(1 + abs(x)));

if nargout > 1    
  b_expba = b.*exp(b.*(A*x));
  exp_square = (1 + exp(b.*(A*x))).^2;
  %
  g = A'*(-b_expba./exp_square)/N + lambda*sign(x)./((1 + abs(x)).^2);
end

if nargout > 2    
   [N_test, ~] = size(A_test);
   pred = A_test*x;  
   correct = b_test.*pred >= 0;  
   % 
   acc = sum(correct)/N_test*100;  
end
end




