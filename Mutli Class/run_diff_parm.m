clear;clc;clear;
%% 
lambda = 0.000001;   % regularization parameter
batchsize = 128;
alpha = 1;  % step size '0.1', '0.5', '1.0'
omega = 1;

test('dataset/dna', 'dataset/dna_test', batchsize, lambda, alpha, omega);   
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
K = max(b);    
x0 = randn(m, K);    

opts = struct();
opts.batchsize = batchsize;    
opts.maxit = 10*N/opts.batchsize;  
opts.q = 100;
opts.momentum = 0.9;
opts.alpha = alpha;    
opts.rho = rho;    
opts.gamma = gamma;
opts.lambda = lambda; 
opts.omega = omega;
opts.step_type = 'fixed';    % type of step size 'fixed', 'diminishing' or 'hybrid'

fun = @(x)lr_loss(x, A, b, A_test, b_test, lambda, K);   

% optimization algorithms
if strcmp(stoalg,'SGD')
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
   [~, out] = stofun(x0, N, A, b, K, fun, opts);
else
   [~, out] = stofun(x0, N, @(x, ind)pgfun(x, ind, A, b, K), fun, opts);
end
end    


%% 
function test(dataset_train, dataset_test, batchsize, lambda, alpha, omega)
    
    fixed_rho = 1;    % '0.001', '0.01', '0.1', '1', '10'  
    gamma1 = research(alpha, fixed_rho, 0, lambda, omega, batchsize, dataset_train, dataset_test, 'SPADMM', @lr_loss, @pgfun);
    save('results\gamma\gamma1.mat', 'gamma1');
    gamma2 = research(alpha, fixed_rho, 0.5, lambda, omega, batchsize, dataset_train, dataset_test,'SPADMM', @lr_loss, @pgfun);
    save('results\gamma\gamma2.mat', 'gamma2');
    gamma3 = research(alpha, fixed_rho, 1, lambda, omega, batchsize, dataset_train, dataset_test, 'SPADMM', @lr_loss, @pgfun);
    save('results\gamma\gamma3.mat', 'gamma3');
    gamma4 = research(alpha, fixed_rho, 1.5, lambda, omega, batchsize, dataset_train, dataset_test, 'SPADMM', @lr_loss, @pgfun);
    save ('results\gamma\gamma4.mat', 'gamma4');
    gamma5 = research(alpha, fixed_rho, 2, lambda, omega, batchsize, dataset_train, dataset_test, 'SPADMM', @lr_loss, @pgfun);
    save ('results\gamma\gamma5.mat', 'gamma5');
end

%% Function values & Full gradient & Accuracy
function [f,g,acc] = lr_loss(w, X_train, Y_train, X_test, Y_test, lambda, K)

  [N, ~] = size(X_train);
  Y_hot = full(ind2vec(Y_train', K))';
  Y_pred = softmax((X_train*w)')';
  Loss = -(1/N)*sum(sum((Y_hot.*log(Y_pred))));

  f = Loss + lambda*sum(sum(abs(w)./(1+abs(w)))); 
  if nargout > 1
     diff = Y_pred - Y_hot;
     % 
     g = X_train'*diff/N + lambda*sign(w)./((1+abs(w)).^2); 
  end

  if nargout > 2
      [N_test, ~] = size(X_test);
      % 
      Y_test_pred = softmax((X_test * w)')';
      [~, test_pred_labels] = max(Y_test_pred, [], 2);  
      acc = sum(test_pred_labels == Y_test)/N_test * 100;
  end
end

