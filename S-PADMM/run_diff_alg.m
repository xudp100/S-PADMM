clear;clc;clear;
  
lambda = 0.00001;
batchsize = 4096; 
num_epoch = 1; 
seed = 40;
gamma = 1; 
rho = 1;

dataset_train= 'dataset/mnist'; 
dataset_test= 'dataset/mnist_test';
   

[~, dataset_name, ~] = fileparts(dataset_train);
if strcmp(dataset_name, 'dataset')
    [~, dataset_name, ~] = fileparts(fileparts(dataset_train));
end

L_folder = 'data_L';
L_filename = fullfile(L_folder, [dataset_name, '_L.mat']);

[b, A] = libsvmread(dataset_train);    
[N, m] = size(A);
[b_test, A_test] = libsvmread(dataset_test);    
C = max(b);

if exist(L_filename, 'file')
    loaded_data = load(L_filename);
    
    if isfield(loaded_data, 'L')
        L = loaded_data.L;
    end
end

data.gamma = gamma;
data.num_iter = num_epoch;
data.seed = seed;
data.A = A;
data.b = b;
data.A_test = A_test;
data.b_test = b_test;
data.C = C;
data.L = L;
data.N = N;
data.m = m; 
test(data, batchsize, rho, lambda);

function [out] = research(data, alpha, rho, lambda, batchsize, stoalg, lr_loss, pgfun)

num_iter = data.num_iter;
A = data.A;
b = data.b;
A_test = data.A_test;
b_test = data.b_test;
L = data.L;
C = data.C;
N = data.N;
m = data.m;

seed = data.seed;
ss = RandStream('mt19937ar', 'Seed', seed);
RandStream.setGlobalStream(ss);

x0 = randn(m, C);    

opts = struct();
opts.batchsize = batchsize;    
opts.maxit = num_iter*N/opts.batchsize; 
opts.rho = rho;     
opts.alpha = alpha;
opts.gamma = data.gamma;
opts.s = 2.0;
opts.epsilon = 1e-08;
opts.lambda = lambda;
opts.step_type = 'fixed';

fun = @(x)lr_loss(x, A, b, A_test, b_test, lambda, C, L);  

if strcmp(stoalg, 'SPADMM')
    stofun = @SPADMM;
    [~, out] = stofun(x0, L, N, @(x, ind)pgfun(x, ind, A, b, C), fun, opts);
end
end   

%% 
function test(data, batchsize, rho, lambda)
    out = research(data, 1, rho, lambda, batchsize, 'SPADMM', @lr_loss, @pgfun);
end

function [f, acc] = lr_loss(w, X_train, Y_train, X_test, Y_test, lambda, C, B)

  [N, ~] = size(X_train);
  Y_hot = full(ind2vec(Y_train', C))';
  Y_pred = softmax((X_train*w)')';
  Loss = -(1/N)*sum(sum((Y_hot.*log(Y_pred))));
  
  f = Loss + lambda*sum(sum(abs(B*w), 1));

  if nargout > 1
      [N_test, ~] = size(X_test);
      Y_test_pred = softmax((X_test*w)')';
      [~, test_pred_labels] = max(Y_test_pred, [], 2);  
      acc = sum(test_pred_labels == Y_test)/N_test*100;
  end
end