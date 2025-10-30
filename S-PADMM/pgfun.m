function g = pgfun(x, ind, A, b, K)
  bind = b(ind);
  Aind = A(ind, :);
  Y_hot = full(ind2vec(bind', K))';
  Y_pred = softmax((Aind * x)')';
  diff = Y_pred - Y_hot;
  g = Aind' * diff/length(ind);
end