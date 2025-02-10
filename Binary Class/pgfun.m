function g = pgfun(x, ind, A, b)
  bind = b(ind);
  Aind = A(ind, :);
  b_expba = bind.*exp(bind.*Aind*x);
  exp_sqare = (1 + exp(bind.*Aind*x)).^2;
  g = Aind'*(-b_expba./exp_sqare)/length(ind);
end