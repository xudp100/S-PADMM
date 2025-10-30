% Soft Thresholding 
function y = soft_threshold(x, tau)
    
    if tau < 0
        error('tau must be non-negative');
    end
    
    %%%
    y = sign(x).* max(abs(x) - tau, 0);

end