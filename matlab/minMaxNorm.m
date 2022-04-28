function normalized = minMaxNorm(x)

normalized = ...
    (x - min(x)) ... 
    / (max(x)-min(x));
end

