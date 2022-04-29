function normalized = minMaxNorm(x,min,max)

normalized = ...
    (x - min) ... 
    / (max-min);
end

