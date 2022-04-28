data = [ 2,  3,  2,  1,  0;
                1.2,1,  1,  1,  0;
                6,  2,  5,  1,  1;
                10, 4,  8,  4,  1;
                1,  1,  1,  1, NaN];
for i = 1:size(data,2)-1
    data(:,i) = minMaxNorm(data(:,i));
end
testing_set = data(1:end-1,:);
x = data(end, 1:end-1);


classID = knn(x,testing_set,2,3);