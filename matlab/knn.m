function decision = knn(x, trainingSet, m, k)
    n=size(x,2);
    K = zeros(m,1);
    distance = inf*ones(k,1);
    X_nearest = zeros(k,n+1);
    for i = 1:size(trainingSet,1)
        d = norm(x-trainingSet(i,1:n),2);
        [furthest_distance, j] = max(distance);
        if d < furthest_distance
            X_nearest(j,:) = trainingSet(i,:);
            distance(j) = d;
        end
    end
    for i = 1:size(X_nearest,1)
        id = X_nearest(i,n+1) + 1; % IDs are 0 indexed
        K(id) = K(id) + 1;
    end

    [max_1, max_index_1] = max(K);
    decision = max_index_1 - 1;
    % TIE BREAKER ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    tiebreaker = false;
    equal_count = zeros(m,1);
    K(max_index_1) = 0;
    equal_count(max_index_1) = 1;

    [max_i, max_index_i] = max(K);
    while max_1 == max_i
        tiebreaker = true;
        equal_count(max_index_i) = 1;
        K(max_index_i) = 0;
        [max_i, max_index_i] = max(K);        
    end
    least_distance = inf;
    for i = 1:length(equal_count)
        if equal_count(i) == 1 && tiebreaker
            label = i-1;
            distance = 0;
            for j = 1:size(X_nearest,1)
                if X_nearest(j,end) == label
                    distance = distance + norm(x-X_nearest(j,1:n));
                end
            end
            if distance < least_distance
                least_distance = distance;
                decision = label;
            end
        end
    end
end
