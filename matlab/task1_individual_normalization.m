%Rad 2-793 er training data
%Kolonne 7, 11, 41,  42 er features
% Here we normalize one sample at a time with the training data


s_r_mean_training = GenreClassData30s(1:792,11);
s_r_mean_test = GenreClassData30s(793:end,11);

s_c_mean_training = GenreClassData30s(1:792,7);
s_c_mean_test = GenreClassData30s(793:end,7);

mfcc1_mean_training = GenreClassData30s(1:792,41);
mfcc1_mean_test = GenreClassData30s(793:end,41);

tempo_training = GenreClassData30s(1:792,42);
tempo_test = GenreClassData30s(793:end,42);

training_labels = GenreClassData30s(1:792,66);
test_labels = GenreClassData30s(793:end,66);

training_data = [s_r_mean_training, s_c_mean_training,...
    mfcc1_mean_training,tempo_training, training_labels];

test_data = [s_r_mean_test, s_c_mean_test, ...
    mfcc1_mean_test, tempo_test];

ClassifierGuesses=zeros(size(test_labels,1),1);

for i=1:size(ClassifierGuesses,1)
    ClassifierGuesses(i)=knn_min_max(test_data(i,:),training_data,10,5);
end

ActualClasses=GenreClassData30s(793:end,66);

conf_matrix=confusionmat(ActualClasses,ClassifierGuesses);
confusionchart(conf_matrix);

misses=0;
hits=0;
for i= 1:length(ActualClasses)
    if ActualClasses(i) ~= ClassifierGuesses(i)
        misses=misses+1;
    end
    if ActualClasses(i) == ClassifierGuesses(i)
        hits=hits+1;
    end
end

errorrate=misses/length(ClassifierGuesses)
successrate=hits/length(ClassifierGuesses)

