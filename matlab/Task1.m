%Rad 2-793 er training data
%Kolonne 7, 11, 41,  42 er features
% Here we normalize all of the testing set along with the training set

s_r_mean = GenreClassData30s(1:end,11);
s_r_mean = minMaxNorm(s_r_mean);
s_r_mean_training=s_r_mean(1:792,1);
s_r_mean_test = s_r_mean(793:end,1);

s_c_mean = GenreClassData30s(1:end,7);
s_c_mean = minMaxNorm(s_c_mean);
s_c_mean_trainig = s_c_mean(1:792,1);
s_c_mean_test = s_c_mean(793:end,1);

mfcc1_mean = GenreClassData30s(1:end,41);
mfcc1_mean = minMaxNorm(mfcc1_mean);
mfcc1_mean_training = mfcc1_mean(1:792,1);
mfcc1_mean_test = mfcc1_mean(793:end,1);

tempo = GenreClassData30s(1:end,42);
tempo = minMaxNorm(tempo);
tempo_training=tempo(1:792,1);
tempo_test=tempo(793:end,1);

training_labels = GenreClassData30s(1:792,66);
test_labels = GenreClassData30s(793:end,66);

training_data = [s_r_mean_training, s_c_mean_trainig,...
    mfcc1_mean_training,tempo_training, training_labels];
test_data = [s_r_mean_test, s_c_mean_test, ...
    mfcc1_mean_test, tempo_test];


ClassifierGuesses=zeros(size(test_labels,1),1);

for i=1:size(ClassifierGuesses,1)
    ClassifierGuesses(i)=knn(test_data(i,:),training_data,10,5);
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






