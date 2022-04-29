%Row 2-793 is training data
%Column 7, 11, 41,  42 are wanted features
%Using the min and max from training to normalize both training and test
s_r_mean = GenreClassData30s(1:end,11);
s_r_mean_training=s_r_mean(1:792,1);
s_r_mean_test = s_r_mean(793:end,1);
srmt_max=max(s_r_mean_training);
srmt_min=min(s_r_mean_training);
s_r_mean_training=minMaxNorm(s_r_mean_training,srmt_min,srmt_max);
s_r_mean_test = minMaxNorm(s_r_mean_test,srmt_min,srmt_max);


s_c_mean = GenreClassData30s(1:end,7);
s_c_mean_training = s_c_mean(1:792,1);
s_c_mean_test = s_c_mean(793:end,1);
scmt_min=min(s_c_mean_training);
scmt_max=max(s_c_mean_training);
s_c_mean_training = minMaxNorm(s_c_mean_training,scmt_min,scmt_max);
s_c_mean_test = minMaxNorm(s_c_mean_test,scmt_min,scmt_max);


mfcc1_mean = GenreClassData30s(1:end,41);
mfcc1_mean_training = mfcc1_mean(1:792,1);
mfcc1_mean_test = mfcc1_mean(793:end,1);
mmt_min=min(mfcc1_mean_training);
mmt_max=max(mfcc1_mean_training);
mfcc1_mean_training=minMaxNorm(mfcc1_mean_training,mmt_min,mmt_max);
mfcc1_mean_test=minMaxNorm(mfcc1_mean_test,mmt_min,mmt_max);

tempo = GenreClassData30s(1:end,42);
tempo_training=tempo(1:792,1);
tempo_test=tempo(793:end,1);

t_min=min(tempo_training);
t_max=max(tempo_test);

tempo_training=minMaxNorm(tempo_training,t_min,t_max);
tempo_test=minMaxNorm(tempo_test,t_min,t_max);

training_labels = GenreClassData30s(1:792,66);
test_labels = GenreClassData30s(793:end,66);

training_data = [s_r_mean_training, s_c_mean_training, mfcc1_mean_training, tempo_training, training_labels];
test_data     = [s_r_mean_test, s_c_mean_test, mfcc1_mean_test, tempo_test];


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






