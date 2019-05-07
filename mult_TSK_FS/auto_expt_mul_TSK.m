% 2019-05-07 XiaobinTian xiaobin9652@163.com
% 
% load and preprocess dataset
% construction the multi-view TSK fuzzy system
% Calculate the performance of the constructed system

clear;
clc;
data_num = 1;
folds_num = 5;
mean_result = zeros(folds_num, 5);
result = zeros(2,5);
for k = 1:folds_num
    load(['../data/feature/fold_' num2str(k) '/data_'  num2str(data_num)  '_train.mat']);
    load(['../data/feature/fold_' num2str(k) '/data_'  num2str(data_num)  '_predict.mat']);
    mulview_tr_cell = {tr_X_1; tr_X_2; tr_X_3};
    mulview_te_cell = {te_X_1; te_X_2; te_X_3};
    [ best_acc_result, TSK_result ] = expt_mul_TSK( mulview_tr_cell, mulview_te_cell, tr_Y, te_Y, k);
    mean_result(k,1) = mean_result(k,1) + best_acc_result.acc_mean;
    mean_result(k,2) = mean_result(k,2) + best_acc_result.sen_mean;
    mean_result(k,3) = mean_result(k,3) + best_acc_result.spe_mean;
end
result(1,:) = mean(mean_result);
save(['../data/result/data' num2str(data_num) '_result.mat'], 'result', 'mean_result');