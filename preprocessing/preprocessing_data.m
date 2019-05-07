% 2019-05-07 XiaobinTian xiaobin9652@163.com
% 
% preprocessed dataset 
% construction of initial multi-view EEG features

clc;
clear;  
for k = 1:8
    filename = ['../data/raw_data/data' num2str(k) '.mat'];
    fprintf('load data_set:%d\n',k);
	[ X, Y ]=load_data(filename);

    fprintf('transform data_set:%d\n',k);
    [ X_1, X_2, X_3 ]=domain_transform(X);

    X_1 = single(X_1);
    X_2 = single(X_2);
    X_3 = single(X_3);

	X = {X_1; X_2; X_3};

    fprintf('save data_set:%d\n',k);
    save(strcat('../data/domain_feature/train_data',num2str(k),'.mat'), 'X', 'Y');
end