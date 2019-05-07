function [ X, Y ]=load_data(filename)
% 2019-05-07 XiaobinTian xiaobin9652@163.com
% 
% load and preprocessed dataset
% part of the non-seizure data is abandoned 
% and over-sampling method is applied for the seizure data,
% where using sliding window to capture clips and allowing overlapping between two windows. 
% this approach reduces the difference between the number of non-seizure and seizure data.
% 
% filename:the filename of dataset 
% X:features of the dataset
% Y:labels of the dataset(one-of-hot)

    overlap_ratio = 3/4;
    segment_size = 256;
    overlap_size = segment_size*(1-overlap_ratio);
	X = [];
	Y = [];
    load(filename);
    %split segment using sliding windows
    for i = 0 : (size(seizure_data,1)-segment_size) / overlap_size
        X = cat(3, X, seizure_data(1 + overlap_size * i:segment_size + overlap_size * i,:));
		Y = cat(1, Y, [0, 1]);
    end
    for i = 0 : (size(nonseizure_data,1)-segment_size) / segment_size
        X = cat(3, X, nonseizure_data(1 + segment_size * i:segment_size + segment_size * i,:));
		Y = cat(1, Y, [1, 0]);
    end
    %disrupt dataset
	rank = randperm(size(X, 3));
	X = X(:,:,rank);
	Y = Y(rank,:);