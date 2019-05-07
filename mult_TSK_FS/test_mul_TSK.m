function [Y_te] = test_mul_TSK( test_data_cell ,TSK_cell, view_nums, c)
% 2019-05-07 XiaobinTian xiaobin9652@163.com
% 
% Calculate the predicted values of test data based on pg, v, b.
% 
% test_data_cell:test data of multi-view dataset
% c;the number of categories
% Y_te:predicted label

Y_te = zeros( size( test_data_cell{1,1},1), c);

for t = 1:view_nums
    acc_pg = TSK_cell{t,1};
    acc_v = TSK_cell{t,2};
    acc_b = TSK_cell{t,3};
    acc_w = TSK_cell{t,4};
    acc_x = test_data_cell{t,1};
    acc_x = fromXtoZ(acc_x,acc_v,acc_b);
    FS_output = acc_x*acc_pg;
    Y_te = Y_te + acc_w*FS_output;
end