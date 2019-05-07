function [ best_acc_result, TSK_result ] = expt_mul_TSK( mulview_tr_cell, mulview_te_cell, tr_label, te_label, fold)
% 2019-05-07 XiaobinTian xiaobin9652@163.com
% 
% construction the multi-view TSK fuzzy system
% Calculate the performance of the constructed system

view_nums = size(mulview_tr_cell,1);
TSK_cell = cell(view_nums,4);
folds_num = 5;
options.view_nums = view_nums;
maxIter = 10;
options.maxIter = maxIter;
lamda1s = 2.^(-6:6);
lamda2s = 2.^(-6:6);
lamda3s = 2.^(-6:6);
n = size(tr_label,2);
te_label = vec2lab(te_label);

% zero-centered train data of each view
for view_num = 1:view_nums
    acc_data = mulview_tr_cell{view_num,1};
    acc_data = mapminmax(acc_data', 0, 1)';
    mulview_tr_cell{view_num,1} = acc_data;
end
% zero-centered test data of each view
for view_num = 1:view_nums
    acc_data = mulview_te_cell{view_num,1};
    acc_data = mapminmax(acc_data', 0, 1)';
    mulview_te_cell{view_num,1} = acc_data;
end


% train classifier of each view
TSK_result = cell(view_nums,1);
for view_num = 1:view_nums
    tr_data = mulview_tr_cell{view_num,1};
	te_data = mulview_te_cell{view_num,1};
    [pg, v, b, best_TSK_result] = train_TSK_FS( tr_data, te_data, tr_label, te_label, folds_num, view_num, fold);
    TSK_cell{ view_num, 1 } = pg;
    TSK_cell{ view_num, 2 } = v;
    TSK_cell{ view_num, 3 } = b;
    TSK_cell{ view_num, 4 } = 1/view_nums;
    TSK_result{ view_num, 1 } = best_TSK_result;
end

best_acc_te = 0;
a = 0;  
for lamda1 = lamda1s
    a = a + 1;
    b = 0;
    for lamda2 = lamda2s
        b = b + 1;
        c = 0;
        for lamda3 = lamda3s
            c = c + 1;
            options.lamda1 = lamda1;
            options.lamda2 = lamda2;
            options.lamda3 = lamda3;
            result = zeros(folds_num,1);
            try
                for fold_num = 1:folds_num              
                    [best_TSK_cell, best_lamada_scale] = train_mul_TSK( mulview_tr_cell, TSK_cell, tr_label, options);
                    [te_Y] = test_mul_TSK( mulview_te_cell ,best_TSK_cell, view_nums, n);
                    te_Y = vec2lab(te_Y);
                    [ acc, sen, spe ] = confusion_matrix(te_label, te_Y);
                    result(fold_num,1) = acc;
                    result(fold_num,2) = sen;
                    result(fold_num,3) = spe;
                end
            catch err
                disp(err);
                    warning('Something wrong when using function pinv!');
                break;
            end
            acc_te_mean = mean(result(:,1));
            sen_te_mean = mean(result(:,2));
            spe_te_mean = mean(result(:,3));
            if (acc_te_mean>best_acc_te)
                best_acc_te = acc_te_mean;
                best_acc_result.best_model = best_TSK_cell;
                best_acc_result.acc_mean = acc_te_mean;
                best_acc_result.sen_mean = sen_te_mean;
                best_acc_result.spe_mean = spe_te_mean;          
                best_acc_result.lamda_scale = best_lamada_scale;
            end
            fprintf('train mul TSK FS:%d/5\nNumber of iterations:%d-----%d-----%d\n', fold, a, b, c);
            fprintf('best acc result:\nacc:%.4f  sen:%.4f  spe:%.4f\n',best_acc_result.acc_mean, best_acc_result.sen_mean, best_acc_result.spe_mean);
        end %end lamda3s
    end %end lamda2s
end %end lamda1s
