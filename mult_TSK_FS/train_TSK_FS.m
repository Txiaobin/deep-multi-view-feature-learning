function [best_pg, best_v, best_b, best_TSK_result] = train_TSK_FS( tr_data, te_data, tr_label, te_label, folds_num, view_num, k)
% 2019-05-07 XiaobinTian xiaobin9652@163.com
% 
% train classifier of each view

Ms = [ 2:1:6 ];
lamdas = [0,2.^(-5:5)];
a = 0;
best_acc_mean = 0;
for lamda = lamdas
    a = a + 1;
    c = 0;
    for M = Ms
        c = c + 1;
        result = zeros(folds_num,3);
        for fold=1:folds_num
            [v,b] = preproc(tr_data, M);
            Xg = fromXtoZ(tr_data,v,b);   %Xg:N*K
            Xg1 = Xg'*Xg;
            pg = pinv(Xg1 + lamda*eye( size(Xg1)))*Xg'*tr_label;    %Solving the consequent parameters of the TSK-FS
            [te_Y] = test_TSK_FS( te_data , pg, v, b);
            te_Y = vec2lab(te_Y);
            [ acc, sen, spe ] =  confusion_matrix(te_label, te_Y );
            result(fold,1)=acc;
            result(fold,2)=sen;
            result(fold,3)=spe;
        end
        acc_te_mean = mean(result(:,1));
        sen_te_mean = mean(result(:,2));
        spe_te_mean = mean(result(:,3));
        if acc_te_mean>best_acc_mean
			best_acc_mean = acc_te_mean；
            best_TSK_result.acc = acc_te_mean;
            best_TSK_result.sen = sen_te_mean;
            best_TSK_result.spe = spe_te_mean;
            best_pg = pg;
            best_v = v;
            best_b = b;
        end
        fprintf('train TSK FS:%d/5------view:%d\nNumber of iterations:%d/%d------%d/%d\n', k, view_num, a, size(lamdas,2), c, size(Ms,2));
        fprintf('best acc result:\nacc:%.4f  sen:%.4f  spe:%.4f\n\n',best_TSK_result.acc, best_TSK_result.sen, best_TSK_result.spe);
    end %end M
end %end lamda
