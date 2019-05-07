function [ accuracy, sensitivity, specificity ] = ...
    confusion_matrix(labels, te_Y )
% 2019-05-07 XiaobinTian xiaobin9652@163.com
% 
% Calculate the confusion matrix of the data set based on the predicted values
% Calculate performance indicators based on the confusion matrix
% 
% labels:real label
% te_Y:predicted label
% accuracy, sensitivity, specificity:three performance indicators

    confusion_matrix = zeros(2,2);
    for j = 1:size(te_Y,1)
        if(te_Y(j) == labels(j))
            if(labels(j) == 1)
                confusion_matrix(1,1) = confusion_matrix(1,1) + 1;
            else
                confusion_matrix(2,2) = confusion_matrix(2,2) + 1;
            end
        else
            if(labels(j) == 1)
                confusion_matrix(1,2) = confusion_matrix(1,2) + 1;
            else
                confusion_matrix(2,1) = confusion_matrix(2,1) + 1;
            end
        end
    end
    accuracy =  (confusion_matrix(1,1) + confusion_matrix(2,2)) / size(labels,1);
    sensitivity = confusion_matrix(1,1) / (confusion_matrix(1,1) + confusion_matrix(1,2));
    specificity = confusion_matrix(2,2) / (confusion_matrix(2,2) + confusion_matrix(2,1));
end