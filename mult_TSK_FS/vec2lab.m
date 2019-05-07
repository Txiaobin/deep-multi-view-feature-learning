function [ labels ] = vec2lab( vectors )
% 2019-05-07 XiaobinTian xiaobin9652@163.com
% 
% convert the vector(one of hot) of the label to a scalar



[~,labels]=max(vectors,[],2);

end

