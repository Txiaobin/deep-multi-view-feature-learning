function [v,U]=preproc(data,M)
% 2019-05-07 XiaobinTian xiaobin9652@163.com
% 
% implementation of a Hierarchical Clustering Method
% a detailed description is described in the following paper
% :Transfer Representation Learning with TSK Fuzzy System, pengxu
% :Deep multi-view feature learning for epileptic seizure detection, xiaobin tian
% 
% data:input data 
% M:Number of categories
% v,U:the antecedent parameter of TSK fuzzy system

[N, d]=size(data);
h = 1;

if M > N
    M = N;
end

if M > 1
    corrDist = pdist(data, 'euclidean');
    clusterTree = linkage(corrDist, 'average');
    clusters = cluster(clusterTree, 'maxclust', M);
    v_std = std(data);
else
    v_std = std(data);
    clusters = ones(N, 1);
end

v = zeros(M, d);
delta = zeros(M, d);

for i = 1:M
    hi = find(clusters == i);
    tem = data(hi, :);
    if length(hi) > 1
        v(i, :) = mean(tem);
        delta(i, :) = std(tem) + 1e-6;
    else
        v(i, :) = tem;
        delta(i, :) = v_std + 1e-6;
    end
end
delta = delta/h;
U = delta;