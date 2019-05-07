function zt=fromXtoZ(data,v,b)
% 2019-05-07 XiaobinTian xiaobin9652@163.com
% 
% Convert X to Xg according to the antecedent parameters
% 
% data:X
% zt:Xg

N = size(data, 1);
xt = [data, ones(N,1)];
[M, d] = size(v);

for i = 1:M
    v1 = repmat(v(i,:), N,1);
    bb = repmat(b(i,:), N,1);
    wt(:,i) = exp(-sum((data - v1) .^ 2 ./ bb, 2));
end

wt2 = sum(wt, 2);
wt = wt ./ repmat(wt2, 1, M);

zt = [];
for i=1:M
    wt1 = wt(:,i);
    wt2 = repmat(wt1, 1, d+1);
    zt = [zt, xt .* wt2];
end
Mask = isnan(zt);
zt(Mask) = 1e-5;
%zt:N*K