function [ X_1, X_2, X_3]=domain_transform(X)
% 2019-05-07 XiaobinTian xiaobin9652@163.com
% 
% using FFT and WPT to obtain the frequency features 
% and time-frequency features of the dataset
% 
% the adopted wavelet basis function is Daubechies (dbN)
% the order of the wavelet function is set to 4
% the level of decomposition layers of the wavelet transform is 6
% They can all change according to your own situation
% 
% X:features of the dataset;
% X_1:time features;
% X_2:frequency features;
% X_3:time-frequency features;

    L = size(X, 1);
    chan = size(X, 2);
    num = size(X, 3);

    X_1 = [];
    X_2 = [];
    X_3 = [];
% processing dataset into time features
    fprintf('    transform time feature\n');
    for i = 1:num
        a = [];
        for j = 1:chan
            a = cat(1, a, X(:,j,i));
        end
        X_1 = cat(1, X_1, a'); 
    end
% processing dataset into frequency features, using FFT
    fprintf('    transform frequency feature\n');
    NFFT = 2 ^ nextpow2(L);
    for i = 1:num
        b = [];
        for j = 1:chan
            a = fft(X(:,j,i),NFFT) / L;
            a = 2 * abs(a);
            b = cat(1, b, a(4:30));
        end
        X_2 = cat(1, X_2, b');
    end
% processing dataset into time-frequency features, using WPD
    fprintf('    transform time-frequency feature\n');
    WPD_layers = 6;
    wavelet_basis = 'db4';
    fs = 256;
    for i = 1:num
        a = [];
        for j = 1:chan
            wpt = wpdec(X(:,j,i), WPD_layers, wavelet_basis);
            [SPEC,~,~] = wpspectrum(wpt,fs);
            a = cat(2, a, reshape(SPEC(2:15,:)', [14 * 256,1])');
        end
        X_3 = cat(1, X_3, a);                
    end
end