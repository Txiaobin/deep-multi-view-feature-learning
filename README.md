# deep multi-view feature learning
#
## Abstract
&emsp; Epilepsy is a common neurological illness caused by abnormal discharge of brain neurons, where epileptic seizure could lead to life-threatening emergencies. By analyzing the encephalogram (EEG) signals of patients with epilepsy, their conditions can be monitored so that epileptic seizure can be detected and intervened in time. In epilepsy research, use of appropriate methods to obtain effective features is of great importance to the detection accuracy. In order to obtain features that can produce better detection results, this paper proposes a multi-view deep feature extraction method. The method first uses fast Fourier transform (FFT) and wavelet packet decomposition (WPD) to construct the initial multi-view features. Convolutional neural network (CNN) is then used to automatically learn the deep feature from the initial multi-view features, which can reduce the dimensionality and obtain features with better ability for seizure identification. Furthermore, the multi-view Takagi-Sugeno-Kang fuzzy system (MV-TSK-FS), an interpretable rule-based classifier, is used to obtain a classification model with stronger generalizability based on the obtained deep multi-view features. Experimental studies show that the proposed multi-view deep feature extraction method has better performance than common feature extraction methods such as principal component analysis (PCA), FFT and WPD. The performance of classification using the multi-view deep features is better than that using single-view deep features.
## author
Xiaobin Tian, Zhaohong Deng, Senior Member, IEEE, Kup-Sze Choi, Dongrui Wu, Senior Member,
IEEE, Bin Qin, Jun Wan, Hongbin Shen, Shitong Wang
## using this code
```
├── data
|  ├──raw\_data
├──preprocessing
|  ├──preprocessing_data.m
|  ├──load_data.m
|  ├──domain_transform.m
├──CNN\_feature\_extracting
|  |  ├──feature_extracting.py
|  |  ├──view1_CNNmodel.py
|  |  ├──view2_CNNmodel.py
|  |  ├──view3_CN
├──mult\_TSK\_FS
|  |  ├──auto_expt_mul_TSK.m
|  |  ├──confusion_matrix.m
|  |  ├──expt_mul_TSK.m
|  |  ├──fromXtoZ.m
|  |  ├──lab2vec.m
|  |  ├──preproc.m
|  |  ├──test_mul_TSK.m
|  |  ├──test_TSK_FS.m
|  |  ├──train_mul_TSK.m
|  |  ├──train_TSK_FS.m
|  |  ├──vec2lab.m
```
1.Please set the original datset in "data/raw_data".

2.Run "/preprocessing/preprocessing_data.m" using matlab and get the initial multi-view EEG features.

3.python3 is needed. Your environment should include numpy, scipy and tensorflow. Run "/CNN\_feature\_extracting/feature_extracting.py" using python3 and get the deep multi-view features.

4.Run "/mult\_TSK\_FS/auto_expt_mul_TSK.m" using matlab and calculate the performance of this study.