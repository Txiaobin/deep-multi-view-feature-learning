# 2019-05-07 XiaobinTian xiaobin9652@163.com
#
# construction three deep feature extraction network
# Construction of deep multi-view features

import view1_CNNmodel
import view2_CNNmodel
import view3_CNNmodel
import scipy.io as sio
import numpy as np
import h5py


def data_preprocess(file_name):
	X_1 = h5py.File(file_name)['X_1']
	X_1 = np.transpose(X_1)

	X_2 = h5py.File(file_name)['X_2']
	X_2 = np.transpose(X_2)

	X_3 = h5py.File(file_name)['X_3']
	X_3 = np.transpose(X_3)
	
	Y = h5py.File(file_name)['Y']
	Y = np.transpose(Y)
	
	return X_1, X_2, X_3, Y


def split(X_1, X_2, X_3, Y, k):
	fold_num = 5
	tr_X_1 = np.empty(shape=[0, X_1.shape[1]], dtype='float32')
	tr_X_2 = np.empty(shape=[0, X_2.shape[1]], dtype='float32')
	tr_X_3 = np.empty(shape=[0, X_3.shape[1]], dtype='float32')
	tr_Y = np.empty(shape=[0, Y.shape[1]])
	
	for fold in range(fold_num):
		if fold+1 == k:
			te_X_1 = X_1[fold:: fold_num, :]
			te_X_2 = X_2[fold:: fold_num, :]
			te_X_3 = X_3[fold:: fold_num, :]
			te_Y = Y[fold:: fold_num, :]
		else:
			tr_X_1 = np.append(tr_X_1, X_1[fold:-1:fold_num, :], axis=0)
			tr_X_2 = np.append(tr_X_2, X_2[fold:-1:fold_num, :], axis=0)
			tr_X_3 = np.append(tr_X_3, X_3[fold:-1:fold_num, :], axis=0)
			tr_Y = np.append(tr_Y, Y[fold:-1:fold_num, :], axis=0)

	return te_X_1, te_X_2, te_X_3, te_Y, tr_X_1, tr_X_2, tr_X_3, tr_Y


steps_1 = 3000
steps_2 = 2000
steps_3 = 5000
for i in range(7, 25):
	#modeldir_name = 'model'
	#if os.path.isdir(modeldir_name):
		#shutil.rmtree(modeldir_name)
	for k in range(1, 6):
		print("\ndata_set:{}".format(i))
		print("loop:{}".format(k))

		file_name = '../data/domain_feature/train_data' + str(i) + '.mat'
		X_1, X_2, X_3, Y = data_preprocess(file_name)
		
		te_X_1, te_X_2, te_X_3, te_Y, tr_X_1, tr_X_2, tr_X_3, tr_Y = split(X_1, X_2, X_3, Y, k)
		
		print("\ntrain view_1 CNNmodel")
		view1_CNNmodel.extracting_feature('train', tr_X_1, tr_Y, steps_1, i, k)
		print("\ntrain view_2 CNNmodel")
		view2_CNNmodel.extracting_feature('train', tr_X_2, tr_Y, steps_2, i, k)
		print("\ntrain view_3 CNNmodel")
		view3_CNNmodel.extracting_feature('train', tr_X_3, tr_Y, steps_3, i, k)

		print("\nextracting multi-view feature")
		tr_X_1 = view1_CNNmodel.extracting_feature('predict', tr_X_1, tr_Y, steps_1, i, k)
		tr_X_2 = view2_CNNmodel.extracting_feature('predict', tr_X_2, tr_Y, steps_2, i, k)
		tr_X_3 = view3_CNNmodel.extracting_feature('predict', tr_X_3, tr_Y, steps_3, i, k)
		
		tr_X_1 = np.array(tr_X_1)
		tr_X_2 = np.array(tr_X_2)
		tr_X_3 = np.array(tr_X_3)
		sio.savemat('../data/feature/fold_' + str(k) + '/data_' + str(i) + '_train', {'tr_X_1':tr_X_1,
															'tr_X_2': tr_X_2,
															'tr_X_3': tr_X_3,
															'tr_Y': tr_Y
															})

		te_X_1 = view1_CNNmodel.extracting_feature('predict', te_X_1, te_Y, steps_1, i, k)
		te_X_2 = view2_CNNmodel.extracting_feature('predict', te_X_2, te_Y, steps_2, i, k)
		te_X_3 = view3_CNNmodel.extracting_feature('predict', te_X_3, te_Y, steps_3, i, k)
		
		te_X_1 = np.array(te_X_1)
		te_X_2 = np.array(te_X_2)
		te_X_3 = np.array(te_X_3)
		sio.savemat('../data/feature/fold_' + str(k) + '/data_' + str(i) + '_predict', {'te_X_1': te_X_1,
																'te_X_2': te_X_2,
																'te_X_3': te_X_3,
																'te_Y': te_Y
																}) 
