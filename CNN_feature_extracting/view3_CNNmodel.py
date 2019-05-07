# 2019-05-07 XiaobinTian xiaobin9652@163.com
#
# construction the time-frequency domain deep feature extraction network
# Calculate the time-frequency deep feature

import tensorflow as tf


def extracting_feature(mode, data, labels, steps, i, k):
	classifier = tf.estimator.Estimator(model_fn=cnn_model, params={
											'channels': 23,
											'frequency_point': 14,
											'time_point': 256,
											'n_classes': 2,
												},
										model_dir="model/loop_" + str(k) + "/data_" + str(i) + "/view3_model")

	if mode == "train":
		train_input_fn = tf.estimator.inputs.numpy_input_fn(
			x={"x": data},
			y=labels,
			batch_size=100,
			num_epochs=None,
			shuffle=True
		)
		tensor_to_log = {"probabilities": "softmax_tensor"}
		logging_hook = tf.train.LoggingTensorHook(tensors=tensor_to_log, every_n_iter=50)
		classifier.train(input_fn=train_input_fn, steps=steps, hooks=[logging_hook])

	elif mode == "eval":
		eval_input_fn = tf.estimator.inputs.numpy_input_fn(
			x={"x": data},
			y=labels,
			num_epochs=1,
			shuffle=False
		)
		eval_result = classifier.evaluate(input_fn=eval_input_fn)
		return eval_result

	elif mode == "predict":
		predict_input_fn = tf.estimator.inputs.numpy_input_fn(
			x={"x": data},
			num_epochs=1,
			shuffle=False
		)
		predictions = list(classifier.predict(input_fn=predict_input_fn))
		#将特征图作为结果输出;
		classes = [p["feature_mapping"] for p in predictions]
		return classes


def cnn_model(features, labels, mode, params):
	#1D cnn模型
	#四个卷积层和两个全连接层;
	#第一个全连接层输出的1*1024的向量即为特征图;

	X = tf.reshape(features["x"], [-1, params['channels'], params['frequency_point'], params['time_point'], 1])
	X = tf.transpose(X, perm=[0, 3, 1, 2, 4])
	X_conv1 = tf.layers.conv3d(inputs=X, filters=1, kernel_size=[129, 1, 1], padding='valid', activation=tf.nn.tanh)
	X_conv2 = tf.layers.conv3d(inputs=X_conv1, filters=30, kernel_size=[65, 4, 4], padding='valid', activation=tf.nn.tanh)
	X_conv3 = tf.layers.conv3d(inputs=X_conv2, filters=20, kernel_size=[33, 4, 4], padding='valid', activation=tf.nn.tanh)
	X_conv4 = tf.layers.conv3d(inputs=X_conv3, filters=10, kernel_size=[17, 8, 1], padding='valid', activation=tf.nn.tanh)
	X_dense1 = tf.layers.dense(inputs=tf.reshape(X_conv4, [-1, 16 * 10 * 8 * 10]), units=2048, activation=tf.nn.tanh)
	X_dense2 = tf.layers.dense(inputs=X_dense1, units=1024, activation=tf.nn.tanh)
	X_dense3 = tf.layers.dense(inputs=X_dense2, units=100, activation=tf.nn.tanh)
	logits = tf.layers.dense(inputs=X_dense3, units=params['n_classes'], activation=None)

	predictions = {
		"classes": tf.argmax(input=logits, axis=1)[:, tf.newaxis],
		"probabilities": tf.nn.softmax(logits, name='softmax_tensor'),
		"feature_mapping": X_dense3,
	}

	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

	elif mode == tf.estimator.ModeKeys.TRAIN:
		loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits))
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
		train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

	elif mode == tf.estimator.ModeKeys.EVAL:
		loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits))

		eval_metric_ops = {
			"Accuracy": tf.metrics.accuracy(labels=tf.argmax(input=labels, axis=1), predictions=predictions["classes"]),
			"TP": tf.metrics.true_positives(labels=tf.argmax(input=labels, axis=1), predictions=predictions["classes"]),
			"FN": tf.metrics.false_negatives(labels=tf.argmax(input=labels, axis=1), predictions=predictions["classes"]),
			"TN": tf.metrics.true_negatives(labels=tf.argmax(input=labels, axis=1), predictions=predictions["classes"]),
			"FP": tf.metrics.false_positives(labels=tf.argmax(input=labels, axis=1), predictions=predictions["classes"]),
		}
		return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)