import tensorflow as tf
import pandas as pd
import numpy as np
import random

def my_model_fn(features, labels, mode, params, config):
	X1 = features['X1']
	X2 = features['X2']
	Y = labels
	print("X1:" + str(X1))
	print("X2:" + str(X2))
	print("Y:" + str(Y))
	W1 = tf.Variable(tf.truncated_normal((1,1)), name="W1")
	W2 = tf.Variable(tf.truncated_normal((1,1)), name="W2")
	b = tf.Variable(tf.truncated_normal((1,)), name="b")
	_Y = tf.matmul(X1, W1) + tf.matmul(X2, W2) + b
	if not Y is None:
		loss = tf.reduce_mean(tf.square(Y - _Y))

		optimizer = tf.train.AdamOptimizer(0.005)
		train_op = optimizer.minimize(loss, global_step = tf.train.get_global_step())
	else:
		loss = None
		train_op = None
	return tf.estimator.EstimatorSpec(
		mode = mode,
		predictions = _Y,
		loss = loss,
		train_op = train_op,
		)

def my_input_fn_ds():
	count = 10
	X1 = [[random.random() * 100] for i in range(count)]
	X2 = [[random.random() * 100] for i in range(count)]
	Y = [X1[i][0] * 2 + X2[i][0] * 3 + 4 for i in range(count)]

	print("make train inputs dataset")
	print('X1: ', X1)
	print('X2: ', X2)
	print('Y: ', Y)
	# 可以通过numpy的对象直接创建数据集，这应该是正确的打开方式了
	dataset = tf.data.Dataset.from_tensor_slices((
		{'X1':X1, 'X2':X2},
		Y
		))
	dataset = dataset.repeat().batch(50)
	return dataset

def my_pred_input_fn():
	features = [[float(i)] for i in range(20)]
	print("make pred inputs")
	print(features)
	return (features, )

def main():
	tf.logging.set_verbosity(tf.logging.INFO) # 打开这句话，可以让训练过程中输出loss和步骤等信息
	est = tf.estimator.Estimator(my_model_fn, model_dir='save_est/')
	# train
	est.train(my_input_fn_ds, steps=1000)

	# evaluate
	# 输出： {'loss': 33548.453, 'global_step': 2000}
	# 看起来，应该可以在model_fn中，通过eval_metric_ops这个参数来定义需要验证的数值
	#print("evaluate")
	#print(est.evaluate(my_input_fn, steps=1))

	# pred
	# preds = est.predict(my_pred_input_fn)
	# for i in range(20):
	# 	print(next(preds))

	# dump
	names = est.get_variable_names()
	for name in names:
		print("\t%s: %f" % (name, est.get_variable_value(name)))

if __name__ == '__main__':
	main()
