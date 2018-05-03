import tensorflow as tf
import pandas as pd
import numpy as np
import random

def my_model_fn(features, labels, mode, params, config):
	X = features['X']
	Y = labels
	print("X:" + str(X))
	print("Y:" + str(Y))
	print("params:" + str(params))
	W = tf.Variable(tf.truncated_normal((2,1)), name="W")
	b = tf.Variable(0., name="b")
	_Y = tf.reshape(tf.matmul(X, W) + b, (-1,))
	if not Y is None:
		loss = tf.reduce_mean(tf.square(Y - _Y))

		# optimizer = tf.train.AdamOptimizer(params['lr'])
		optimizer = tf.train.GradientDescentOptimizer(params['lr'])
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
	count = 1000
	X = [[random.random(), random.random()] for i in range(count)]
	Y = [X[i][0] * 2 + X[i][1] * 3 + 4 for i in range(count)]

	print("make train inputs dataset")
	# print('X: ', X)
	# print('Y: ', Y)
	# 可以通过numpy的对象直接创建数据集，这应该是正确的打开方式了
	dataset = tf.data.Dataset.from_tensor_slices((
		{'X':X},
		Y
		))
	dataset = dataset.repeat().batch(50)
	return dataset

def my_pred_input_fn(X):
	def fn():
		dataset = tf.data.Dataset.from_tensor_slices((
			{'X':X},
			))
		dataset = dataset.batch(1)
		return dataset
	return fn

def main():
	tf.logging.set_verbosity(tf.logging.INFO) # 打开这句话，可以让训练过程中输出loss和步骤等信息
	est = tf.estimator.Estimator(my_model_fn, model_dir='save_est/', params={'lr':0.01})
	# train
	est.train(my_input_fn_ds, steps=1000)

	# evaluate
	# 输出： {'loss': 33548.453, 'global_step': 2000}
	# 看起来，应该可以在model_fn中，通过eval_metric_ops这个参数来定义需要验证的数值
	#print("evaluate")
	#print(est.evaluate(my_input_fn, steps=1))

	# pred
	X = [[float(i / 10.), float(i / 10.)] for i in range(5)]
	print(X)
	preds = est.predict(my_pred_input_fn(X))
	print(list(preds))

	# dump
	names = est.get_variable_names()
	for name in names:
		if not 'Adam' in name:
			print("\t%s: %s" % (name, str(est.get_variable_value(name))))

if __name__ == '__main__':
	main()
