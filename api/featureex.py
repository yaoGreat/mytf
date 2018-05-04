import tensorflow as tf
import pandas as pd
import numpy as np
import random

FEATURE_COUNT = 638

# 模型函数的写法
def my_model_fn(features, feature_columns, labels, mode, params, config):
	Y = labels
	_input = tf.feature_column.input_layer(features, feature_columns)
	print("input", _input)
	hidden = tf.layers.dense(_input, 8, activation=tf.nn.tanh, name='hidden')
	output = tf.layers.dense(hidden, 10, name='output')
	print("output", output)
	predictions = tf.argmax(output, axis=1)
	if not Y is None:
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=tf.one_hot(Y, 10)))

		# optimizer = tf.train.AdamOptimizer(params['lr'])
		optimizer = tf.train.GradientDescentOptimizer(params['lr'])
		train_op = optimizer.minimize(loss, global_step = tf.train.get_global_step())
	else:
		loss = None
		train_op = None
	return tf.estimator.EstimatorSpec(
		mode = mode,
		predictions = predictions,
		loss = loss,
		train_op = train_op,
		)

# 输入函数的写法
def my_input_fn(X, Y, batch, repeat):
	dataset = tf.data.Dataset.from_tensor_slices((
		{'X':X},
		Y
		))
	if repeat:
		dataset = dataset.repeat()
	dataset = dataset.batch(batch)
	return dataset

def main():
	global FEATURE_COUNT

	tf.logging.set_verbosity(tf.logging.INFO) # 打开这句话，可以让训练过程中输出loss和步骤等信息
	
	'''
	numeric_column 的 shape:
	shape: An iterable of integers specifies the shape of the `Tensor`. An
	        integer can be given which means a single dimension `Tensor` with given
	        width. The `Tensor` representing the column will have the shape of
	        [batch_size] + `shape`.
	'''
	feature_columns = [
		tf.feature_column.numeric_column(key='X', shape=(5,4)),
	]
	est = tf.estimator.Estimator( \
		lambda features, labels, mode, params, config : my_model_fn(features, feature_columns, labels, mode, params, config) \
		, model_dir='save_est/' \
		, params={'lr':0.01})

	# train
	count = 10
	X = [[random.random() for k in range(20)] for i in range(count)]
	# X = [np.array([random.random() for k in range(20)]).reshape(-1, 4) for i in range(count)]
	y = lambda x, i : int(np.sum(x[i]) % 10)
	Y = [y(X, i) for i in range(count)]
	est.train(lambda : my_input_fn(X, Y, 50, True), steps=100)

	# dump
	# print_x = lambda name : print("%s: \n%s" % (name, str(est.get_variable_value(name))))
	# names = est.get_variable_names()
	# for name in names:
	# 	if not 'Adam' in name:
	# 		print_x(name)

if __name__ == '__main__':
	main()
