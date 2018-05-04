import tensorflow as tf
import pandas as pd
import numpy as np
import random

FEATURE_COUNT = 638

# 模型函数的写法
def my_model_fn(features, feature_columns, labels, mode, params, config):
	print("features:" + str(features))
	print("labels:" + str(labels))
	print("params:" + str(params))
	print("feature_columns" + str(feature_columns))
	Y = labels
	'''
	这里的 _input 会把所有的feature合并起来，如果想要对X进行单独处理，就有问题了。
	不过，应该也可以先拆分，再他们进行单独的处理
	经过测试，[X,T]的特征列合并之后，输出的顺序是[T0,T1,T2,X0,X1]，居然是颠倒的
	'''
	_input = tf.feature_column.input_layer(features, feature_columns)
	print("input", _input)
	hidden = tf.layers.dense(_input, 4, activation=tf.nn.tanh, name='hidden')
	output = tf.layers.dense(hidden, 1, name='output')
	_Y = tf.layers.flatten(output)
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

# 输入函数的写法
def my_input_fn(X, T, K, Y, batch, repeat):
	dataset = tf.data.Dataset.from_tensor_slices((
		{'X':X, 'T':T, 'K':K},
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
	可以用tf.estimator.RunConfig来控制日志输出频率，模型保存频率等参数
	甚至还包括worker、并行等
	'''
	
	count = 100
	X = [[random.random(), random.random()] for i in range(count)]
	T = [[random.random(), random.random(), random.random()] for i in range(count)]
	K = [[random.random() - 0.5] for i in range(count)]
	y = lambda x, t, k, i : x[i][0] * 2 + x[i][1] * 3 \
					+ t[i][0] * 4 + t[i][1] * 5 + t[i][2] * 6 \
					+ (10) if k[i][0] > 0 else (0) \
					+ 7
	Y = [y(X, T, K, i) for i in range(count)]

	'''
	numeric_column 的 shape:
	shape: An iterable of integers specifies the shape of the `Tensor`. An
	        integer can be given which means a single dimension `Tensor` with given
	        width. The `Tensor` representing the column will have the shape of
	        [batch_size] + `shape`.
	'''
	feature_columns = [
		tf.feature_column.numeric_column(key='X', shape=2),
		tf.feature_column.numeric_column(key='T', shape=3),
		tf.feature_column.bucketized_column(tf.feature_column.numeric_column(key='K', shape=1), [0.]),
	]
	est = tf.estimator.Estimator( \
		lambda features, labels, mode, params, config : my_model_fn(features, feature_columns, labels, mode, params, config) \
		, model_dir='save_est/' \
		, params={'lr':0.01})

	# train
	est.train(lambda : my_input_fn(X, T, K, Y, 50, True), steps=1000)

	# dump
	print_x = lambda name : print("%s: \n%s" % (name, str(est.get_variable_value(name))))
	names = est.get_variable_names()
	for name in names:
		if not 'Adam' in name:
			print_x(name)

if __name__ == '__main__':
	main()
