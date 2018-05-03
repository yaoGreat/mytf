import tensorflow as tf
import pandas as pd
import numpy as np
import random

FEATURE_COUNT = 638

# 模型函数的写法
def my_model_fn(features, labels, mode, params, config):
	X = features['X']
	Y = labels
	print("X:" + str(X))
	print("Y:" + str(Y))
	print("params:" + str(params))
	x_len = int(X.shape[1])
	W = tf.Variable(tf.truncated_normal((x_len,1)), name="W")
	b = tf.Variable(tf.constant(0.1, shape=(1,)), name="b")
	_Y = tf.reshape(tf.matmul(X, W) + b, (-1,))
	if not Y is None:
		loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=_Y, labels=Y))

		# optimizer = tf.train.AdamOptimizer(params['lr'])
		optimizer = tf.train.GradientDescentOptimizer(params['lr'])
		train_op = optimizer.minimize(loss, global_step = tf.train.get_global_step())
	else:
		loss = None
		train_op = None
	return tf.estimator.EstimatorSpec(
		mode = mode,
		predictions = tf.nn.sigmoid(_Y),
		loss = loss,
		train_op = train_op,
		)

# 输入函数的写法
def my_input_fn_csv(filename, feature_cnt):
	
	def _parse_line(line):
		# 此函数中接受到的line参数是一个tensor，而不是数值。所以，这个函数应该是一次性构建的函数。
		print("in parse line")
		cols = ['id', 'label']
		for i in range(feature_cnt):
			cols.append("f%d" % i)
		defs = [[0.] for i in range(feature_cnt + 2)] # 这里的成员要有一个维度，而不能是scalar

		# print(line)
		# print(defs)
		# 这个函数解析出数据列，然后至于如何组合，就看后买呢自己的做法了
		fields = tf.decode_csv(line, defs)
		'''
		# 注释中是官方做法，每个数据作为一个数据列，我来修改一下
		features = dict(zip(cols, fields))

		features.pop('id')
		label = features.pop('label')
		'''
		# 讲数据列合并成向量，然后标签列单独提出。
		# 这里将来还可以使用特征列的api，应该会更方便
		assert len(fields) == feature_cnt + 2
		features = {'X':tf.stack(fields[2:])}
		label = fields[1]

		return features, label

	ds = tf.data.TextLineDataset(filename)
	ds = ds.map(_parse_line)
	ds = ds.shuffle(1000).repeat()
	ds = ds.batch(100)
	'''
	测试由此函数创建的ds
>>> ds
<BatchDataset shapes: ({X: (?, 638)}, (?,)), types: ({X: tf.float32}, tf.float32)>
>>> it = ds.make_one_shot_iterator().get_next()
>>> it
({'X': <tf.Tensor 'IteratorGetNext_10:0' shape=(?, 638) dtype=float32>}, <tf.Tensor 'IteratorGetNext_10:1' shape=(?,) dtype=float32>)
>>> it[0]['X'].eval()
array([[2.398, 1.099, 3.045, ..., 0.   , 0.   , 0.   ],
       [2.197, 0.693, 2.398, ..., 0.   , 0.   , 0.   ],
       [2.079, 0.693, 2.398, ..., 0.   , 0.   , 0.   ],
       ...,
       [2.079, 0.693, 2.398, ..., 0.   , 0.   , 0.   ],
       [2.833, 0.693, 2.398, ..., 0.   , 0.   , 0.   ],
       [2.079, 0.693, 2.398, ..., 0.   , 0.   , 0.   ]], dtype=float32)
>>> it[0]['X'].eval()
array([[2.398, 0.693, 2.398, ..., 0.   , 0.   , 0.   ],
       [1.792, 0.693, 2.398, ..., 0.   , 0.   , 0.   ],
       [2.197, 0.693, 2.398, ..., 0.   , 0.   , 0.   ],
       ...,
       [3.091, 0.693, 2.398, ..., 0.   , 0.   , 0.   ],
       [2.773, 0.693, 2.398, ..., 0.   , 0.   , 0.   ],
       [1.792, 0.693, 2.398, ..., 0.   , 0.   , 0.   ]], dtype=float32)
>>> it[1].eval()
array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
      dtype=float32)
	'''
	return ds


def main():
	global FEATURE_COUNT

	tf.logging.set_verbosity(tf.logging.INFO) # 打开这句话，可以让训练过程中输出loss和步骤等信息
	
	est = tf.estimator.Estimator(my_model_fn, model_dir='save_est/', params={'lr':0.01})
	# train
	est.train(lambda : my_input_fn_csv('data.csv', FEATURE_COUNT), steps=1000)

if __name__ == '__main__':
	main()
