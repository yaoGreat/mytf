import tensorflow as tf
import pandas as pd
import numpy as np
import random

'''
 |        model_fn: Model function. Follows the signature:
 |      
 |          * Args:
 |      
 |            * `features`: This is the first item returned from the `input_fn`
 |                   passed to `train`, `evaluate`, and `predict`. This should be a
 |                   single `Tensor` or `dict` of same.
 |            * `labels`: This is the second item returned from the `input_fn`
 |                   passed to `train`, `evaluate`, and `predict`. This should be a
 |                   single `Tensor` or `dict` of same (for multi-head models). If
 |                   mode is `ModeKeys.PREDICT`, `labels=None` will be passed. If
 |                   the `model_fn`'s signature does not accept `mode`, the
 |                   `model_fn` must still be able to handle `labels=None`.
 |            * `mode`: Optional. Specifies if this training, evaluation or
 |                   prediction. See `ModeKeys`.
 |            * `params`: Optional `dict` of hyperparameters.  Will receive what
 |                   is passed to Estimator in `params` parameter. This allows
 |                   to configure Estimators from hyper parameter tuning.
 |            * `config`: Optional configuration object. Will receive what is passed
 |                   to Estimator in `config` parameter, or the default `config`.
 |                   Allows updating things in your model_fn based on configuration
 |                   such as `num_ps_replicas`, or `model_dir`.
 |      
 |          * Returns:
 |            `EstimatorSpec`

 class ModeKeys(builtins.object)
 |  Standard names for model modes.
 |  
 |  The following standard keys are defined:
 |  
 |  * `TRAIN`: training mode.
 |  * `EVAL`: evaluation mode.
 |  * `PREDICT`: inference mode.
 |  
 |  Data and other attributes defined here:
 |  
 |  EVAL = 'eval'
 |  
 |  PREDICT = 'infer'
 |  
 |  TRAIN = 'train'

'''

def my_model_fn(features, labels, mode, params, config):
	X = features
	Y = labels
	# 如果input_fn返回的是list，那么这里接收到的就是list，而不是tensor
	# 
	# 如果input_fn中调用了convert_to_tensor，那么这里的输出是：
	# X:Tensor("Const:0", shape=(500, 1), dtype=float32, device=/device:CPU:0)
	# Y:Tensor("Const_1:0", shape=(500,), dtype=float32, device=/device:CPU:0)
	# 
	# 如果input_fn使用了batch的dataset，那么输出：
	# X:Tensor("IteratorGetNext:0", shape=(?, 1), dtype=float32)
	# Y:Tensor("IteratorGetNext:1", shape=(?,), dtype=float32)
	print("X:" + str(X))
	print("Y:" + str(Y))
	W = tf.Variable(tf.truncated_normal((1,1)))
	b = tf.Variable(tf.truncated_normal((1,)))
	_Y = tf.matmul(X, W) + b
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

'''
 |        input_fn: A function that constructs the input data for evaluation.
 |          See @{$get_started/premade_estimators#create_input_functions} for more
 |          information. The function should construct and return one of
 |          the following:
 |      
 |            * A 'tf.data.Dataset' object: Outputs of `Dataset` object must be a
 |              tuple (features, labels) with same constraints as below.
 |            * A tuple (features, labels): Where features is a `Tensor` or a
 |              dictionary of string feature name to `Tensor` and labels is a
 |              `Tensor` or a dictionary of string label name to `Tensor`. Both
 |              features and labels are consumed by `model_fn`. They should satisfy
 |              the expectation of `model_fn` from inputs.

'''
# 目前来看，input_fn仅被调用一次，就是说，每次都是使用相同的数据进行训练，没有批次的概念。
# 如果分批次，就是下一步需要研究的了
def my_input_fn():
	features = [[random.random() * 100] for i in range(500)]
	labels = [x[0] * 5 + 2 for x in features]
	print("make train inputs")

	# return (features, labels)
	return (tf.convert_to_tensor(features), tf.convert_to_tensor(labels))

def my_input_fn_ds():
	features = [[random.random() * 100] for i in range(50000)]
	labels = [(x[0] * 5 + 2) for x in features]
	print("make train inputs dataset")
	# 使用每一个切片来构造数据，然后分批次
	dataset = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(features), tf.convert_to_tensor(labels)))
	dataset = dataset.batch(50).repeat(100)
	return dataset

def my_pred_input_fn():
	features = [[float(i)] for i in range(20)]
	print("make pred inputs")
	print(features)
	return (features, )

def main():
	est = tf.estimator.Estimator(my_model_fn, model_dir='save_est/')
	# est.train(my_input_fn, steps=100)
	est.train(my_input_fn_ds, steps=1000)
	preds = est.predict(my_pred_input_fn)

	for i in range(20):
		print(next(preds))

	names = est.get_variable_names()
	for name in names:
		print("%s: %f" % (name, est.get_variable_value(name)))

if __name__ == '__main__':
	main()
