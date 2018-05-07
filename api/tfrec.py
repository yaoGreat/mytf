import tensorflow as tf
import numpy as np

filename = 'data_tfrec.tfrecord'

'''
https://blog.csdn.net/u012222949/article/details/72875281
目前看来，TFRecord中保存的数据都是一维的，读取之后，需要自己按照规则还原形状。
因为，协议定义如此：
message Example {  
    Features features = 1;  
};  
message Features {  
    map<string, Feature> feature = 1;  
};  
message Feature {  
	oneof kind {  
	    BytesList bytes_list = 1;  
	    FloatList float_list = 2;  
	    Int64List int64_list = 3;  
	}  
};  
'''
def write():
	cnt = 10
	f1 = [i for i in range(cnt)]
	f2 = [[i,i * 2] for i in range(cnt)]
	f3 = [[[i, i + 1] for k in range(10)] for i in range(cnt)]
	make_label = lambda i : f1[i] + np.sum(f2[i]) + np.sum(f3[i])
	label = [make_label(i) for i in range(cnt)]

	writer = tf.python_io.TFRecordWriter(filename)
	for i in range(cnt):
		features = {
			"f1":tf.train.Feature(int64_list=tf.train.Int64List(value=[f1[i]])),
			"f2":tf.train.Feature(int64_list=tf.train.Int64List(value=f2[i])),
			"f3":tf.train.Feature(int64_list=tf.train.Int64List(value=np.array(f3[i]).reshape([-1]))),
			"label":tf.train.Feature(int64_list=tf.train.Int64List(value=[label[i]]))
		}
		example = tf.train.Example(features = tf.train.Features(feature = features))
		writer.write(example.SerializeToString())

	writer.close()
	print(f1)
	print(f2)
	print(f3)
	print(label)

def read():
	feature = {
		"f1":tf.FixedLenFeature([], tf.int64),
		"label":tf.FixedLenFeature([], tf.int64),
	}
	filename_queue = tf.train.string_input_producer([filename], num_epochs=1)
	reader = tf.TFRecordReader()
	key, value = reader.read(filename_queue)
	features = tf.parse_single_example(value, features=feature)
	print(features)

	tf.InteractiveSession()
	features['f1'].eval() # 这句话卡住，现在不知道什么原因

def read_ds():
	ds = my_input_fn()
	it = ds.make_one_shot_iterator().get_next()
	tf.InteractiveSession()
	print(it)
	for i in range(1):
		print("f1: " + str(it[0]['f1'].eval()))
		print("f2: " + str(it[0]['f2'].eval()))
		print("f3: " + str(it[0]['f3'].eval()))
		print("label: " + str(it[1].eval()))

def my_input_fn():
	# 这段代码是测试有效的
	def _parse_ex(exam):
		feature = {
			"f1":tf.FixedLenFeature([], tf.int64),
			"f2":tf.FixedLenFeature([2], tf.int64),
			"f3":tf.FixedLenFeature([10,2], tf.int64),
			"label":tf.FixedLenFeature([], tf.int64),
		}
		parsed = tf.parse_single_example(exam, feature)
		label = parsed.pop('label')
		return parsed, label

	filenames = [filename]
	ds = tf.data.TFRecordDataset(filenames).map(_parse_ex)
	ds.repeat().batch(5)
	return ds

if __name__ == '__main__':
	#write()
	read_ds()
	pass
