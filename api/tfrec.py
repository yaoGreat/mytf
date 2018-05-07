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
	make_label = lambda i : f1[i]
	label = [make_label(i) for i in range(cnt)]

	writer = tf.python_io.TFRecordWriter(filename)
	for i in range(cnt):
		features = {
			"f1":tf.train.Feature(int64_list=tf.train.Int64List(value=[f1[i]])),
			"label":tf.train.Feature(int64_list=tf.train.Int64List(value=[label[i]]))
		}
		example = tf.train.Example(features = tf.train.Features(feature = features))
		writer.write(example.SerializeToString())

	writer.close()

def read():
	feature = {
		"f1":tf.FixedLenFeature([], tf.int64),
		"label":tf.FixedLenFeature([], tf.int64),
	}
	filename_queue = tf.train.string_input_producer([filename], num_epochs=1)
	reader = tf.TFRecordReader()
	key, value = reader.read(filename_queue)
	print(key)
	print(value)

	tf.InteractiveSession()
	print(key.eval())
	print(value.eval())

if __name__ == '__main__':
	# write()
	read()
	pass
