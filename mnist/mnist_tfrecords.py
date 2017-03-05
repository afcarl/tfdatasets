from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/data/mnist')

data = mnist.train.images
labels = mnist.train.labels

writers = [
    tf.python_io.TFRecordWriter('records/mnist_{}.tfrecords'.format(i)) for i in range(0, 30)
]
for i, example_idx in enumerate(tqdm(range(0, data.shape[0]))):
    features = data[example_idx]
    label = labels[example_idx]

    example = tf.train.Example(
        features=tf.train.Features(
          feature={
            'label': tf.train.Feature(
                int64_list=tf.train.Int64List(value=[label])),
            'image': tf.train.Feature(
                float_list=tf.train.FloatList(value=features.astype("float"))),
    }))
    serialized = example.SerializeToString()
    writers[i%30].write(serialized)

for i in range(0, 30):
    writers[i].close()

