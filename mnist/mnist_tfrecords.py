from skdata.mnist.views import OfficialVectorClassification
from tqdm import tqdm
import numpy as np
import tensorflow as tf

data = OfficialVectorClassification()
trIdx = data.sel_idxs[:]

np.random.shuffle(trIdx)


writer = tf.python_io.TFRecordWriter("mnist.tfrecords")
for example_idx in tqdm(trIdx):
    features = data.all_vectors[example_idx]
    label = data.all_labels[example_idx]

    example = tf.train.Example(
        features=tf.train.Features(
          feature={
            'label': tf.train.Feature(
                int64_list=tf.train.Int64List(value=[label])),
            'image': tf.train.Feature(
                float_list=tf.train.FloatList(value=features.astype("float"))),
    }))
    serialized = example.SerializeToString()
    writer.write(serialized)
writer.close()
