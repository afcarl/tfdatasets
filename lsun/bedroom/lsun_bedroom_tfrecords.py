import os
import tensorflow as tf
import numpy as np
from glob import glob
from tqdm import tqdm


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


if __name__ == '__main__':
    paths = glob(os.path.join('/data/data/lsun/bedroom', "*.npy"))
    it = range(0, len(paths))

    writers = [
        tf.python_io.TFRecordWriter('records/lsun_bedroom_{}.tfrecords'.format(i)) for i in range(0, 150)
    ]

    cnt = 0
    for i, path in enumerate(paths):
        img = np.load(path)
        img = (img * 256).astype(np.uint8)
        for j in tqdm(range(0, img.shape[0])):
            img_raw = img[j].tostring()

            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'image_raw': _bytes_feature(img_raw)
            }))
            writers[cnt % 150].write(example.SerializeToString())
            cnt += 1

    for i in range(0, 150):
        writers[i].close()