import os
import tensorflow as tf
from glob import glob
from scipy import misc
from tqdm import tqdm


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def center_crop(x, crop_h=108, crop_w=None, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h) / 2.))
    i = int(round((w - crop_w) / 2.))
    return misc.imresize(x[j:j + crop_h, i:i + crop_w],
                         [resize_w, resize_w])


if __name__ == '__main__':
    paths = glob(os.path.join('/ssd_data/CelebA', "*.jpg"))
    it = range(0, len(paths))

    writers = [
        tf.python_io.TFRecordWriter('records/celeba_{}.tfrecords'.format(i)) for i in range(0, 100)
    ]

    for i, path in enumerate(tqdm(paths)):
        img = center_crop(misc.imread(path))
        #img = img / 127.5 - 1.
        height = img.shape[0]
        width = img.shape[1]

        img_raw = img.tostring()

        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    # 'height': _int64_feature(height),
                    # 'width': _int64_feature(width),
                    'image_raw': _bytes_feature(img_raw)
        }))

        writers[i % 100].write(example.SerializeToString())

    for i in range(0, 100):
        writers[i].close()