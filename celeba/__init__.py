import tensorflow as tf
import time
from tqdm import tqdm


def read_and_decode_single_example(filenames):
    filename_queue = tf.train.string_input_producer(filenames,
                                                    num_epochs=None)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            # 'height': tf.FixedLenFeature([], tf.int64),
            # 'width': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string)
        })
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image = tf.reshape(image, [64, 64, 3])
    return byte_to_data(image)


def read_and_decode(path):
    filenames = ['{}_{}.tfrecords'.format(path, i) for i in range(0, 100)]
    return read_and_decode_single_example(filenames)


def byte_to_data(image):
    return tf.divide(tf.to_float(image), 127.5) - 1.0


def data_to_image(image):
    rescaled = tf.divide(image + 1.0, 2.0)
    return tf.clip_by_value(rescaled, 0.0, 1.0)


if __name__ == '__main__':
    image = read_and_decode('/ssd_data/celeba_tfrecords/celeba')
    image_batch = tf.train.shuffle_batch([image], batch_size=128, capacity=11024, min_after_dequeue=10000, num_threads=1)
    display = data_to_image(image_batch)
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        start = time.time()

        for i in tqdm(range(0, 1000)):
            img = sess.run(display)

        end = time.time()
        print(end-start)
        coord.request_stop()
        coord.join(threads)