import numpy as np
import sys
import os
import tensorflow as tf

###################################################################
# Variables                                                       #
# When launching project or scripts from Visual Studio,           #
# input_dir and output_dir are passed as arguments automatically. #
# Users could set them from the project setting page.             #
###################################################################

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("input_dir", "input", "Input directory where training dataset and meta data are saved")
tf.app.flags.DEFINE_string("output_dir", "output", "Output directory where output such as logs are saved.")
tf.app.flags.DEFINE_string("model_dir", "output", "Model directory where final model files are saved.")


def fn1(x):
    r = tf.one_hot(x[2] + 1, 5, dtype=tf.float32)
    return r


def main(_):
    #a: 3x2x3
    a = tf.convert_to_tensor(
        [
            [[1, 2, 3], [4, 5, 6]],
            [[1, 3, 5], [2, 4, 6]],
            [[1, 2, 3], [6, 5, 4]]
        ], dtype=tf.int32)
    b = tf.convert_to_tensor(
        [
            [1, 2, 3]
        ], dtype=tf.int32)
    c = a + b

    a = tf.convert_to_tensor([1, 2, 3])
    b = tf.convert_to_tensor([2, 2, 3])

    fn = lambda x: tf.equal(x[1], 2)

    with tf.Session() as sess:
        initializer = tf.global_variables_initializer()
        sess.run(initializer)
        r = tf.map_fn(fn, (a, b, c), dtype=tf.bool)
        print(sess.run(r))
    exit(0)


if __name__ == "__main__":
    tf.app.run()
