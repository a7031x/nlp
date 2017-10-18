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


def main(_):
    #a: 3x2x3
    a = tf.convert_to_tensor(
        [
            [[1, 2, 3], [4, 5, 6]],
            [[1, 3, 5], [2, 4, 6]],
            [[1, 2, 3], [6, 5, 4]]
        ])
    b = tf.convert_to_tensor(
        [
            [1, 2, 3]
        ])
    c = a + b

    d0 = tf.convert_to_tensor([0, 1, 2])
    d1 = tf.convert_to_tensor([1, 1, 1])
    indices = tf.stack([d0, d1])
    indices = tf.transpose(indices)
    d = tf.gather_nd(a, indices)
    e = tf.reduce_sum(a, axis=1)

    with tf.Session() as sess:
        welcome = sess.run(e)
        print(welcome)
    exit(0)


if __name__ == "__main__":
    tf.app.run()
