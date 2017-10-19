from bilstm_tagger import *
import tensorflow as tf

#text = 'This album proved to be more commercial and more techno-based than Osc-Dis, with heavily synthesized songs .'
text = 'I love you .'
with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-1, 1)
    inputs = tf.placeholder(tf.int32, [None])

    with tf.name_scope('Valid'):
        with tf.variable_scope('Model', reuse=None, initializer=initializer):
            scores = create_model(inputs, None)

    sv = tf.train.Supervisor(logdir=FLAGS.model_dir)
    with sv.managed_session() as sess:
            feed = {
                inputs: [w2i[x] for x in text.split(' ')],
            }
            scores_val = sess.run([scores], feed_dict=feed)[0]
            predictions = np.argmax(scores_val, 1)
            print(predictions)