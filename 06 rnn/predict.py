import rnn
import numpy as np
import tensorflow as tf

train, valid, test = rnn.read()
initializer = tf.random_uniform_initializer(-rnn.INIT_SCALE, rnn.INIT_SCALE)

with tf.Graph().as_default():

    with tf.name_scope('Valid'):
        #test_input = ['i', 'love']
        test_input = 'the most impressive part of this movie is'
        test_input = test_input.split(' ')
        test_input = [rnn.w2i[x] for x in test_input]
        if len(test_input) < rnn.NUMBER_STEPS:
            test_input = [rnn.S] * (rnn.NUMBER_STEPS - len(test_input)) + test_input
        test_input = tf.convert_to_tensor([test_input], dtype=tf.int32)
        with tf.variable_scope('Model', reuse=None, initializer=initializer):
            train_model = rnn.PTBModel(is_training=False, input=test_input, label=None)

    sv = tf.train.Supervisor(logdir=rnn.FLAGS.output_dir)
    with sv.managed_session() as session:
        fetches = {
            'logits': train_model.logits
        }
        vals = session.run(fetches)
        logits = vals['logits']
        logits = [x.tolist() for x in logits]
        wids = [x.index(max(x)) for x in logits]
        words = [rnn.i2w[x] for x in wids]
        print(words)

        