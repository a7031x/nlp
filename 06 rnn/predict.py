import rnn
import numpy as np
import tensorflow as tf

tf.app.flags.DEFINE_string('model_dir', 'model', 'Model directory where final model files are saved.')

train, valid, test = rnn.read()
initializer = tf.random_uniform_initializer(-rnn.INIT_SCALE, rnn.INIT_SCALE)

def predict(text):
    with tf.Graph().as_default():
        with tf.name_scope('Valid'):
            text = text.split(' ')
            test_input = [rnn.w2i[x] if x in rnn.w2i.keys() else rnn.UNK for x in text]
            if len(test_input) < rnn.NUMBER_STEPS:
                test_input = [rnn.S] * (rnn.NUMBER_STEPS - len(test_input)) + test_input
            test_input = tf.convert_to_tensor([test_input], dtype=tf.int32)
            with tf.variable_scope('Model', reuse=None, initializer=initializer):
                train_model = rnn.PTBModel(is_training=False, input=test_input, label=None)

        sv = tf.train.Supervisor(logdir=rnn.FLAGS.model_dir)
        with sv.managed_session() as session:
            fetches = {
                'logits': train_model.logits
            }
            vals = session.run(fetches)
            logits = vals['logits']
            logits = [sorted(zip(range(len(x)), x.tolist()), key=lambda p: -p[1]) for x in logits[-len(text):]]
            wids = [x[:8] for x in logits]
            words = [[rnn.i2w[y[0]] for y in x] for x in wids]
            for w, p in zip(text, words):
                print('{}: {}'.format(w, p))

predict('')
