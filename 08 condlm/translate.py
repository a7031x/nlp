import batch_enc_dec as ed
import tensorflow as tf
import numpy as np

def predict(texts):
    end_marks = ['.', '?']
    end_marks = [ed.w2i_trg[x] for x in end_marks]

    with tf.Graph().as_default():
        input_src = tf.placeholder(tf.int32, shape=[1, None], name='input_src')
        input_trg = tf.placeholder(tf.int32, shape=[1, None], name='input_trg')
        length_src = tf.placeholder(tf.int32, shape=[1], name='length_src')
        length_trg = tf.placeholder(tf.int32, shape=[1], name='length_trg')

        src_output, _ = ed.create_encoder(input_src, length_src)
        trg_output, trg_lengths, trg_labels = ed.create_decoder(src_output, input_trg, length_trg)

        sv = tf.train.Supervisor(logdir=ed.FLAGS.model_dir)
        with sv.managed_session() as sess:
            for text in texts:
                test_input = text.split(' ')
                test_input = [ed.w2i_src[x] if x in ed.w2i_src.keys() else ed.unk_src for x in test_input]
                test_output = [ed.eos_trg]
                while len(test_output) < 64:
                    feed = {
                        input_src: [test_input],
                        length_src: [len(test_input)],
                        input_trg: [test_output],
                        length_trg: [len(test_output)]
                    }
                    outputs_val = sess.run([trg_output], feed_dict=feed)
                    wids_evl = [np.argmax(x) for x in outputs_val[0][0]]
                    if wids_evl[-1] in end_marks:
                        break
                    test_output = wids_evl + [ed.eos_trg]
                print('input: ' + text)
                translated = [ed.i2w_trg[x] for x in wids_evl]
                print('output: ' + ' '.join(translated))

texts = [
    '君 は １ 日 で それ が でき ま す か 。',
    '皮肉 な 笑い を 浮かべ て 彼 は 私 を 見つめ た 。',
    '私 たち の 出発 の 時間 が 差し迫 っ て い る 。',
    'あなた は 午後 何 を し た い で す か 。',
    'いつ 仕事 が 終わ る 予定 で す か 。',
    '件 の 一 件 で メール を いただ き ありがとう ござ い ま し た 。',
    '５ 分 お 待 ち くださ い 。',
    '人 を 外見 で 判断 す べ き で は な い 。',
    '動物 園 から 一 頭 の トラ が 脱走 し た 。',
]
predict(texts)

