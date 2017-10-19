
import attention as ed
import tensorflow as tf
import numpy as np
import html

def predict(texts):
    end_marks = ['.', '?', '!']
    end_marks = [ed.w2i_trg[x] for x in end_marks]

    with tf.Graph().as_default():
        input_src = tf.placeholder(tf.int32, shape=[1, ed.MAX_TIMESTEPS], name='input_src')
        input_trg = tf.placeholder(tf.int32, shape=[1, ed.MAX_TIMESTEPS], name='input_trg')
        length_src = tf.placeholder(tf.int32, shape=[1], name='length_src')
        length_trg = tf.placeholder(tf.int32, shape=[1], name='length_trg')

        src_outputs, src_last_output = ed.create_encoder(input_src, length_src, False)
        trg_output, trg_lengths, trg_labels = ed.create_decoder(src_outputs, src_last_output, input_trg, length_trg, False)

        sv = tf.train.Supervisor(logdir=ed.FLAGS.model_dir)
        with sv.managed_session() as sess:
            for text in texts:
                test_input = text.split(' ')
                test_input = [ed.w2i_src[x] if x in ed.w2i_src.keys() else ed.unk_src for x in test_input]
                test_output = [ed.eos_trg]
                while len(test_output) < 64:
                    feed = {
                        input_src: [test_input + [ed.eos_src] * (ed.MAX_TIMESTEPS - len(test_input))],
                        length_src: [len(test_input)],
                        input_trg: [test_output + [ed.eos_trg] * (ed.MAX_TIMESTEPS - len(test_output))],
                        length_trg: [len(test_output)]
                    }
                    outputs_val = sess.run([trg_output], feed_dict=feed)
                    wids_evl = [np.argmax(x) for x in outputs_val[0][0]]
                    if wids_evl[-1] in end_marks:
                        break
                    test_output = wids_evl + [ed.eos_trg]
                print('input: ' + text)
                translated = [ed.i2w_trg[x] for x in wids_evl]
                translated = [html.unescape(x) for x in translated]
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
    'これ は 主要 な 決定 要素 が 存在 し て い な い 興味 深 い 例 で あ る 。',
    '布団 を たた み なさ い 。',
    '僕 は これ から 彼 に 重大 な 申し出 を する つもり で す 。',
    '遅く まで 音楽 かけ て い る けど 、 ご 近所 さん は 平気 ？',
    '私 は 普通 六 時 まで に は 帰宅 する 。',
    'この 映画 は 何 度 も 見 る 価値 が あ る 。',
    '式 の ため の 準備 が 進行 中 で あ る 。',
    '困 っ た こと に 、 私 は その 仕事 に 耐え られ な い 。',
    '私 の 両親 も また 農場 を 持 っ て い る 。',
    '君 は しょっちゅう 間違い を し て い る 。',
    '彼女 は 慈善 伝導 団 と 呼 ば れ る 修道 女 達 の 集団 を 指導 し て い た 。',
]
predict(texts)

