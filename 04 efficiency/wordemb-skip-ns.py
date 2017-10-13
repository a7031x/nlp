from collections import defaultdict
from collections import Counter
import numpy as np
import sys
import os
import random
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

WINDOW_SIZE=2 #length of window on each side (so N=2 gives a total window size of 5, as in t-2 t-1 t t+1 t+2)
EMB_SIZE = 256 # The size of the embedding
NUM_SAMPLED = 100
BATCH_SIZE = 128
EPOCHS = 1000

VALID_SIZE = 16
VALID_WINDOW = 100
# We reuse the data reading from the language modeling class
w2i = defaultdict(lambda: len(w2i))
S = w2i["<s>"]
UNK = w2i["<unk>"]
i2w = {}
valid_samples = []
word_counts = defaultdict(int)
graph = tf.Graph()

def read_dataset(filename):
    global w2i
    with open(filename, 'r') as f:
        lines = []
        for line in f:
            line = line.strip().split(' ')
            for word in line:
                word_counts[w2i[word]] += 1
            lines.extend(line)
        return [w2i[x] for x in lines]


def read():
    global w2i, i2w
    train_path = os.path.join(FLAGS.input_dir, 'ptb/train.txt')
    train = read_dataset(train_path)
    w2i = defaultdict(lambda: UNK, w2i)
    valid_path = os.path.join(FLAGS.input_dir, 'ptb/valid.txt')
    valid = read_dataset(valid_path)
    i2w = {v: k for k, v in w2i.items()}

    nwords = len (w2i)
    counts =  np.array([list(x) for x in word_counts.items()])[:,1]**.75
    normalizing_constant = sum(counts)
    word_probabilities = np.zeros(nwords)
    for word_id in word_counts:
        word_probabilities[word_id] = word_counts[word_id]**.75/normalizing_constant
    return train, valid


def preprocess(words, freq = 5):
    word_counts = Counter(words)
    return [word if word_counts[word] > freq else UNK for word in words]


def create_model(input):
    with graph.as_default():
        nwords = len(w2i)
        embedding = tf.Variable(tf.random_uniform([nwords, EMB_SIZE], -1.0, 1.0), dtype=tf.float32, name='embedding')
        embed = tf.nn.embedding_lookup(embedding, input, name='embed')
        return embed, embedding


def create_loss(embed, labels):
    with graph.as_default():
        nwords = len(w2i)
        softmax_w = tf.Variable(tf.truncated_normal([nwords, EMB_SIZE], stddev=0.1), dtype=tf.float32, name='softmax_w')
        softmax_b = tf.Variable(tf.zeros(nwords))
        loss = tf.nn.sampled_softmax_loss(softmax_w, softmax_b, labels, embed, NUM_SAMPLED, nwords)
        cost = tf.reduce_mean(loss)
        return cost


def create_similarity(embedding):
    global valid_samples
    valid_samples0 = np.array(random.sample(range(30, 30 + VALID_WINDOW), VALID_SIZE // 2))
    valid_samples1 = np.array(random.sample(range(1000, 1000 + VALID_WINDOW), VALID_SIZE // 2))
    valid_samples = np.append(valid_samples0, valid_samples1)
    valid_samples[0] = w2i['two']
    valid_samples[1] = w2i['three']
    valid_size = len(valid_samples)
    valid_dataset = tf.constant(valid_samples, dtype=tf.int32)
    norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keep_dims=True))
    normalized_embedding = embedding / norm
    valid_embedding = tf.nn.embedding_lookup(normalized_embedding, valid_dataset)
    similarity = tf.matmul(valid_embedding, normalized_embedding, transpose_b=True)
    return similarity


def remove_high_freq_words(words):
    t = 1e-5
    threshold = 0.95
    word_counts = Counter(words)
    total_count = len(words)
    word_freqs = {w: c/total_count for w, c in word_counts.items()}
    prob_drop = {w: 1 - np.sqrt(t/f) for w, f in word_freqs.items()}
    removed_words = set([w for w in words if prob_drop[w] >= threshold])
    removed_words = [i2w[w] for w in removed_words]
    print('removed words: {}'.format(removed_words))
    return [w for w in words if prob_drop[w] < threshold]


def get_targets(words, index, window_size):
    target_window = np.random.randint(1, window_size + 1)
    start = index - target_window if index > target_window else 0
    end = index + target_window
    targets = set(words[start:index] + words[index+1:end+1])
    return list(targets)


def get_batches(words, batch_size, window_size):
    n_batches = len(words) // batch_size
    words = words[:n_batches*batch_size]
    for idx in range(0, len(words), batch_size):
        x, y = [], []
        batch = words[idx:idx+batch_size]
        for i in range(batch_size):
            batch_x = batch[i]
            batch_y = get_targets(batch, i, window_size)
            x.extend([batch_x] * len(batch_y))
            y.extend(batch_y)
        yield x, y


def main(_):
    train, valid = read()
    train = preprocess(train)
    valid = preprocess(valid)
    print('total words: {}'.format(len(train)))
    print('unique words: {}'.format(len(set(train))))
    print('vocab size: {}'.format(len(w2i)))

    trimmed_train = remove_high_freq_words(train)
    print('trimmed words: {}'.format(len(trimmed_train)))
    print('unique trimmed words: {}'.format(len(set(trimmed_train))))
    print(trimmed_train[:20])

    labels_location = os.path.join(FLAGS.output_dir, 'labels.txt')
    with open(labels_location, 'w') as labels_file:
      for i in range(len(i2w)):
        labels_file.write(i2w[i] + '\n')

    batch_size = BATCH_SIZE
    window_size = WINDOW_SIZE
    with graph.as_default():
        inputs = tf.placeholder(tf.int32, shape=[None])
        labels = tf.placeholder(tf.int32, shape=[None, 1])
        embed, embedding = create_model(inputs)
        cost = create_loss(embed, labels)
        optimizer = tf.train.AdamOptimizer().minimize(cost)
        similarity = create_similarity(embedding)
        iteration = 0
        loss = 0
        checkpoint_dir = FLAGS.output_dir
        saver = tf.train.Saver()

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('model resored')

            for e in range(EPOCHS):
                batches = get_batches(trimmed_train, batch_size, window_size)
                for x, y in batches:
                    iteration += 1
                    feed = {
                        inputs: x,
                        labels: np.array(y)[:, None]
                    }
                    train_loss, _ = sess.run([cost, optimizer], feed_dict=feed)
                    loss += train_loss
                    evaluate_size = 1000
                    if iteration % evaluate_size == 0:
                        print('epoch {}/{}, iteration {}'.format(e, EPOCHS, iteration),
                              'avg loss {:.4f}'.format(loss / evaluate_size))
                        loss = 0

                    if iteration % 1000 == 0:
                        sim = similarity.eval()
                        for i in range(VALID_SIZE):
                            valid_word = i2w[valid_samples[i]]
                            top_k = 8
                            nearest = (-sim[i, :]).argsort()[1:top_k+1]
                            log = valid_word + ':'
                            for iw in nearest:
                                log += ' ' + i2w[iw]
                            print(log)
                        saver.save(sess, os.path.join(checkpoint_dir, 'model.ckpt'), global_step=iteration)


if __name__ == "__main__":
    tf.app.run()
#tf.app.run(main=main)
