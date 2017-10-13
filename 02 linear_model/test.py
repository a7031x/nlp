from collections import defaultdict
import math
import time
import random
import numpy as np
import os
import sys
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("input_dir", "input", "Input directory where training dataset and meta data are saved")
tf.app.flags.DEFINE_string("output_dir", "output", "Output directory where output such as logs are saved.")
tf.app.flags.DEFINE_string("model_dir", "output", "Model directory where final model files are saved.")

# The length of the n-gram
N = 2

# Functions to read in the corpus
# NOTE: We are using data from the Penn Treebank, which is already converted
#       into an easy-to-use format with "<unk>" symbols. If we were using other
#       data we would have to do pre-processing and consider how to choose
#       unknown words, etc.
w2i = defaultdict(lambda: len(w2i))
S = w2i["<s>"]
UNK = w2i["<unk>"]
def read_dataset(filename):
  with open(filename, "r") as f:
    for line in f:
      yield [w2i[x] for x in line.strip().split(" ")]

# Read in the data
train_path = os.path.join(FLAGS.input_dir, "ptb/train.txt")
train = list(read_dataset(train_path))
w2i = defaultdict(lambda: UNK, w2i)
valid_path = os.path.join(FLAGS.input_dir, "ptb/valid.txt")
dev = list(read_dataset(valid_path))
i2w = {v: k for k, v in w2i.items()}

joined = []
xx = tf.convert_to_tensor(train, dtype=tf.int32)
for sent_id, sent in enumerate(train):
    print([sent_id, sent])