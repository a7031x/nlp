from collections import defaultdict

# Functions to read in the corpus
w2i = defaultdict(lambda: len(w2i))
t2i = defaultdict(lambda: len(t2i))
UNK = w2i["<unk>"]
def read_dataset(filename):
  with open(filename, "r") as f:
    for line in f:
      tag, words = line.lower().strip().split(" ||| ")
      yield ([w2i[x] for x in words.split(" ")], t2i[tag])

# Read in the data
train = list(read_dataset("input/classes/train.txt"))
w2i = defaultdict(lambda: UNK, w2i)
dev = list(read_dataset("input/classes/test.txt"))
i2t = dict((v, k) for k, v in t2i.items())

r = [1.0, 2.0, 3.0, 4.0, 5.0]
r = [x[0] for x in sorted(zip(r, range(len(r))), key=lambda x: i2t[x[1]])]
nwords = len(w2i)
ntags = len(t2i)
