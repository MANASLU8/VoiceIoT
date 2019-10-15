import os, re, sys
import json
import pandas as pd
from nltk.corpus import stopwords
import re
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
import itertools
from gensim.models import KeyedVectors
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from .. import utils, file_operators as fo
from . import normalize

config = utils.load_config(utils.parse_args().config)
DEFAULT_MODEL_PATH = "~/models/ArModel100w2v.txt"#config['paths']['models']['ar100w2v']#"/home/zeio/viot/models/w2v/test.txt"
DEFAULT_ARRAY_FILE_NAME = "embeddings"

#
# make word embeddings
#

# normalize annotated commands
df = normalize.handle_files(utils.list_dataset_files(config['paths']['datasets']['annotations']))

types = df.type.unique()
types_dict = dict(zip(types, range(len(types))))
df = df.replace({'type': types_dict})
print(f"Replacing utterance types according to mapping: {types_dict}")

# prepare texts for training
stopwords = stopwords.words('russian')
pattern = re.compile("^[а-яА-ЯёЁ]+")
train_utterances = list(map(lambda text: [word for word in text.split(" ") if word not in stopwords and pattern.match(word)], df.text))

words = list(itertools.chain(*train_utterances))

most_frequent_words = [word for word in sorted(set(words), key = lambda ele: words.count(ele), reverse = True)[:100] if word != 'холодно']
print(len(most_frequent_words))
print(most_frequent_words)

model = KeyedVectors.load_word2vec_format(DEFAULT_MODEL_PATH, binary=False)

array_to_save = np.array([i for i in list(map(lambda word: model[word], most_frequent_words)) if type(i) != np.float64])
np.save(os.path.join(config['paths']['models']['w2v'], DEFAULT_ARRAY_FILE_NAME), array_to_save)

print(f"Resulting shape: {array_to_save.shape}")

tsne = TSNE(n_components=2)
tsne_coordinates = tsne.fit_transform(array_to_save)
plt.figure(figsize=(17,12))
plt.scatter(tsne_coordinates[:, 0], tsne_coordinates[:, 1])
for i, txt in enumerate(most_frequent_words):
    plt.annotate(txt, (tsne_coordinates[i, 0], tsne_coordinates[i, 1]))
#plt.title("50 most frequent words")
#plt.show()
plt.savefig('images/tsne.png')

# Join uterances and write to an external file
#joined_utterances = [' '.join(utterance) for utterance in train_utterances if len(utterance) >= 1]
#fo.write_lines(config['paths']['datasets']['w2v']['train_utterances'], joined_utterances)

# train word2ver
# path = get_tmpfile("word2vec.model")
# model = Word2Vec(train_utterances, size=10, window=5, min_count=1, workers=4)
# model.save(os.path.join(config['paths']['models']['w2v'], "test.model"))
# model.wv.save_word2vec_format(os.path.join(config['paths']['models']['w2v'], "test.txt"))
