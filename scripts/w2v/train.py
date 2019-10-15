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

from .. import utils, file_operators as fo
from . import normalize

config = utils.load_config(utils.parse_args().config)

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
train_utterances = list(map(lambda text: [word for word in text.split(" ")], df.text))# if word not in stopwords and pattern.match(word)], df.text))

# Join uterances and write to an external file
joined_utterances = [' '.join(utterance) for utterance in train_utterances if len(utterance) >= 1]
fo.write_lines(config['paths']['datasets']['w2v']['train_utterances'], joined_utterances)

# train word2ver
# path = get_tmpfile("word2vec.model")
# model = Word2Vec(train_utterances, size=10, window=5, min_count=1, workers=4)
# model.save(os.path.join(config['paths']['models']['w2v'], "test.model"))
# model.wv.save_word2vec_format(os.path.join(config['paths']['models']['w2v'], "test.txt"))
