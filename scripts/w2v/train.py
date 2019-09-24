import os, re, sys
import json
import pandas as pd
from nltk.corpus import stopwords
import re
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec

from .. import utils
from . import normalize

config = utils.load_config(utils.parse_args().config)

# normalize annotated commands
df = normalize.handle_files(utils.list_dataset_files(config['paths']['datasets']['annotations']))#.groupby('type')['text'].nunique()

# prepare texts for training
stopwords = stopwords.words('russian')
pattern = re.compile("^[а-яА-ЯёЁ]+")
train_utterances = list(map(lambda text: [word for word in text.split(" ") if word not in stopwords and pattern.match(word)], df.text))

# train
path = get_tmpfile("word2vec.model")
model = Word2Vec(train_utterances, size=10, window=5, min_count=1, workers=4)
model.save(os.path.join(config['paths']['models']['w2v'], "test.model"))

# check result
def check(word):
	print(f"vector for word '{word}': {model.wv[word]}")

check("сколько")

# map given results with input data
df.text = list(map(lambda text: [[value for value in model[word]] for word in text.split(" ") if word in model], df.text))
print(df.text)

# save transormed df
df.to_csv(os.path.join(config['paths']['datasets']['w2v']['root'], "test.csv"))
