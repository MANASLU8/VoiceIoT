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

from .. import utils
from . import normalize

config = utils.load_config(utils.parse_args().config)

#
# make word embeddings
#

# normalize annotated commands
df = normalize.handle_files(utils.list_dataset_files(config['paths']['datasets']['annotations']))#.groupby('type')['text'].nunique()

types = df.type.unique()
types_dict = dict(zip(types, range(len(types))))
df = df.replace({'type': types_dict})
print(f"Replacing utterance types according to mapping: {types_dict}")

# prepare texts for training
stopwords = stopwords.words('russian')
pattern = re.compile("^[а-яА-ЯёЁ]+")
train_utterances = list(map(lambda text: [word for word in text.split(" ") if word not in stopwords and pattern.match(word)], df.text))

# train word2ver
path = get_tmpfile("word2vec.model")
model = Word2Vec(train_utterances, size=10, window=5, min_count=1, workers=4)
model.save(os.path.join(config['paths']['models']['w2v'], "test.model"))

# check result
def check(word):
	print(f"vector for word '{word}': {model.wv[word]}")

check("сколько")

# map given results with input data
df.text = list(map(lambda text: np.average([[value for value in model[word]] for word in text.split(" ") if word in model], axis=0), df.text))
#print(df.text)

# save transormed df
df.to_csv(os.path.join(config['paths']['datasets']['w2v']['root'], "test.csv"))

#
# classify using linear regression classifier
#

#print(df[:5])

def split(df, pieces=2):
	number_of_lines = df.shape[0]
	chunk_size = number_of_lines // pieces
	return [{"test": df[i * chunk_size: (i + 1) * chunk_size], "train": pd.concat([df[: i * chunk_size], df[(i + 1) * chunk_size:]])} for i in range(pieces)]

def classify(df, classifier, pieces=5):
	results = []

	for item in split(df, pieces=pieces):
		x_train =  np.array([[value for value in item] for item in item['train'].text.to_numpy()])
		x_test = np.array([[value for value in item] for item in item['test'].text.to_numpy()])
		y_train = item['train'].type.to_numpy()
		y_test = item['test'].type.to_numpy()

		print(f"X train = {x_train}")
		print(f"Y train = {y_train}")

		classifier.fit(x_train, y_train)
		y_pred = classifier.predict(x_test)

		results.append({"accuracy": accuracy_score(y_test, y_pred), "f1-score": f1_score(y_test, y_pred, average='weighted')})

	print(f"Average accuracy: {np.average([item['accuracy'] for item in results])}")
	print(f"Average f1-score: {np.average([item['f1-score'] for item in results])}")

classify(df, LogisticRegression(n_jobs=1, C=1e9))


# y_train, X_train = vec_for_learning(model_dbow, train_tagged)
# y_test, X_test = vec_for_learning(model_dbow, test_tagged)
# logreg = LogisticRegression(n_jobs=1, C=1e5)
# logreg.fit(X_train, y_train)
# y_pred = logreg.predict(X_test)
# from sklearn.metrics import accuracy_score, f1_score
# print('Testing accuracy %s' % accuracy_score(y_test, y_pred))
# print('Testing F1 score: {}'.format(f1_score(y_test, y_pred, average='weighted')))