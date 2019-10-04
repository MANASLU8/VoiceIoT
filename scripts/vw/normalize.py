import os, re, sys
import json
import numpy as np
from .. import w2v
from gensim.models import KeyedVectors
from .. import field_extractors as fe, converters, utils, file_operators as fo

config = utils.load_config(utils.parse_args().config)

DEFAULT_MODEL_PATH = config['paths']['models']['ar100w2v']#"/home/zeio/viot/models/w2v/test.txt"

def handle_files(input_files):
	samples = []
	texts = set({})

	df = w2v.normalize.handle_files(utils.list_dataset_files(config['paths']['datasets']['annotations']))
	model = KeyedVectors.load_word2vec_format(DEFAULT_MODEL_PATH, binary=False)
	df['text_encoded'] = list(map(lambda text: np.average([[value for value in model[word]] for word in text.split(" ") if word in model], axis=0), df.text))
	df = df[df['text_encoded'].notnull()]

	types = df.type.unique()
	type_dict = dict(zip(types, range(1, len(types) + 1)))
	df['type_encoded'] = df.replace({"type": type_dict}).type

	print(f"Writing data: {df}")
	
	return df.apply(lambda row: ' | '.join([' '.join([str(row[3]), f"'{row[1].replace(' ','_')}/{row[0]}"]), ' '.join([f'{i}:{v}' for i,v in enumerate(row[2])])]), axis = 1)

fo.write_lines(config['paths']['datasets']['vw']['data'], handle_files(utils.list_dataset_files(config['paths']['datasets']['annotations'])))
