import os, re, sys
import json
import numpy as np

from .. import field_extractors as fe, converters, utils, file_operators as fo

config = utils.load_config(utils.parse_args().config)

def make_indices(annotations, text):
	word_lengths = [len(w) for w in text]
	char_indices = [[int(np.sum(tuple(word_lengths[:i])) + i), int(np.sum(tuple(word_lengths[:i + 1])) - 1 + i)] for i in range(len(text))]
	return {label: [char_indices[i if i < len(text) else len(text) - 1] for i in annotations[label]] for label in annotations}

def stringify_indices(indices, ontology_label):
	return ",".join([f"{pair[0]}:{pair[1]}:{ontology_label.strip()}.{label.strip()}" for label in indices for pair in indices[label]])

def handle_files(input_files):
	samples = []
	utterance_types = set({})
	for input_file in input_files:
		with open(input_file) as f:
			annotation = json.loads(f.read())
		utterance_type = fe.extract_utterance_type(annotation)
		if not utterance_type:
			continue
		utterance_types.add(utterance_type)
		text = re.sub(r'[^\w\s]','',fe.extract_text(annotation)).lower().split(' ')
		samples.append(f"text:{' '.join(text)}\tlabels:{utterance_type}\tepisode_done:True")
	return list(map(lambda sample: sample + "\tlabel_candidates:" + '|'.join(utterance_types), samples))

fo.write_lines(config['paths']['datasets']['parlai']['data'], handle_files(utils.list_dataset_files(config['paths']['datasets']['annotations'])))
