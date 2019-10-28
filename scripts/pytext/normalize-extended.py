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
	return ",".join([f"{pair[0]}:{pair[1]}:{label.strip()}" for label in indices for pair in indices[label]]) #{ontology_label.strip()}

def intersection(lst1, lst2): 
    return list(set(lst1) & set(lst2)) 

forbidden_words = ['если', 'ежели', 'когда', 'и', 'вдруг', 'или']

def handle_files(input_files):
	samples = []
	counter = 0
	entity_names = set({})
	for input_file in input_files:
		with open(input_file) as f:
			annotation = json.loads(f.read())
		ontology_label = fe.extract_ontology_label(annotation)
		text = fe.extract_text(annotation)
		simplified_text = re.sub(r'[^\w\s]','',text).lower()
		splitted_text = simplified_text.split(' ')
		if 'slots-indices' not in annotation or len(annotation['slots-indices']) < 1 or not ontology_label or not fe.extract_utterance_type(annotation):\
			#or fe.extract_utterance_type(annotation) != 'Command' or len(intersection(forbidden_words, splitted_text)) > 0:
			continue
		counter += 1
		if 'slots-indices' in annotation:
			slots_char_indices = stringify_indices(make_indices(annotation['slots-indices'][0], splitted_text), ontology_label)
		elif 'slots-indices-bio' in annotation:
			slots_char_indices = stringify_indices(make_indices(converters.decode_bio(annotation['slots-indices-bio']), splitted_text), ontology_label)
		samples.append(f"{ontology_label}\t{slots_char_indices}\t{simplified_text}\t1\t1")
	print(f'Extracted {counter} samples')
	return samples

fo.write_lines(config['paths']['datasets']['pytext']['data-extended'], handle_files(utils.list_dataset_files(config['paths']['datasets']['annotations'])))
