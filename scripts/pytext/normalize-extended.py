import os, re, sys
import json
import numpy as np

sys.path.append("../")
from field_extractors import extract_text, extract_ontology_label
from converters import decode_bio
from utils import list_dataset_files
from file_operators import write_lines

INPUT_FOLDER = "../../dataset/annotations"
OUTPUT_FILE = "../../dataset/pytext/data-extended.tsv"

def make_indices(annotations, text):
	word_lengths = [len(w) for w in text]
	char_indices = [[int(np.sum(tuple(word_lengths[:i])) + i), int(np.sum(tuple(word_lengths[:i + 1])) - 1 + i)] for i in range(len(text))]
	return {label: [char_indices[i if i < len(text) else len(text) - 1] for i in annotations[label]] for label in annotations}

def stringify_indices(indices, ontology_label):
	return ",".join([f"{pair[0]}:{pair[1]}:{label.strip()}" for label in indices for pair in indices[label]])

def handle_files(input_files):
	samples = []
	entity_names = set({})
	for input_file in input_files:
		print(f'handling file {input_file}')
		with open(input_file) as f:
			annotation = json.loads(f.read())
		ontology_label = extract_ontology_label(annotation)
		text = extract_text(annotation)
		simplified_text = re.sub(r'[^\w\s]','',text).lower()
		splitted_text = simplified_text.split(' ')
		if 'slots-indices' in annotation:
			slots_char_indices = stringify_indices(make_indices(annotation['slots-indices'][0], splitted_text), ontology_label)
		elif 'slots-indices-bio' in annotation:
			slots_char_indices = stringify_indices(make_indices(decode_bio(annotation['slots-indices-bio']), splitted_text), ontology_label)
		samples.append(f"{ontology_label}\t{slots_char_indices}\t{simplified_text}\t1\t1")
	return samples

write_lines(OUTPUT_FILE, handle_files(list_dataset_files(INPUT_FOLDER)))
