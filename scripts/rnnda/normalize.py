import os, re, sys
import json

from .. import field_extractors as fe, converters, utils, file_operators as fo

config = utils.load_config(utils.parse_args().config)

VERBOSE = True

def handle_files(input_files):
	result = []
	entity_names = set({})
	for j in range(len(input_files)):
		input_file = input_files[j]
		if VERBOSE:
			print(f'handling file {input_file}')
		with open(input_file) as f:
			annotation = json.loads(f.read())
		utterance_type = fe.extract_utterance_type(annotation)
		text = fe.extract_text(annotation)
		normalized_text = re.sub(r'[^\w\s]','',text).lower()
		if not utterance_type:
			continue
		text_with_labels = f"{normalized_text}|{utterance_type}"
		if VERBOSE:
			print(text_with_labels)
		result.append(text_with_labels)
	if VERBOSE:
		print(f"Collected {len(result)} samples")
	return result

#print(handle_files(utils.list_dataset_files(config['paths']['datasets']['annotations'])))
fo.write_lines(config['paths']['datasets']['rnnda']['data'], handle_files(utils.list_dataset_files(config['paths']['datasets']['annotations'])))