import os, re, sys
import json
import itertools

from .. import field_extractors as fe, converters, utils, file_operators as fo

config = utils.load_config(utils.parse_args().config)

def get_flawy(input_files, get_file_names=False):
	flawy = {}
	for j in range(len(input_files)):
		input_file = input_files[j]
		with open(input_file) as f:
			annotation = json.loads(f.read())
		if not fe.extract_ontology_label(annotation) or not fe.extract_utterance_type(annotation):
			print(f"File {input_file} is flawy")
			flawy[input_file] = annotation
			continue
	return flawy

if __name__ == "__main__":
	fo.write_json(config['paths']['datasets']['flawy-denormalized'], handle_files(utils.list_dataset_files(config['paths']['datasets']['annotations'])))
