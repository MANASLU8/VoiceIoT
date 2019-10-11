import os, re, sys
import json
import itertools
import argparse

from .. import field_extractors as fe, converters, utils, file_operators as fo

parser = argparse.ArgumentParser()
parser.add_argument('--nice', dest='nice', action='store_true', help='denormalize labelled commands as well as unlabelled')
utils.add_config_arg(parser)
args = parser.parse_args()
config = utils.load_config(args.config)

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

def get_two(input_files, get_file_names=False):
	flawy = {}
	nice = {}
	for j in range(len(input_files)):
		input_file = input_files[j]
		with open(input_file) as f:
			annotation = json.loads(f.read())
		if not 'slots-indices' in annotation:
			print(f"File {input_file} is flawy")
			flawy[input_file] = annotation
		else:
			nice[input_file] = annotation
	return flawy, nice

if __name__ == "__main__":
	if args.nice:
		flawy, nice = get_two(utils.list_dataset_files(config['paths']['datasets']['annotations']))
		fo.write_json(config['paths']['datasets']['flawy-denormalized'], flawy)
		fo.write_json(config['paths']['datasets']['nice-denormalized'], nice)
	else:
		flawy = get_flawy(utils.list_dataset_files(config['paths']['datasets']['annotations']))
		fo.write_json(config['paths']['datasets']['flawy-denormalized'], flawy)
