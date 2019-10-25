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
		if not fe.extract_ontology_label(annotation) or not fe.extract_utterance_type(annotation) or not 'slots-indices' in annotation:
			print(f"File {input_file} is flawy")
			flawy[input_file] = annotation
			continue
	return flawy

forbidden_words = ['если', 'ежели', 'когда', 'и', 'вдруг', 'или']

def intersection(lst1, lst2): 
    return list(set(lst1) & set(lst2))

ontology_devices = ["AirConditioning", "AlarmClock", "AudioSystem", "CoffeeMachine", "CurtainsRelay", "DoorBell", "FloorLite", "HeatingFloor", "Printer", "SmartFridge", "SmartMicrowave", "WindowRelay", "Lamp", "SmartBreadMaker", "WashingMachine"]

def get_two(input_files, get_file_names=False):
	flawy = {}
	nice = {}
	counter = 0
	texts_nice = set()
	for j in range(len(input_files)):
		input_file = input_files[j]
		with open(input_file) as f:
			annotation = json.loads(f.read())
		text = fe.extract_text(annotation)
		simplified_text = re.sub(r'[^\w\s]','',text).lower()
		splitted_text = simplified_text.split(' ')
		if 'slots-indices' not in annotation or len(annotation['slots-indices']) < 1 or not fe.extract_ontology_label(annotation) or not fe.extract_utterance_type(annotation) or fe.extract_utterance_type(annotation) != 'Command' or\
			len(intersection(forbidden_words, splitted_text)) > 0 or fe.extract_ontology_label(annotation) not in ontology_devices:
			continue
		if not 'slots-indices' in annotation:
			print(f"File {input_file} is flawy")
			flawy[input_file] = annotation
		else:
			if simplified_text in texts_nice:
				continue
			counter += 1
			nice[input_file] = annotation
			texts_nice.add(simplified_text)
	print(f'Extracted {counter} nice samples')
	return flawy, nice

def get_complex(input_files):
	flawy = {}
	nice = {}
	counter = 0
	texts_nice = set()
	for j in range(len(input_files)):
		input_file = input_files[j]
		with open(input_file) as f:
			annotation = json.loads(f.read())
		text = fe.extract_text(annotation)
		simplified_text = re.sub(r'[^\w\s]','',text).lower()
		splitted_text = simplified_text.split(' ')
		if 'slots-indices' not in annotation or len(annotation['slots-indices']) < 1 or not fe.extract_ontology_label(annotation) or not fe.extract_utterance_type(annotation) or fe.extract_utterance_type(annotation) != 'Command' or\
			len(intersection(forbidden_words, splitted_text)) == 0 or fe.extract_ontology_label(annotation) not in ontology_devices:
			continue
		if not 'slots-indices' in annotation:
			print(f"File {input_file} is flawy")
			flawy[input_file] = annotation
		else:
			if simplified_text in texts_nice:
				continue
			counter += 1
			nice[input_file] = annotation
			texts_nice.add(simplified_text)
	print(f'Extracted {counter} nice samples')
	return flawy, nice

if __name__ == "__main__":
	# get complex commands (with conditions, etc.)
	# flawy, nice = get_complex(utils.list_dataset_files(config['paths']['datasets']['annotations']))
	# fo.write_json("complex.json", nice)
	
	if args.nice:
		flawy, nice = get_two(utils.list_dataset_files(config['paths']['datasets']['annotations']))
		fo.write_json(config['paths']['datasets']['flawy-denormalized'], flawy)
		fo.write_json(config['paths']['datasets']['nice-denormalized'], nice)
	else:
		flawy = get_flawy(utils.list_dataset_files(config['paths']['datasets']['annotations']))
		fo.write_json(config['paths']['datasets']['flawy-denormalized'], flawy)
