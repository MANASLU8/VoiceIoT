import os, re, sys
import json

from . import field_extractors as fe, converters, utils, file_operators as fo

config = utils.load_config(utils.parse_args().config)

labels_to_filter = ["WashingMachine", "Dishwasher", "Lamp", "AlarmClock", "CurtainsRelay", "AudioSystem", "Printer", "WindowRelay", "CoffeeMachine", "Humidifier", "SmartFridge", "HeatingFloor", "SmartBreadMaker", "SmartMicrowave"]

def handle_files(input_files):
	result = []
	entity_names = set({})
	for j in range(len(input_files)):
		input_file = input_files[j]
		print(f'handling file {input_file}')
		with open(input_file) as f:
			annotation = json.loads(f.read())
		ontology_label = fe.extract_ontology_label(annotation)
		text = fe.extract_text(annotation)
		if ontology_label in labels_to_filter:
			result.append(f"{ontology_label}\t{text}")
	return result

print("\n".join(handle_files(utils.list_dataset_files(config['paths']['datasets']['annotations']))))