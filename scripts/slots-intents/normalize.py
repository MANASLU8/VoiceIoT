import os, re, sys
import json

from .. import field_extractors as fe, converters, utils, file_operators as fo

config = utils.load_config(utils.parse_args().config)

def join_labels(labels, out_label = 'O'):
	res = []
	current_label = ''
	for label in labels:
		if label == out_label:
			res.append(out_label)
		else:
			if current_label != label:
				res.append(f"B-{label}")
			else:
				res.append(f"I-{label}")
			current_label = label
	return res

def make_chunk(i, entities, entity_names, word, ontology_label):
	for entity_name in entities.keys():
		if i in entities[entity_name]:
			entity_names.add(entity_name.strip())
			return f"{ontology_label.strip()}.{entity_name.strip()}"
	return 'O'

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
		sent = "BOS " + re.sub(r'[^\w\s]','',text).lower() + " EOS O"
		labels = []
		if 'slots-indices' not in annotation or len(annotation['slots-indices']) < 1 or not ontology_label:
			continue
		for (i, word) in enumerate(re.sub(r'[^\w\s]','',text).lower().split(' ')):
			appended = False
			if 'slots-indices' in annotation:
				labels.append(make_chunk(i, annotation['slots-indices'][0], entity_names, word, ontology_label))
			elif 'slots-indices-bio' in annotation:
				labels.append(make_chunk(i, converters.decode_bio(annotation['slots-indices-bio']), entity_names, word, ontology_label))
		sent += "\t"
		sent += ' '.join(join_labels(labels))
		sent += ' '
		sent += ontology_label
		result.append(sent)
		continue
		res = join_empty(res)
		if ontology_label in dicti:
			dicti[ontology_label]["utterances"].append({"data": res})
		else:
			dicti[ontology_label] = {"utterances": [{"data": res}]}
	return result

fo.write_lines(config['paths']['datasets']['slots-intents']['data'], handle_files(utils.list_dataset_files(config['paths']['datasets']['annotations'])))