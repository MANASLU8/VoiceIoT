import os, re, sys
import json

from .. import field_extractors as fe, converters, utils, file_operators as fo

config = utils.load_config(utils.parse_args().config)

def join(collection, start_i, end_i):
	return {"text": ' '.join([item['text'] for item in collection[start_i: end_i]])}

def join_empty(collection):
	empty_seq_start_i = -1
	res = []
	for i in range(len(collection)):
		if 'slot_name' in collection[i] and empty_seq_start_i >= 0 and i - empty_seq_start_i > 1:
			res.append(join(collection, empty_seq_start_i, i))
			empty_seq_start_i = -1
			res.append(collection[i])
		elif 'slot_name' not in collection[i] and empty_seq_start_i < 0:
			empty_seq_start_i = i
		elif 'slot_name' in collection[i]:
			if empty_seq_start_i >= 0:
				res.append(collection[empty_seq_start_i])
				empty_seq_start_i = -1
			res.append(collection[i])
	if empty_seq_start_i >= 0:
		res.append(join(collection, empty_seq_start_i, len(collection)))
	return res

def make_chunk(i, entities, entity_names, word, ontology_label):
	if entities and ontology_label:
		for entity_name in entities.keys():
			if i in entities[entity_name]:
				entity_names.add(entity_name.strip())
				return {'text': word, 'entity': entity_name.strip(), 'slot_name': f"{ontology_label.strip()}.{entity_name.strip()}"}
	return {'text': word}

def handle_files(input_files):
	dicti = {}
	entity_names = set({})
	for j in range(len(input_files)):
		input_file = input_files[j]
		print(f'handling file {input_file}')
		with open(input_file) as f:
			annotation = json.loads(f.read())
		ontology_label = fe.extract_ontology_label(annotation)
		text = fe.extract_text(annotation)
		res = []
		if 'slots-indices' not in annotation or len(annotation['slots-indices']) < 1:
			continue
		for (i, word) in enumerate(re.sub(r'[^\w\s]','',text).lower().split(' ')):
			appended = False
			if 'slots-indices' in annotation:
				res.append(make_chunk(i, annotation['slots-indices'][0] if len(annotation['slots-indices']) >= 1 else None, entity_names, word, ontology_label))
			elif 'slots-indices-bio' in annotation:
				res.append(make_chunk(i, converters.decode_bio(annotation['slots-indices-bio']), entity_names, word, ontology_label))
		res = join_empty(res)
		if ontology_label in dicti:
			dicti[ontology_label]["utterances"].append({"data": res})
		else:
			dicti[ontology_label] = {"utterances": [{"data": res}]}
	return dicti, entity_names

intents, entities = handle_files(utils.list_dataset_files(config['paths']['datasets']['annotations']))
fo.write_json(config['paths']['datasets']['snips-nlu']['data'], 
	{"intents": intents, "entities": {entity: {"data": [], "use_synonyms": True, "automatically_extensible": True, "matching_strictness": 1.0} for entity in entities}, "language": "en"})
