import os, re
import json

INPUT_FOLDER = "../../dataset/annotations"
OUTPUT_FILE = "../../dataset/snips_nlu/train-all.json"

def list_dataset_files(dataset_path):
	annotators_folders = [os.path.join(dataset_path, annotator_folder) for annotator_folder in os.listdir(dataset_path)]
	return [os.path.join(folder, file) for folder in annotators_folders for file in os.listdir(folder)]

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

def extract_ontology_label(annotation):
	if type(annotation['ontology-label']) == list:
		return annotation['ontology-label'][0]
	else:
		return annotation['ontology-label']

def decode_bio(bio):
	dicti = {}
	for pair in bio:
		entity_name = pair[1].split("-", maxsplit = 1)[1]
		if entity_name in dicti:
			dicti[entity_name].append(pair[0])
		else:
			dicti[entity_name] = [pair[0]]
	return dicti

def make_chunk(i, entities, entity_names, word, ontology_label):
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
		ontology_label = extract_ontology_label(annotation)
		res = []
		for (i, word) in enumerate(re.sub(r'[^\w\s]','',annotation['text'][0]).lower().split(' ')):
			appended = False
			if 'slots-indices' in annotation:
				res.append(make_chunk(i, annotation['slots-indices'][0], entity_names, word, ontology_label))
			elif 'slots-indices-bio' in annotation:
				res.append(make_chunk(i, decode_bio(annotation['slots-indices-bio']), entity_names, word, ontology_label))
		res = join_empty(res)
		if ontology_label in dicti:
			dicti[ontology_label]["utterances"].append({"data": res})
		else:
			dicti[ontology_label] = {"utterances": [{"data": res}]}
	return dicti, entity_names

with open(OUTPUT_FILE, "w") as f:
	intents, entities = handle_files(list_dataset_files(INPUT_FOLDER))
	sample_entity = {"data": [], "use_synonyms": True, "automatically_extensible": True, "matching_strictness": 1.0}
	f.write(json.dumps({"intents": intents, "entities": {entity: sample_entity for entity in entities}, "language": "en"}, indent=2).encode().decode('unicode-escape'))
