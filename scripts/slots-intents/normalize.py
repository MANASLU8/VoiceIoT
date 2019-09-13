import os, re, sys
import json

sys.path.append("../")
from field_extractors import extract_text, extract_ontology_label
from converters import decode_bio
from utils import list_dataset_files
from file_operators import write_lines

INPUT_FOLDER = "../../dataset/annotations"
OUTPUT_FILE = "../../dataset/slots-intents/train.csv"

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
		ontology_label = extract_ontology_label(annotation)
		text = extract_text(annotation)
		sent = "BOS " + re.sub(r'[^\w\s]','',text).lower() + " EOS O"
		
		labels = []
		for (i, word) in enumerate(re.sub(r'[^\w\s]','',text).lower().split(' ')):
			appended = False
			if 'slots-indices' in annotation:
				labels.append(make_chunk(i, annotation['slots-indices'][0], entity_names, word, ontology_label))
			elif 'slots-indices-bio' in annotation:
				labels.append(make_chunk(i, decode_bio(annotation['slots-indices-bio']), entity_names, word, ontology_label))
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

write_lines(OUTPUT_FILE, handle_files(list_dataset_files(INPUT_FOLDER)))