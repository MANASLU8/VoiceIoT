import os, re, sys
import json
import itertools

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

def write_dataset(folder, items, vocab_folder = None):
	fo.write_lines(os.path.join(folder, config['paths']['datasets']['joint-nlu']['filenames']['data']['texts']), [item[0] for item in items])
	fo.write_lines(os.path.join(folder, config['paths']['datasets']['joint-nlu']['filenames']['data']['slots']), [item[1] for item in items])
	fo.write_lines(os.path.join(folder, config['paths']['datasets']['joint-nlu']['filenames']['data']['labels']), [item[2] for item in items])
	if vocab_folder:
		fo.write_lines(os.path.join(vocab_folder, config['paths']['datasets']['joint-nlu']['filenames']['vocabulary']['slots']), set(itertools.chain(*[item[1].split(' ') for item in items])))
		fo.write_lines(os.path.join(vocab_folder, config['paths']['datasets']['joint-nlu']['filenames']['vocabulary']['texts']), set(itertools.chain(*[item[0].split(' ') for item in items])))
		fo.write_lines(os.path.join(vocab_folder, config['paths']['datasets']['joint-nlu']['filenames']['vocabulary']['labels']), set([item[2] for item in items]))

def handle_files(input_files):
	result = []
	entity_names = set({})
	for j in range(len(input_files)):
		input_file = input_files[j]
		with open(input_file) as f:
			annotation = json.loads(f.read())
		ontology_label = fe.extract_ontology_label(annotation)
		text = fe.extract_text(annotation)
		sent = re.sub(r'[^\w\s]','',text).lower()
		labels = []
		for (i, word) in enumerate(re.sub(r'[^\w\s]','',text).lower().split(' ')):
			appended = False
			if 'slots-indices' in annotation:
				labels.append(make_chunk(i, annotation['slots-indices'][0], entity_names, word, ontology_label))
			elif 'slots-indices-bio' in annotation:
				labels.append(make_chunk(i, converters.decode_bio(annotation['slots-indices-bio']), entity_names, word, ontology_label))
		result.append([sent, ' '.join(join_labels(labels)), ontology_label])
	return result

if __name__ == "__main__":
	write_dataset(config['paths']['datasets']['joint-nlu']['data'],
		handle_files(utils.list_dataset_files(config['paths']['datasets']['annotations'])), config['paths']['datasets']['joint-nlu']['vocabulary'])
