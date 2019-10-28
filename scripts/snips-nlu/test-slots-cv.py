from snips_nlu import SnipsNLUEngine
import json
from .. import metrics
from .. import file_operators as fo, utils
import numpy as np

FOLDS = 10

config = utils.load_config(utils.parse_args().config)

def flatten(data):
	for label in data['intents'].keys():
		for choice in data['intents'][label]['utterances']:
			yield {'utterance': choice, 'label': label}

def make_sets(chunks, data):
	for i in range(len(chunks)):
		chunk = chunks[i]
		test_set = {}
		train_set = {"intents": {}, 'entities': data['entities'], 'language': data['language']}
		for sample in chunk:
			label = sample['label']
			if label not in test_set:
				test_set[label] = []
			test_set[label].append(sample['utterance'])

		antichunk = [sample for j in range(len(chunks)) for sample in chunks[j] if i != j]
		for sample in antichunk:
			label = sample['label']
			if label not in train_set['intents']:
				train_set['intents'][label] = {}
				train_set['intents'][label]['utterances'] = []
			train_set['intents'][label]['utterances'].append(sample['utterance'])

		yield {'train': train_set, 'test': test_set}

# load dataset

data = fo.read_json(config['paths']['datasets']['snips-nlu']['data'])
#test_dataset = fo.read_json(config['paths']['datasets']['snips-nlu']['test'])

# split
# print(train_dataset)
utterances = np.array(list(flatten(data)))
np.random.shuffle(utterances)
chunks = np.array_split(utterances, FOLDS)
# print(chunks[0].shape)

sets = list(make_sets(chunks, data))

#print(len(utterances))

accuracies = []
recalls = []
output_commands = []
for i in range(len(sets)):
	train_dataset = sets[i]['train']
	test_dataset = sets[i]['test']

	print(f"Fitting model #{i}...")
	engine = SnipsNLUEngine()
	engine.fit(train_dataset)

	total_recall = []

	counter = 0
	positive_counter = 0

	print(f"Calculating metrics #{i}...")
	print(f"{'Sample':80s}\t{'recognized-label':20s}\t{'true-label':20s}\t{'correctly-recognized':30s}")
	for label in test_dataset.keys():
		for sample in test_dataset[label]:
			true_slots = list(filter(lambda entity: entity, [item.get('entity', None) for item in sample['data']]))
			command = ' '.join([item.get('text', '') for item in sample['data']])
			splitted_command = command.split(' ')
			parser_response = engine.parse(command)
			recognized_slots = [slot['entity'] for slot in parser_response['slots']]
			parsed_command = list((chunk['rawValue'], chunk['entity']) for chunk in parser_response['slots'])
			for command_word in splitted_command:
				slotted = False
				for item in parsed_command:
					if item[0] == command_word:
						slotted = True
				if not slotted:
					parsed_command.append((command_word, '-'))
			output_commands.append({'command': parsed_command, 'device': label})
			recognized_label = parser_response['intent']['intentName']
			if not recognized_label:
				recognized_label = '-'
			recall = metrics.get_recall(true_slots, recognized_slots)
			total_recall.append(recall)
			#print(parsed_command)
			print(f"{command:80s}\t{recognized_label:20s}\t{label:20s}\t{recognized_label==label}")
			if recognized_label == label:
				positive_counter += 1
			counter += 1
	accuracy = round(positive_counter / float(counter) * 100, 2)
	accuracies.append(accuracy)
	recall = round(np.mean(total_recall), 4)
	recalls.append(recall)
	print(f"Correctly recognized {positive_counter} of {counter} ({accuracy} %)")
	print(f"Average slot recall is {recall}")

fo.write_json("labelled-commands.json", output_commands)
print(f"Mean accuracy: {np.mean(accuracies)}")
print(f"Mean recall: {np.mean(recalls)}")
