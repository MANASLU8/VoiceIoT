from snips_nlu import SnipsNLUEngine
import json
from .. import metrics
from .. import file_operators as fo, utils
import numpy as np

config = utils.load_config(utils.parse_args().config)

train_dataset = fo.read_json(config['paths']['datasets']['snips-nlu']['train'])
test_dataset = fo.read_json(config['paths']['datasets']['snips-nlu']['test'])

print("Fitting model...")
engine = SnipsNLUEngine()
engine.fit(train_dataset)

total_recall = []

counter = 0
positive_counter = 0

print("Calculating metrics...")
print(f"{'Sample':80s}\t{'recognized-label':20s}\t{'true-label':20s}\t{'correctly-recognized':30s}")
for label in test_dataset.keys():
	for sample in test_dataset[label]:
		true_slots = list(filter(lambda entity: entity, [item.get('entity', None) for item in sample['data']]))
		command = ' '.join([item.get('text', '') for item in sample['data']])
		parser_response = engine.parse(command)
		recognized_slots = [slot['entity'] for slot in parser_response['slots']]
		recognized_label = parser_response['intent']['intentName']
		if not recognized_label:
			recognized_label = '-'
		recall = metrics.get_recall(true_slots, recognized_slots)
		total_recall.append(recall)
		print(f"{command:80s}\t{recognized_label:20s}\t{label:20s}\t{recognized_label==label}")
		if recognized_label == label:
			positive_counter += 1
		counter += 1

print(f"Correctly recognized {positive_counter} of {counter} ({round(positive_counter / float(counter) * 100, 2)} %)")
print(f"Average slot recall is {round(np.mean(total_recall), 4)}")
