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

print("Calculating recall...")
for label in test_dataset.keys():
	for sample in test_dataset[label]:
		true_slots = list(filter(lambda entity: entity, [item.get('entity', None) for item in sample['data']]))
		command = ' '.join([item.get('text', '') for item in sample['data']])
		recognized_slots = [slot['entity'] for slot in engine.parse(command)['slots']]
		recall = metrics.get_recall(true_slots, recognized_slots)
		total_recall.append(recall)

print(f"Average recall is {round(np.mean(total_recall), 4)}")
