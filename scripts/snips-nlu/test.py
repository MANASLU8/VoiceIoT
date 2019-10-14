from snips_nlu import SnipsNLUEngine
import json

from .. import file_operators as fo, utils

config = utils.load_config(utils.parse_args().config)

train_dataset = fo.read_json(config['paths']['datasets']['snips-nlu']['train'])
test_dataset = fo.read_json(config['paths']['datasets']['snips-nlu']['test'])

engine = SnipsNLUEngine()
engine.fit(train_dataset)

counter = 0
positive_counter = 0

print(f"{'Sample':80s}\t{'recognized-label':20s}\t{'true-label':20s}\t{'correctly-recognized':30s}")
for label in test_dataset.keys():
	for sample in test_dataset[label]:
		recognized_label = engine.parse(sample)['intent']['intentName']
		if not recognized_label:
			recognized_label = '-'
		print(f"{sample:80s}\t{recognized_label:20s}\t{label:20s}\t{recognized_label==label}")
		if recognized_label == label:
			positive_counter += 1
		counter += 1

print(f"Correctly recognized {positive_counter} of {counter} ({round(positive_counter / float(counter) * 100, 2)} %)")
