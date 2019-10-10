import json, random, sys

from .. import file_operators as fo, utils

config = utils.load_config(utils.parse_args().config)

data = fo.read_json(config['paths']['datasets']['snips-nlu']['data'])

test_samples = {}

print("Samples per label:")
for label in data['intents'].keys():
	print(f"{label:30s}: {len(data['intents'][label]['utterances'])}")
	quantity_for_test = len(data['intents'][label]['utterances']) * config['test-percentage'] / float(100)
	if quantity_for_test < 1:
		continue
	test_samples[label] = []
	counter = 0
	while counter < quantity_for_test:
		choice = random.choice(data['intents'][label]['utterances'])
		test_samples[label].append(choice)
		data['intents'][label]['utterances'].remove(choice)
		counter += 1

fo.write_json(config['paths']['datasets']['snips-nlu']['train'], data)
fo.write_json(config['paths']['datasets']['snips-nlu']['test'], test_samples)
