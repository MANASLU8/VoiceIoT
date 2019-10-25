import json, random, sys

from .. import file_operators as fo, utils

config = utils.load_config(utils.parse_args().config)

data = fo.read_json(config['paths']['datasets']['snips-nlu']['data'])

test_samples = {}

print("Samples per label:")
counter = 0
all_counter = 0
for label in data['intents'].keys():
	print(f"{label:30s}: {len(data['intents'][label]['utterances'])}")
	all_counter += len(data['intents'][label]['utterances'])
	quantity_for_test = len(data['intents'][label]['utterances']) * config['test-percentage'] / float(100)
	#if quantity_for_test < 1:
	#	continue
	test_samples[label] = []
	
	#while counter < quantity_for_test:
	#choice = random.choice(data['intents'][label]['utterances'])
	for choice in data['intents'][label]['utterances']:
		test_samples[label].append(' '.join([item['text'] for item in choice['data']]))
		#data['intents'][label]['utterances'].remove(choice)
		counter += 1
print(f"Found {counter}/{all_counter} test samples")

fo.write_json(config['paths']['datasets']['snips-nlu']['train'], data)
fo.write_json(config['paths']['datasets']['snips-nlu']['test'], test_samples)
