import json, random, sys

sys.path.append("../")
from file_operators import read_json, write_json

TEST_PERCENTAGE = 20

INPUT_FILE = "../../dataset/snips-nlu/data.json"

TRAIN_FILE = "../../dataset/snips-nlu/train.json"
TEST_FILE = "../../dataset/snips-nlu/test.json"

data = read_json(INPUT_FILE)

test_samples = {}

print("Samples per label:")
for label in data['intents'].keys():
	print(f"{label:30s}: {len(data['intents'][label]['utterances'])}")
	quantity_for_test = len(data['intents'][label]['utterances']) * TEST_PERCENTAGE / float(100)
	if quantity_for_test < 1:
		continue
	print("Selecting samples for test...")
	test_samples[label] = []
	counter = 0
	while counter < quantity_for_test:
		choice = random.choice(data['intents'][label]['utterances'])
		test_samples[label].append(' '.join([item['text'] for item in choice['data']]))
		data['intents'][label]['utterances'].remove(choice)
		counter += 1

write_json(TRAIN_FILE, data)
write_json(TEST_FILE, test_samples)
