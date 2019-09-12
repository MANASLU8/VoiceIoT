import json, random

TEST_PERCENTAGE = 20

INPUT_FILE = "../../dataset/snips-nlu/data.json"

TRAIN_FILE = "../../dataset/snips-nlu/train.json"
TEST_FILE = "../../dataset/snips-nlu/test.json"

with open(INPUT_FILE) as f:
	data = json.load(f)

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

with open(TRAIN_FILE, "w") as f:
	f.write(json.dumps(data, indent=2).encode().decode('unicode-escape'))

with open(TEST_FILE, "w") as f:
	f.write(json.dumps(test_samples, indent=2).encode().decode('unicode-escape'))


