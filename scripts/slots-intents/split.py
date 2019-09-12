import json, random

TEST_PERCENTAGE = 20

INPUT_FILE = "../../dataset/slots-intents/data.csv"

TRAIN_FILE = "../../dataset/slots-intents/train.csv"
TEST_FILE = "../../dataset/slots-intents/test.json"
VALIDATE_FILE = "../../dataset/slots-intents/validate.csv"

with open(INPUT_FILE) as f:
	data = [line.replace('\n', '') for line in f.readlines()]

test_samples = {}
validate_samples = []

# group samples by labels
samples = {}
for sample in data:
	label = sample.split(' ')[-1]
	if label in samples:
		samples[label].append(sample)
	else:
		samples[label] = [sample]

print("Samples per label:")
for label in samples.keys():
 	print(f"{label:30s}: {len(samples[label])}")
 	quantity_for_test = len(samples[label]) * TEST_PERCENTAGE / float(100)
 	if quantity_for_test < 1:
 		continue
 	test_samples[label] = []
 	counter = 0
 	while counter < quantity_for_test:
 		choice = random.choice(samples[label])
 		test_samples[label].append(' '.join(choice.split('\t')[0].split(' ')[1:-2]))
 		validate_samples.append(choice)
 		samples[label].remove(choice)
 		counter += 1

with open(TRAIN_FILE, "w") as f:
 	f.write('\n'.join([sample for label in samples.keys() for sample in samples[label] ]))

with open(VALIDATE_FILE, "w") as f:
 	f.write('\n'.join(validate_samples))

with open(TEST_FILE, "w") as f:
 	f.write(json.dumps(test_samples, indent=2).encode().decode('unicode-escape'))