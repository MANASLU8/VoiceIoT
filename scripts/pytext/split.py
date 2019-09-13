import json, random, sys

sys.path.append("../")
from file_operators import read_lines, write_lines, write_json

TEST_PERCENTAGE = 20

INPUT_FILE = "../../dataset/pytext/data.tsv"

TRAIN_FILE = "../../dataset/pytext/train.tsv"
TEST_FILE = "../../dataset/pytext/test.json"
VALIDATE_FILE = "../../dataset/pytext/validate.tsv"

data = read_lines(INPUT_FILE)

test_samples = {}
validate_samples = []

# group samples by labels
samples = {}
for sample in data:
	label = sample.split('\t')[0]
	if label in samples:
		samples[label].append(sample)
	else:
		samples[label] = [sample]

print("Samples per label:")
for label in samples.keys():
 	print(f"{label:80s}: {len(samples[label])}")
 	quantity_for_test = len(samples[label]) * TEST_PERCENTAGE / float(100)
 	if quantity_for_test < 1:
 		continue
 	test_samples[label] = []
 	counter = 0
 	while counter < quantity_for_test:
 		choice = random.choice(samples[label])
 		test_samples[label].append(choice.split('\t')[-1])
 		validate_samples.append(choice)
 		samples[label].remove(choice)
 		counter += 1

write_lines(TRAIN_FILE, [sample for label in samples.keys() for sample in samples[label]])
write_lines(VALIDATE_FILE, validate_samples)
write_json(TEST_FILE, test_samples)
