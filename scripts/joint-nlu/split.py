import json, random, sys, os

from normalize import write_dataset

sys.path.append("../")
from file_operators import read_lines, write_lines, write_json


TEST_PERCENTAGE = 20

INPUT_FOLDER = "../../dataset/joint-nlu/data"

TRAIN_FOLDER = "../../dataset/joint-nlu/train"
TEST_FOLDER = "../../dataset/joint-nlu/test"
VALIDATE_FOLDER = "../../dataset/joint-nlu/valid"

test_samples = []
train_samples = []

labels = enumerate(read_lines(os.path.join(INPUT_FOLDER, "label")))
texts = read_lines(os.path.join(INPUT_FOLDER, "seq.in"))
slots = read_lines(os.path.join(INPUT_FOLDER, "seq.out"))

# group samples by labels
samples = {}
for label in labels:
	if label[1] not in samples:
		samples[label[1]] = [label]
	else:
		samples[label[1]].append(label)

print("Samples per label:")
for label in samples.keys():
  	print(f"{label:30s}: {len(samples[label])}")
  	quantity_for_test = len(samples[label]) * TEST_PERCENTAGE / float(100)
  	if quantity_for_test < 1:
  		continue
  	counter = 0
  	while counter < quantity_for_test:
  		choice = random.choice(samples[label])
  		i = choice[0]
  		test_samples.append([texts[i], slots[i], choice[1]])
  		samples[label].remove(choice)
  		counter += 1

for label in samples.keys():
	for item in samples[label]:
		i = item[0]
		train_samples.append([texts[i], slots[i], item[1]])

write_dataset(TEST_FOLDER, test_samples)
write_dataset(VALIDATE_FOLDER, test_samples)
write_dataset(TRAIN_FOLDER, train_samples)
