from snips_nlu import SnipsNLUEngine
import json

TRAIN_FILE = "../../dataset/snips_nlu/train.json"
TEST_FILE = "../../dataset/snips_nlu/test.json"

with open(TRAIN_FILE) as f:
	train_dataset = json.load(f)

with open(TEST_FILE) as f:
	test_dataset = json.load(f)

engine = SnipsNLUEngine()
engine.fit(train_dataset)

counter = 0
positive_counter = 0

print(f"{'Sample':80s}\t{'recognized-label':20s}\t{'true-label':20s}")
for label in test_dataset.keys():
	for sample in test_dataset[label]:
		recognized_label = engine.parse(sample)['intent']['intentName']
		if not recognized_label:
			recognized_label = '-'
		print(f"{sample:80s}\t{recognized_label:20s}\t{label:20s}")
		if recognized_label == label:
			positive_counter += 1
		counter += 1

print(f"Correctly recognized {positive_counter} of {counter} ({round(positive_counter / float(counter) * 100, 2)} %)")
