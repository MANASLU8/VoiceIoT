import json, sys, pickle

sys.path.append("../../vendor/slots_intents/intents_slots")
from model import BiLSTM_CRF
from utils import intent_entities

MODEL = "../../models/slots-intents/model.h5"
DATASET_INFO = "../../dataset/slots-intents/dataset_info"
TEST_FILE = "../../dataset/slots-intents/test.json"

with open(TEST_FILE) as f:
	test_dataset = json.load(f)

counter = 0
positive_counter = 0


# load in pretrained model & corresponding vocab mappings
loaded_model = BiLSTM_CRF()
loaded_model.load(MODEL)
with open(DATASET_INFO,'rb') as f:
   loaded_info = pickle.load(f)

def extract_slots_intents(message):
	intent, entities = intent_entities(message,loaded_model,loaded_info)
	es = {"WORDS":[],"SLOTS (ENTITIES)":[]}
	for word, entity in entities:
		es["WORDS"].append(word)
		es["SLOTS (ENTITIES)"].append(entity)
	return es

for label in test_dataset.keys():
	for sample in test_dataset[label]:
		intent, entities = intent_entities(sample, loaded_model, loaded_info)
		recognized_label = intent[0]
		if not recognized_label:
			recognized_label = '-'
		print(f"{sample:80s}\t{recognized_label:20s}\t{label.lower():20s}")
		if recognized_label == label.lower():
			positive_counter += 1
		counter += 1

print(f"Correctly recognized {positive_counter} of {counter} ({round(positive_counter / float(counter) * 100, 2)} %)")