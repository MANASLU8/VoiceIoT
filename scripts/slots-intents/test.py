import json, sys, pickle

from .. import file_operators as fo, utils

config = utils.load_config(utils.parse_args().config)

sys.path.append(config['paths']['slots-intents-module'])
from model import BiLSTM_CRF
from utils import intent_entities

test_dataset = fo.read_json(config['paths']['datasets']['slots-intents']['test'])

counter = 0
positive_counter = 0

# load in pretrained model & corresponding vocab mappings
loaded_model = BiLSTM_CRF()
loaded_model.load(config['paths']['etc']['slots-intents']['model'])
with open(config['paths']['etc']['slots-intents']['dataset-info'],'rb') as f:
   loaded_info = pickle.load(f)

print(f"{'Sample':80s}\t{'recognized-label':20s}\t{'true-label':20s}\t{'correctly-recognized':30s}")
for label in test_dataset.keys():
	for sample in test_dataset[label]:
		intent, entities = intent_entities(sample, loaded_model, loaded_info)
		recognized_label = intent[0]
		if not recognized_label:
			recognized_label = '-'
		print(f"{sample:80s}\t{recognized_label:20s}\t{label.lower():20s}\t{recognized_label==label.lower()}")
		if recognized_label == label.lower():
			positive_counter += 1
		counter += 1

print(f"Correctly recognized {positive_counter} of {counter} ({round(positive_counter / float(counter) * 100, 2)} %)")
