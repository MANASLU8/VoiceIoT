import pickle
import sys
import os
from slize import utils, model

from .. import file_operators as fo, utils as u
from .. import metrics

import numpy as np

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

configp = ConfigProto()
configp.gpu_options.allow_growth = True
session = InteractiveSession(config=configp)

config = u.load_config(u.parse_args().config)

sys.path.append(config['paths']['slots-intents-module'])

test_dataset = fo.read_json(config['paths']['datasets']['slots-intents']['test'])

counter = 0
positive_counter = 0

total_recall = []

# load in pretrained model & corresponding vocab mappings
loaded_model = model.BiLSTM_CRF()
loaded_model.load(os.path.join(config['paths']['models']['slots-intents']['path'], config['paths']['models']['slots-intents']['name']))
with open(config['paths']['datasets']['slots-intents']['info'],'rb') as f:
   loaded_info = pickle.load(f)

print(f"{'Sample':80s}\t{'recognized-label':20s}\t{'true-label':20s}\t{'correctly-recognized':30s}")
for label in test_dataset.keys():
	for sample in test_dataset[label]:
		intent, entities = utils.intent_entities(sample['text'], loaded_model, loaded_info)
		true_slots = [slot.lower() for slot in sample['slots']]
		recognized_slots = entities
		recall = metrics.get_recall(true_slots, recognized_slots)
		total_recall.append(recall)
		recognized_label = intent[0]
		if not recognized_label:
			recognized_label = '-'
		print(f"{sample['text']:80s}\t{recognized_label:20s}\t{label.lower():20s}\t{recognized_label==label.lower()}")
		if recognized_label == label.lower():
			positive_counter += 1
		counter += 1

print(f"Correctly recognized {positive_counter} of {counter} ({round(positive_counter / float(counter) * 100, 2)} %)")
print(f"Average slot recall is {round(np.mean(total_recall), 4)}")