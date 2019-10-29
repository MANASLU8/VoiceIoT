import pickle
import sys
import os
from slize import utils, model
from tensorflow.python.keras.utils import to_categorical
from .. import file_operators as fo, utils as u
from .. import metrics

import numpy as np
import argparse

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

configp = ConfigProto()
configp.gpu_options.allow_growth = True
session = InteractiveSession(config=configp)

TMP_OUT_FILE = 'cv/out.json'

#config = u.load_config(u.parse_args().config)

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--train-file', dest='train', help='train file path')
parser.add_argument('-s', '--test-file', dest='test', help='test file path')
u.add_config_arg(parser)
args = parser.parse_args()
config = u.load_config(args.config)


sys.path.append(config['paths']['slots-intents-module'])

test_dataset = fo.read_json(args.test)

print(f"Fitting model...")

#Load in and format the training & test data
intent_dataset = utils.ATIS(train_file=args.train, test_file=args.train, sentence_length=50, word_length=12)

#Convert all entity-slot labels into vectors (bows)
train_y = to_categorical(np.array(intent_dataset.data["train"]["entities"]), 1 + len(intent_dataset.info["entity_vocab"]))
test_y = to_categorical(np.array(intent_dataset.data["test"]["entities"]), 1 + len(intent_dataset.info["entity_vocab"]))

#Convert all intent labels into vectors (one-hot)
train_i = utils.one_hot(intent_dataset.data["train"]["intents"], len(intent_dataset.info["intent_vocab"]))
test_i = utils.one_hot(intent_dataset.data["test"]["intents"], len(intent_dataset.info["intent_vocab"]))

#Convert all words into vectors (poincare)
vec_dimensions = 100
wvectors = utils.wordvectors(config['paths']['etc']['slots-intents']['w2v'])
embedding_matrix = utils.embeddingmatrix(wvectors, vec_dimensions, intent_dataset.info["word_vocab"])

#Configure the Machine Learning Model
machinelearning_model = model.BiLSTM_CRF()
machinelearning_model.build(intent_dataset.info["word_length"], 1+len(intent_dataset.info["entity_vocab"]), len(intent_dataset.info["intent_vocab"]), 2+len(intent_dataset.info["word_vocab"]),2+len(intent_dataset.info["char_vocab"]),word_emb_dims=vec_dimensions,tagger_lstm_dims=100, dropout=0.2)
machinelearning_model.load_embedding_weights(embedding_matrix)

#Train the Machine Learning Model for 50 epochs
train_inputs = [intent_dataset.data["train"]["words"], intent_dataset.data["train"]["chars"]]
test_inputs = [intent_dataset.data["test"]["words"], intent_dataset.data["test"]["chars"]]
train_outputs = [train_i, train_y]
test_outputs = [test_i, test_y]
machinelearning_model.fit(train_inputs, train_outputs, batch_size = 32, epochs = 10, validation = (test_inputs, test_outputs))

#print(args.train)

#print(train_file, test_file)

counter = 0
positive_counter = 0

total_recall = []
output_commands = []
print("Evaluating results...")
# load in pretrained model & corresponding vocab mappings
loaded_model = machinelearning_model #model.BiLSTM_CRF()
#loaded_model.load(os.path.join(config['paths']['models']['slots-intents']['path'], config['paths']['models']['slots-intents']['name']))
with open(config['paths']['datasets']['slots-intents']['info'],'rb') as f:
   loaded_info = pickle.load(f)

print(f"{'Sample':80s}\t{'recognized-label':20s}\t{'true-label':20s}\t{'correctly-recognized':30s}")
for label in test_dataset.keys():
	for sample in test_dataset[label]:
		intent, entities = utils.intent_entities(sample['command'], loaded_model, loaded_info)
		true_slots = [slot.lower() for slot in sample['slots']]
		recognized_slots = [(slot[0], slot[1] if slot[1] else '-') for slot in entities]
		
		recall = metrics.get_recall(true_slots, [slot[1] for slot in recognized_slots])
		total_recall.append(recall)
		recognized_label = intent[0]

		for command_word in sample['command'].split(' '):
			slotted = False
			for item in recognized_slots:
				if item[0] == command_word:
					slotted = True
			if not slotted:
				recognized_slots.append((command_word, '-'))

		if not recognized_label:
			recognized_label = '-'
		print(f"{sample['command']:80s}\t{recognized_label:20s}\t{label.lower():20s}\t{recognized_label==label.lower()}")
		if recognized_label == label.lower():
			positive_counter += 1
		counter += 1
		output_commands.append({'command': recognized_slots, 'device': sample['device']})

fo.write_json(TMP_OUT_FILE.split('.')[0]+f'_{args.test.split("_")[-1].split(".")[0]}.'+TMP_OUT_FILE.split('.')[1], output_commands)
print(f"Correctly recognized {positive_counter} of {counter} ({round(positive_counter / float(counter) * 100, 2)} %)")
print(f"Average slot recall is {round(np.mean(total_recall), 4)}")