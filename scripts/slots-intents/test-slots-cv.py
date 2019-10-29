import pickle
import sys
import os
from slize import utils, model

from .. import file_operators as fo, utils as u
from .. import metrics
import time
import numpy as np

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.python.keras.utils import to_categorical

configp = ConfigProto()
configp.gpu_options.allow_growth = True
session = InteractiveSession(config=configp)

#
# Get ready
#

def parse_lines(lines):
	for line in lines:
		text = line.split('BOS ')[1].split(' EOS')[0]
		true_slots = [slot.split('.')[1] if len(slot.split('.')) > 1 else slot for slot in line.split('\t')[1].split(' ')][:-1]
		label = line.split(' ')[-1]
		yield {'command': text, 'slots': true_slots, 'device': label, 'original': line}

def make_sets(chunks):
	for i in range(len(chunks)):
		chunk = chunks[i]
		test_set = {}
		train_set = []
		for sample in chunk:
			label = sample['device']
			if label not in test_set:
				test_set[label] = []
			test_set[label].append(sample)
		yield {'train': [sample['original'] for j in range(len(chunks)) for sample in chunks[j] if i != j], 'test': test_set}

config = u.load_config(u.parse_args().config)
FOLDS = config['cv-folds']
TMP_TRAIN_FILE = 'tmp.csv'

data = fo.read_lines(config['paths']['datasets']['slots-intents']['data'])
# print(data[0])
parsed_data = list(parse_lines(data))
print(f"Parsed {len(parsed_data)} lines")
np.random.shuffle(parsed_data)
chunks = np.array_split(parsed_data, FOLDS)
print(f"Made {len(chunks)} folds for cross-validation")
sets = list(make_sets(chunks))

#
# Perform
#

sys.path.append(config['paths']['slots-intents-module'])

test_dataset = fo.read_json(config['paths']['datasets']['slots-intents']['test'])




accuracies = []
recalls = []
output_commands = []
for i in range(len(sets)):
	train_dataset = sets[i]['train']
	test_dataset = sets[i]['test']

	counter = 0
	positive_counter = 0

	#
	#	Train
	#

	fo.write_lines(TMP_TRAIN_FILE, train_dataset)

	print(f"Fitting model #{i}...")

	#Load in and format the training & test data
	intent_dataset = utils.ATIS(train_file=TMP_TRAIN_FILE, test_file=TMP_TRAIN_FILE, sentence_length=50, word_length=12)

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
	time.sleep(3)
	machinelearning_model.build(intent_dataset.info["word_length"], 1+len(intent_dataset.info["entity_vocab"]), len(intent_dataset.info["intent_vocab"]), 2+len(intent_dataset.info["word_vocab"]),2+len(intent_dataset.info["char_vocab"]),word_emb_dims=vec_dimensions,tagger_lstm_dims=100, dropout=0.2)
	machinelearning_model.load_embedding_weights(embedding_matrix)

	#Train the Machine Learning Model for 50 epochs
	train_inputs = [intent_dataset.data["train"]["words"], intent_dataset.data["train"]["chars"]]
	test_inputs = [intent_dataset.data["test"]["words"], intent_dataset.data["test"]["chars"]]
	train_outputs = [train_i, train_y]
	test_outputs = [test_i, test_y]
	machinelearning_model.fit(train_inputs, train_outputs, batch_size = 32, epochs = 5, validation = (test_inputs, test_outputs))

	#Save the trained Model and Vocab Mappings
	machinelearning_model.save(os.path.join(config['paths']['models']['slots-intents']['path'], config['paths']['models']['slots-intents']['name']))
	with open(config['paths']['datasets']['slots-intents']['info'], 'wb') as f:
	  pickle.dump(intent_dataset.info, f)

	#
	#	Calculate metrics
	#

	print(f"Calculating metrics #{i}...")

	# load in pretrained model & corresponding vocab mappings
	loaded_model = machinelearning_model #model.BiLSTM_CRF()
	#loaded_model.load(os.path.join(config['paths']['models']['slots-intents']['path'], config['paths']['models']['slots-intents']['name']))
	with open(config['paths']['datasets']['slots-intents']['info'],'rb') as f:
	   loaded_info = pickle.load(f)

	total_recall = []
	print(f"{'Sample':80s}\t{'recognized-label':20s}\t{'true-label':20s}\t{'correctly-recognized':30s}")
	for label in test_dataset.keys():
		for sample in test_dataset[label]:
			intent, entities = utils.intent_entities(sample['command'], loaded_model, loaded_info)
			true_slots = [slot.lower() for slot in sample['slots']]
			recognized_slots = entities
			recall = metrics.get_recall(true_slots, recognized_slots)
			print(f"Parsed command: {list(zip(sample['command'].split(' '), recognized_slots))}")
			total_recall.append(recall)
			recognized_label = intent[0]
			if not recognized_label:
				recognized_label = '-'
			print(f"{sample['command']:80s}\t{recognized_label:20s}\t{label.lower():20s}\t{recognized_label==label.lower()}")
			if recognized_label == label.lower():
				positive_counter += 1
			counter += 1

	accuracy = round(positive_counter / float(counter) * 100, 2)
	recall = round(np.mean(total_recall), 4)
	print(f"Correctly recognized {positive_counter} of {counter} ({accuracy} %)")
	print(f"Average slot recall is {recall}")
	accuracies.append(accuracy)
	recalls.append(recall)

fo.write_json("labelled-commands.json", output_commands)
print(f"Mean accuracy: {np.mean(accuracies)}")
print(f"Mean recall: {np.mean(recalls)}")