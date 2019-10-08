import json, random, sys, os
from . import normalize as n

from .. import file_operators as fo, utils

config = utils.load_config(utils.parse_args().config)

test_samples = []
train_samples = []

labels = enumerate(fo.read_lines(os.path.join(config['paths']['datasets']['joint-nlu']['data'], config['paths']['datasets']['joint-nlu']['filenames']['data']['labels'])))
texts = fo.read_lines(os.path.join(config['paths']['datasets']['joint-nlu']['data'], config['paths']['datasets']['joint-nlu']['filenames']['data']['texts']))
slots = fo.read_lines(os.path.join(config['paths']['datasets']['joint-nlu']['data'], config['paths']['datasets']['joint-nlu']['filenames']['data']['slots']))

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
  	quantity_for_test = len(samples[label]) * config['test-percentage'] / float(100)
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

n.write_dataset(config['paths']['datasets']['joint-nlu']['test'], test_samples, config)
n.write_dataset(config['paths']['datasets']['joint-nlu']['validate'], test_samples, config)
n.write_dataset(config['paths']['datasets']['joint-nlu']['train'], train_samples, config)
