import json, random, sys, collections

from .. import file_operators as fo, utils

config = utils.load_config(utils.parse_args().config)

data = fo.read_lines(config['paths']['datasets']['vw']['data'])

labelled_data = collections.defaultdict(list)
data_test = []

for line in data:
	labelled_data[line.split('|')[0].split('/')[1]].append(line)

print("Samples per label:")
for label in labelled_data.keys():
 	print(f"{label:30s}: {len(labelled_data[label])}")
 	quantity_for_test = len(labelled_data[label]) * config['test-percentage'] / float(100)
 	if quantity_for_test < 1:
 		continue
 	counter = 0
 	while counter < quantity_for_test:
 		choice = random.choice(labelled_data[label])
 		data_test.append(choice)
 		labelled_data[label].remove(choice)
 		counter += 1

print(f"Input data size: {len(data)}")
data = [item for subset in labelled_data.values() for item in subset]

print(f"Resulting train size: {len(data)}")
print(f"Resulting test size: {len(data_test)}")

random.shuffle(data)
random.shuffle(data_test)

fo.write_lines(config['paths']['datasets']['vw']['train'], data)
fo.write_lines(config['paths']['datasets']['vw']['test'], data_test)
