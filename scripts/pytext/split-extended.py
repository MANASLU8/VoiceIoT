import json, random, sys

from .. import file_operators as fo, utils

config = utils.load_config(utils.parse_args().config)

data = fo.read_lines(config['paths']['datasets']['pytext']['data-extended'])

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
 	quantity_for_test = len(samples[label]) * config['test-percentage'] / float(100)
 	if quantity_for_test < 1:
 		continue
 	test_samples[label] = []
 	counter = 0
 	while counter < quantity_for_test:
 		choice = random.choice(samples[label])
 		test_samples[label].append({'text': choice.split('\t')[-3], 'slots': [slot.split(':')[-1] for slot in choice.split('\t')[1].split(',')]})
 		validate_samples.append(choice)
 		samples[label].remove(choice)
 		counter += 1
print(f"{len([sample for label in samples.keys() for sample in samples[label]])} Train samples; {len(test_samples)} Test samples")
fo.write_lines(config['paths']['datasets']['pytext']['train-extended'], [sample for label in samples.keys() for sample in samples[label]])
fo.write_lines(config['paths']['datasets']['pytext']['validate-extended'], validate_samples)
fo.write_json(config['paths']['datasets']['pytext']['test-extended'], test_samples)
