import json, random, sys

from .. import file_operators as fo, utils

config = utils.load_config(utils.parse_args().config)

data = fo.read_lines(config['paths']['datasets']['pytext']['data-extended'])

test_samples = {}
validate_samples = []

NO_SLOT_MARK = "NoLabel"

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

 		text = choice.split('\t')[-3]
 		slots = [slot.split(':') for slot in choice.split('\t')[1].split(',')]
 		all_slots = {tuple(map(int, slot[0:2])): slot[2] for slot in slots}
 		space_indices = [i for i in range(len(text)) if text.startswith(' ', i)]
 		words_indices = []
 		for i in range(len(space_indices)):
 			if len(words_indices) == 0:
 				words_indices.append((0, space_indices[i] - 1))
 			else:
 				words_indices.append((space_indices[i - 1] + 1, space_indices[i] - 1))
 		words_indices.append((space_indices[-1] + 1, len(text) - 1))
 		enriched_slots = [all_slots.get(pair, NO_SLOT_MARK) for pair in words_indices]

 		print(f"Text: {text}")
 		print(f"Slots: {slots}")
 		print(f"Space indices: {space_indices}")
 		print(f"Word indices: {words_indices}")
 		print(f"All slots: {all_slots}")
 		print(f"Enriched slots: {enriched_slots}")


 		test_samples[label].append({'text': choice.split('\t')[-3], 'slots': enriched_slots })#[slot.split(':')[-1] for slot in choice.split('\t')[1].split(',')]})
 		validate_samples.append(choice)
 		samples[label].remove(choice)
 		counter += 1
print(f"{len([sample for label in samples.keys() for sample in samples[label]])} Train samples; {len(test_samples)} Test samples")
fo.write_lines(config['paths']['datasets']['pytext']['train-extended'], [sample for label in samples.keys() for sample in samples[label]])
fo.write_lines(config['paths']['datasets']['pytext']['validate-extended'], validate_samples)
fo.write_json(config['paths']['datasets']['pytext']['test-extended'], test_samples)
