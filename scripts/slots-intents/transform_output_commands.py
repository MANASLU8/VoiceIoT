from .. import file_operators as fo
input_file = "labelled-commands-slots-intents.json"

samples = fo.read_json(input_file)

pairs = {'devicefeature': 'DeviceFeature', 'commandfeature': 'CommandFeature', 'command': 'COMMAND', 'loccurrent': 'LOCCurrent'}
for sample in samples:
	for pair in sample['command']:
		suffix = pair[1].split('.')[1] if len(pair[1].split('.')) > 1 else pair[1]
		if pair[1] != '-':
			if suffix not in pairs:
				print(suffix)
			else:
				pair[1] = pairs[suffix]

fo.write_json(input_file, samples)