import collections
import json
lemmas = collections.defaultdict(lambda: collections.defaultdict(lambda: ['']))

def write_json(filename, content):
	with open(filename, "w") as f:
		f.write(json.dumps(content, indent=2).encode().decode('unicode-escape'))

with open("lemmas.txt") as file:
	for line in file.readlines():
		splitted = line.replace('\n', '').split('\t')
		lemmas[splitted[0]]['system'].append(splitted[1])
		lemmas[splitted[0]]['feature'].append(splitted[2])
		lemmas[splitted[0]]['command'].append(splitted[3])
		lemmas[splitted[0]]['param'].append(splitted[4])

for key in lemmas:
	for k in lemmas[key]:
		lemmas[key][k] = list(set([item for item in lemmas[key][k]])) #if item != '']))

requests = {}
for key in lemmas:
	for system in lemmas[key]['system']:
		if not system in requests:
			requests[system] = dict()
		for feature in lemmas[key]['feature']:
			if feature not in requests[system]:
				requests[system][feature] = dict()
			for command in lemmas[key]['command']:
				if command not in requests[system][feature]:
					requests[system][feature][command] = dict()
				for param in lemmas[key]['param']:
					requests[system][feature][command][param] = '<REQUEST_TYPE>'

write_json('lemmas.json', lemmas)
write_json('requests.json', requests)