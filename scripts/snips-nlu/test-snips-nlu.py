from snips_nlu import SnipsNLUEngine
import json

with open("../../dataset/snips_nlu/train-all.json") as f:
	sample_dataset = json.load(f)
engine = SnipsNLUEngine()
engine.fit(sample_dataset)

while True:
	print(json.dumps(engine.parse(input("> ")), indent=4, sort_keys=True).encode().decode('unicode-escape'))
