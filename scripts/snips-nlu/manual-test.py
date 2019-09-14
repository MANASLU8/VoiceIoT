from snips_nlu import SnipsNLUEngine
import json

from .. import file_operators as fo, utils

config = utils.load_config(utils.parse_args().config)

sample_dataset = fo.read_json(config['paths']['datasets']['snips-nlu']['train'])
engine = SnipsNLUEngine()
engine.fit(sample_dataset)

while True:
	print(json.dumps(engine.parse(input("> ")), indent=4, sort_keys=True).encode().decode('unicode-escape'))
