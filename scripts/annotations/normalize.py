import os, re, sys
import json
import pandas as pd

from .. import field_extractors as fe, converters, utils, file_operators as fo

command_types = {1: "Command", 2: "IndirectCommand", 3: "Request"}

def handle_entry(id, entry):
	splitted_entry = entry.split(",")
	fo.write_json(os.path.join(config['paths']['datasets']['iot-commands']['annotations'], f"command_{id}.json"), {"utterance-type": command_types[int(splitted_entry[0])], 'text': [splitted_entry[1]]})

if __name__ == "__main__":
	config = utils.load_config(utils.parse_args().config)
	for i, entry in enumerate(fo.read_lines(config['paths']['datasets']['iot-commands']['data'])):
		handle_entry(i, entry)
