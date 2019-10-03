import os, re, sys
import json
import pandas as pd
import re
import numpy as np

from .. import field_extractors as fe, converters, utils, file_operators as fo

command_types = {1: "Command", 2: "IndirectCommand", 3: "Request"}

def get_words(words, command_size):
	i = 0
	while i < len(words):
		word_len = int(np.random.normal(5, 2))
		if word_len <= 0:
			continue
		if i + word_len + 1 >= len(words):
			yield words[i:]
		else:
			yield words[i:i + word_len]
		i += word_len

def handle(words, command_size = 4):
	id = 0
	for words in get_words(words, command_size):
		fo.write_json(os.path.join(config['paths']['datasets']['not-commands']['annotations'], f"command_{id}.json"), {"utterance-type": "Other", 'text': [' '.join(words)]})
		id += 1

if __name__ == "__main__":
	config = utils.load_config(utils.parse_args().config)
	handle([word for word in re.split('[^а-яА-ЯёЁ]+', ' '.join(fo.read_lines(config['paths']['datasets']['not-commands']['data'])).lower()) if word != ''], 5)
#for i, entry in enumerate(fo.read_lines(config['paths']['datasets']['iot-commands']['data'])):
#handle_entry(i, entry)