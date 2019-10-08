import os, re, sys
import json
import itertools

from .. import field_extractors as fe, converters, utils, file_operators as fo

config = utils.load_config(utils.parse_args().config)

def handle_file(input_file, get_file_names=False):
	flawy = fo.read_json(input_file)
	for file in flawy.keys():
		fo.write_json(file, flawy[file])

if __name__ == "__main__":
	handle_file(config['paths']['datasets']['flawy-denormalized'])
