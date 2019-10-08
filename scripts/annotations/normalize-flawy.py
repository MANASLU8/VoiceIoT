import os, re, sys
import json
import itertools
import argparse

from .. import field_extractors as fe, converters, utils, file_operators as fo
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', dest='file', help='path to file for normalization')
utils.add_config_arg(parser)
args = parser.parse_args()
config = utils.load_config(args.config)

def handle_file(input_file, get_file_names=False):
	flawy = fo.read_json(input_file)
	for file in flawy.keys():
		fo.write_json(file, flawy[file])

if __name__ == "__main__":
	handle_file(args.file if args.file else config['paths']['datasets']['flawy-denormalized'])
