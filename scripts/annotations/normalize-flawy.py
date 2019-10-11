import os, re, sys
import json
import itertools
import argparse
from functools import reduce

from .. import field_extractors as fe, converters, utils, file_operators as fo

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', dest='file', help='path to file for normalization')
parser.add_argument('--nice', dest='nice', action='store_true', help='denormalize labelled commands as well as unlabelled')
utils.add_config_arg(parser)
args = parser.parse_args()
config = utils.load_config(args.config)

def handle_file(input_file, get_file_names=False):
	flawy = fo.read_json(input_file)
	for file in flawy.keys():
		fo.write_json(file, flawy[file])

def handle_many(input_files, get_file_names=False):
	flawy = reduce(lambda x,y: dict(x, **y), map(fo.read_json, input_files))
	for file in flawy.keys():
		fo.write_json(file, flawy[file])

if __name__ == "__main__":
	handle_file(args.file if args.file else config['paths']['datasets']['flawy-denormalized'])

	if args.nice:
		handle_many([args.file if args.file else config['paths']['datasets']['flawy-denormalized'], config['paths']['datasets']['nice-denormalized']])
	else:
		handle_file(args.file if args.file else config['paths']['datasets']['flawy-denormalized'])
