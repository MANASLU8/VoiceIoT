import os, re, sys
import json
import itertools
import argparse
import pandas
import numpy as np
from .. import field_extractors as fe, converters, utils, file_operators as fo

from snorkel.labeling import PandasLFApplier, labeling_function
from snorkel.augmentation import transformation_function, PandasTFApplier

from snorkel.augmentation import RandomPolicy

parser = argparse.ArgumentParser()
utils.add_config_arg(parser)

args = parser.parse_args()
config = utils.load_config(args.config)

LAMP = 0
WASHING_MACHINE = 1
OTHER = 2

device_names = {
	LAMP: "Lamp",
	WASHING_MACHINE: "WashingMachine",
	OTHER: "Other"
}

def get_texts_as_df(input_files, get_file_names=False):
	texts = []
	for j in range(len(input_files)):
		input_file = input_files[j]
		with open(input_file) as f:
			annotation = json.loads(f.read())
		text = fe.extract_text(annotation)
		if not text:
			continue
		texts.append((input_files[j], text))
	return pandas.DataFrame(texts, columns = ("filename", "text"))

def write_annotated(df, annotations):
	for _, item in df.iterrows():
		fo.write_json(item.filename, {'text': item.text, 'ontology-label': item.label, 'slots-indices': item.slots})

@labeling_function()
def lamp(x):
    return LAMP if "свет" in x.text.lower() else OTHER

@labeling_function()
def washing_machine(x):
    return WASHING_MACHINE if "стирал" in x.text.lower() else OTHER

@transformation_function()
def get_slots(x):
    command = []
    words = x.text.split(' ')
    for word_index in range(len(words)):
    	if re.search(r".+(ить|ай)", words[word_index]):
    		command.append(word_index)
    x.slots = {'COMMAND': command}
    return x

random_policy = RandomPolicy(
    1, sequence_length=1, n_per_original=1, keep_original=False
)

if __name__ == "__main__":
	df = get_texts_as_df(utils.list_dataset_files(config['paths']['datasets']['unannotated']))

	lfs = [lamp, washing_machine]
	applier = PandasLFApplier(lfs=lfs)
	slots_applier = PandasTFApplier(tfs = (get_slots, ), policy = random_policy)
	labels = applier.apply(df=df)
	df['slots'] = [{} for i in range(df.shape[0])]
	df = slots_applier.apply(df=df)
	df['label'] = list(map(lambda values: device_names[np.min(values)], labels))
	write_annotated(df, labels)
