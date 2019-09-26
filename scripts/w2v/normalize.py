import os, re, sys
import json
import pandas as pd

from .. import field_extractors as fe, converters, utils, file_operators as fo

def handle_files(input_files):
	result = []
	for j in range(len(input_files)) :
		input_file = input_files[j]
		print(f'handling file {input_file}')
		with open(input_file) as f:
			annotation = json.loads(f.read())
		if "utterance-type" not in annotation:
			continue
		utterance_type = fe.extract_utterance_type(annotation)
		text = re.sub(r'[^\w\s]','',fe.extract_text(annotation)).lower().split(' ')
		result.append({"type": utterance_type, "text": text})
	return pd.DataFrame.from_dict({"type": [item["type"] for item in result], "text": [" ".join(item["text"]) for item in result]})

if __name__ == "__main__":
	config = utils.load_config(utils.parse_args().config)
	print(handle_files(utils.list_dataset_files(config['paths']['datasets']['annotations'])).groupby('type')['text'].nunique())
