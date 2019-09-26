import os, yaml, argparse

def list_dataset_files(dataset_path):
	annotators_folders = [os.path.join(dataset_path, annotator_folder) for annotator_folder in os.listdir(dataset_path)]
	return [os.path.join(folder, file) for folder in annotators_folders for file in os.listdir(folder)]

def load_config(config_file):
    with open(config_file, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

def parse_args():
	parser = argparse.ArgumentParser(description='Dataset handler')
	add_config_arg(parser)
	return parser.parse_args()

def add_config_arg(parser):
	parser.add_argument('-c', '--config', action="store", dest="config", default="config.yaml")
	return parser