import os

def list_dataset_files(dataset_path):
	annotators_folders = [os.path.join(dataset_path, annotator_folder) for annotator_folder in os.listdir(dataset_path)]
	return [os.path.join(folder, file) for folder in annotators_folders for file in os.listdir(folder)]
