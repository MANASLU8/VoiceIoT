import sys
import pytext

sys.path.append('../')
from file_operators import read_json

CONFIG_FILE = '../../vendor/pytext/docnn.json'
MODEL_FILE = '../../models/pytext/model.caffe2.predictor'
TEST_FILE = '../../dataset/pytext/test.json'

def get_best_label(result):
    doc_label_scores_prefix = ('scores:' if any(r.startswith('scores:') for r in result) else 'doc_scores:')
    return max((label for label in result if label.startswith(doc_label_scores_prefix)), key=lambda label: result[label][0],)[len(doc_label_scores_prefix):]

config = pytext.load_config(CONFIG_FILE)
predictor = pytext.create_predictor(config, MODEL_FILE)

test_dataset = read_json(TEST_FILE)

counter = 0
positive_counter = 0

print(f"{'Sample':80s}\t{'recognized-label':20s}\t{'true-label':20s}")
for label in test_dataset.keys():
    for sample in test_dataset[label]:
        recognized_label = get_best_label(predictor({"text": sample.lower()}))
        if not recognized_label:
            recognized_label = '-'
        print(f"{sample:80s}\t{recognized_label:20s}\t{label:20s}")
        if recognized_label == label:
            positive_counter += 1
        counter += 1

print(f"Correctly recognized {positive_counter} of {counter} ({round(positive_counter / float(counter) * 100, 2)} %)")
