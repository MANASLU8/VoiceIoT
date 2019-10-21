import sys
import pytext

from .. import file_operators as fo, utils

config = utils.load_config(utils.parse_args().config)

def get_best_label(result):
    doc_label_scores_prefix = ('scores:' if any(r.startswith('scores:') for r in result) else 'doc_scores:')
    return max((label for label in result if label.startswith(doc_label_scores_prefix)), key=lambda label: result[label][0],)[len(doc_label_scores_prefix):]
print(config)
configp = pytext.load_config(config['paths']['etc']['pytext']['model-config'])
predictor = pytext.create_predictor(configp, config['paths']['etc']['pytext']['model'])

test_dataset = fo.read_json(config['paths']['datasets']['pytext']['test'])

counter = 0
positive_counter = 0

print(f"{'Sample':80s}\t{'recognized-label':20s}\t{'true-label':20s}\t{'correctly-recognized':30s}")
for label in test_dataset.keys():
    for sample in test_dataset[label]:
        #print(predictor({"text": sample.lower()}).keys())
        recognized_label = get_best_label(predictor({"text": sample.lower()}))
        if not recognized_label:
            recognized_label = '-'
        print(f"{sample:80s}\t{recognized_label:20s}\t{label:20s}\t{recognized_label==label}")
        if recognized_label == label:
            positive_counter += 1
        counter += 1

print(f"Correctly recognized {positive_counter} of {counter} ({round(positive_counter / float(counter) * 100, 2)} %)")
