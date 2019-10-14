import sys
import pytext
import numpy as np
from .. import file_operators as fo, utils
from .. import metrics

config = utils.load_config(utils.parse_args().config)

def get_best_label(result):
    doc_label_scores_prefix = ('scores:' if any(r.startswith('scores:') for r in result) else 'doc_scores:')
    return max((label for label in result if label.startswith(doc_label_scores_prefix)), key=lambda label: result[label][0],)[len(doc_label_scores_prefix):]

def get_best_slots(result):
    slot_prefix = "word_scores"
    word_scores = {label.split(':')[1]: result[label] for label in result if label.startswith(slot_prefix)}
    word_labels = []
    for i in range(len(word_scores[list(word_scores.keys())[0]])):
        best_label = '--'
        best_score = -100
        for label in word_scores.keys():
            if word_scores[label][i] > best_score:
                best_score = word_scores[label][i]
                best_label = label
        word_labels.append(best_label)
    return word_labels
    #doc_label_scores_prefix = ('scores:' if any(r.startswith('scores:') for r in result) else 'doc_scores:')
    #return max((label for label in result if label.startswith(doc_label_scores_prefix)), key=lambda label: result[label][0],)[len(doc_label_scores_prefix):]

configp = pytext.load_config(config['paths']['etc']['pytext']['model-config-extended'])
predictor = pytext.create_predictor(configp, config['paths']['etc']['pytext']['model-extended'])

test_dataset = fo.read_json(config['paths']['datasets']['pytext']['test-extended'])

counter = 0
positive_counter = 0
total_recall = []
print(f"{'Sample':80s}\t{'recognized-label':20s}\t{'true-label':20s}\t{'correctly-recognized':30s}")
for label in test_dataset.keys():
    for sample in test_dataset[label]:
        recognized = [slot for slot in get_best_slots(predictor({"text": sample["text"].lower(), "doc_weight": 1, "word_weight": 1})) if slot != '__UNKNOWN__']
        print(f'Recognized: {recognized}')
        print(f"True: {sample['slots']}")
        total_recall.append(metrics.get_recall(recognized, sample['slots']))

        recognized_label = get_best_label(predictor({"text": sample['text'].lower(), "doc_weight": 1, "word_weight": 1}))
        if not recognized_label:
            recognized_label = '-'
        print(f"{sample['text']:80s}\t{recognized_label:20s}\t{label:20s}\t{recognized_label==label}")
        if recognized_label == label:
            positive_counter += 1
        counter += 1

print(f"Correctly recognized {positive_counter} of {counter} ({round(positive_counter / float(counter) * 100, 2)} %)")
print(f"Average slot recall is {round(np.mean(total_recall), 4)}")
