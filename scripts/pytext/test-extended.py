import numpy as np
import pytext

from .. import file_operators as fo, utils
from .. import metrics
from .. import request_mapping as ul

config = utils.load_config(utils.parse_args().config)

ontology_devices = ["AirConditioning", "AlarmClock", "AudioSystem", "CoffeeMachine", "CurtainsRelay", "DoorBell", "FloorLite", "HeatingFloor", "Printer", "SmartFridge", "SmartMicrowave", "WindowRelay", "LightSensor", "Lamp", "SmartBreadMaker", "WashingMachine"]

def get_best_label(result):
    doc_label_scores_prefix = ('scores:' if any(r.startswith('scores:') for r in result) else 'doc_scores:')
    return max((label for label in result if label.startswith(doc_label_scores_prefix)),
               key=lambda label: result[label][0], )[len(doc_label_scores_prefix):]


def url_to_device_name(url):
    return url.split('/')[-1][:-1]


def get_best_slots(result):
    slot_prefix = "word_scores"
    word_scores = {label.split(':')[1]: result[label] for label in result if label.startswith(slot_prefix)}
    word_labels = []
    for i in range(len(word_scores[list(word_scores.keys())[0]])):
        best_label = '--'
        best_score = 100
        for label in word_scores:
            if word_scores[label][i] < best_score:
                best_score = word_scores[label][i]
                best_label = label
        word_labels.append(best_label)
    return word_labels


configp = pytext.load_config(config['paths']['etc']['pytext']['model-config-extended'])
predictor = pytext.create_predictor(configp, config['paths']['etc']['pytext']['model-extended'])

test_dataset = fo.read_json(config['paths']['datasets']['pytext']['test-extended'])

counter = 0
ontology_counter = 0
ontology_raw_counter = 0
positive_counter = 0
total_recall = []
# print(f"{'Sample':80s}\t{'recognized-label':20s}\t{'true-label':20s}\t{'correctly-recognized':30s}")
print(f"{'command'}\t{'true label'}\t{'ontology label'}\t{'ontology raw label'}\t{'recognized label'}\t{'additional labels'}")
for label in test_dataset.keys():
    if label not in ontology_devices:
        continue
    for sample in test_dataset[label]:
        true_label = sample['intent']
        recognized = [slot for slot in get_best_slots(predictor(
            {"text": sample["text"].lower(), "doc_weight": 1, "word_weight": 1}))]  # if slot != '__UNKNOWN__']
        parsed_command = list(zip(sample['text'].lower().split(' '), recognized))
        parsed_right_command = list(zip(sample['text'].lower().split(' '), sample['slots']))

        # print(f"-- Recognized slots")
        ul.get_labels(parsed_command, filename=config['paths']['datasets']['request-mapping']['lemmas-fine-splitted'])

        # print(f"-- Right slots")
        labels = ul.get_labels(parsed_right_command,
                               filename=config['paths']['datasets']['request-mapping']['lemmas-fine-splitted'])

        # print(f"Label: {true_label}")
        # print(f"Ontology label: {labels[0].split('/')[-1][:-1]}")

        ontology_label = url_to_device_name(labels[0])

        if ontology_label == true_label:
            ontology_counter += 1
            ontology_label += "*"

        raw_labels = ul.get_raw_labels(parsed_right_command,
                                       filename=config['paths']['datasets']['request-mapping']['lemmas-fine-splitted'])
        # print(f"Raw labels for right command: {raw_labels}")

        raw_ontology_label = url_to_device_name(raw_labels[0])

        if raw_ontology_label == true_label:
            ontology_raw_counter += 1
            raw_ontology_label += "*"

        total_recall.append(metrics.get_recall(recognized, sample['slots']))

        recognized_label = get_best_label(
            predictor({"text": sample['text'].lower(), "doc_weight": 1, "word_weight": 1}))
        if not recognized_label:
            recognized_label = '-'
        # print(f"{sample['text']:80s}\t{recognized_label:20s}\t{label:20s}\t{recognized_label==label}")
        if recognized_label == label:
            positive_counter += 1
            recognized_label += "*"
        counter += 1

        print(f"{sample['text']}\t{label}\t{ontology_label}\t{raw_ontology_label}\t{recognized_label}\t{labels[1] if len(labels) > 1 else ''}")

print(f"Correctly recognized {positive_counter} of {counter} ({round(positive_counter / float(counter) * 100, 2)} %)")
print(
    f"(Ontology) Correctly recognized {ontology_counter} of {counter} ({round(ontology_counter / float(counter) * 100, 2)} %)")
print(
    f"(Ontology raw) Correctly recognized {ontology_raw_counter} of {counter} ({round(ontology_raw_counter / float(counter) * 100, 2)} %)")
print(f"Average slot recall is {round(np.mean(total_recall), 4)}")
