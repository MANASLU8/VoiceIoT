import json, pymorphy2
import numpy as np
from .. import utils
import collections

if __name__ == "__main__":
	config = utils.load_config(utils.parse_args().config)

USE_LEMMAS = True
VERBOSE = False
MOST_FREQUENT_LABEL = "--"#"AudioSystem>"

system_slot_name = ["DEVICE", "CommandFeature", "DeviceFeature", "ParameterFeature", "Devicefeature"]
command_slot_name = "COMMAND"
feature_slot_name = ["CommandFeature", "DeviceFeature", "ParameterFeature", "Devicefeature"]
param_slot_name = ["CommandParameter", "DeviceParameter", "DeviceParameterValue", "CommandParameterValue"]

weights = {
	tuple(system_slot_name): 1.0,
	tuple(command_slot_name): 0.9,
	tuple(feature_slot_name): 0.7,
	tuple(param_slot_name): 0.2
}

requests_path = config['paths']['datasets']['request-mapping']['requests'] if __name__ == "__main__" else "requests.json"
lemmas_path = config['paths']['datasets']['request-mapping']['lemmas-fine-splitted'] if __name__ == "__main__" else "lemmas-fine-splitted.json"

empty_result_mark = '<REQUEST_TYPE>'

def read_json(filename):
	with open(filename) as f:
		return json.load(f)

def are_slot_names_equal(hyp_name, ref_names):
	if type(ref_names) == list:
		return hyp_name in ref_names
	else:
		return hyp_name == ref_names

def get_slot_value_single(command, slot_name, morph):
	slot_values = collections.defaultdict(list)
	for pair in command:
		if are_slot_names_equal(pair[1], slot_name): #and slot_value == '':
			slot_values[pair[1]].append(pair[0])
	selected_slot_value = ''
	if type(slot_name) == list:
		for name in slot_name:
			if len(slot_values[name]) > 0:
				selected_slot_value = slot_values[name][0]
				break
	else:
		if len(slot_values[slot_name]) > 0:
			selected_slot_value = slot_values[slot_name][0]
	
	return morph.parse(selected_slot_value)[0].normal_form if USE_LEMMAS else selected_slot_value

def get_slot_value_many(command, slot_name, morph):
	slot_values = collections.defaultdict(list)
	for pair in command:
		if are_slot_names_equal(pair[1], slot_name): #and slot_value == '':
			slot_values[pair[1]].append(pair[0])
	return list(set([morph.parse(slot_value)[0].normal_form if USE_LEMMAS else slot_value for slot_name in slot_values for slot_value in slot_values[slot_name]] + ['']))
	#return result if len(result) > 0 else ['']

def lookup(requests, path):
	node = requests
	for key in path:
		if key in node:
			node = node[key]
		else:
			return None
	return node

def get_request_type(cmd, filename = requests_path, morph = pymorphy2.MorphAnalyzer()): #filename = "requests.json"):
	requests = read_json(filename)
	#print(f"Input command: {cmd}")
	system = get_slot_value_many(cmd, system_slot_name, morph)
	feature = get_slot_value_many(cmd, feature_slot_name, morph)
	command = get_slot_value_many(cmd, command_slot_name, morph)
	param = get_slot_value_many(cmd, param_slot_name, morph)

	slot_sets = np.array(np.meshgrid(system, feature, command, param)).T.reshape(-1,4)

	#print(slot_sets)

	if VERBOSE:
		print("Looking up for:")
		print(f"system = {system}\nfeature = {feature}\ncommand = {command}\nparam = {param}")

	
	lookup_results = list(map(lambda slot_set: lookup(requests, slot_set), slot_sets))
	lookup_primary_result = None
	for lookup_result in lookup_results:
		if lookup_result and (not lookup_primary_result or lookup_primary_result == empty_result_mark):
			lookup_primary_result = lookup_result

	if VERBOSE:
		print(f"Lookup results = {lookup_results}")
		print(f"Lookup primary result = {lookup_primary_result}")
		print("="*50)

def lookup_lemmas(lemmas, slot_set):
	if VERBOSE:
		print(f"Input slot set: {slot_set}")
	found = []
	for device in lemmas:
		found_counter = 0
		found_not_null_counter = 0
		found_labels = []
		found_text = []
		for slot in lemmas[device]:
			cur_slot_value = ''
			if (slot['slot-names'] == system_slot_name):
				cur_slot_value = slot_set[0]
			elif (slot['slot-names'] == feature_slot_name):
				cur_slot_value = slot_set[1]
			elif (slot['slot-names'] == command_slot_name):
				cur_slot_value = slot_set[2]
			elif (slot['slot-names'] == param_slot_name):
				cur_slot_value = slot_set[3]

			# if ((slot['slot-names'] == system_slot_name) and slot_set[0] in slot['slot-values']) or\
			# 	((slot['slot-names'] == feature_slot_name) and slot_set[1] in slot['slot-values']) or\
			# 	((slot['slot-names'] == command_slot_name) and slot_set[2] in slot['slot-values']) or\
			# 	((slot['slot-names'] == param_slot_name) and slot_set[3] in slot['slot-values']):

			if cur_slot_value in slot['slot-values']:
				found_counter += weights.get(tuple(slot['slot-names']), 1.0)
				if cur_slot_value != '':
					found_not_null_counter += weights.get(tuple(slot['slot-names']), 1.0)#1
					found_text.append(cur_slot_value)
				found_labels.append(slot['slot-label'])
				
		if (found_counter > 0):
			slot_set_record = {'device': device, 'found': [i for i in found_labels if i], 'score': found_counter, 'not-null-score': found_not_null_counter, 'text': found_text}
			if 'on' not in slot_set_record['found'] and 'off' not in slot_set_record['found']:
				slot_set_record['found'].append('on')
			found.append(slot_set_record)
	found = sorted(found, key = lambda item: item['not-null-score'], reverse = True)
	if VERBOSE:
		print(f"Counter: {found_counter}; Labels: {found}")
	return found

def lookup_raw_lemmas(lemmas, hyp_lemmas):
	#print(f"Input slot set: {slot_set}")
	found = []
	for device in lemmas:
		found_counter = 0
		found_not_null_counter = 0
		found_labels = []
		found_lemmas = []
		for slot in lemmas[device]:
			for hyp_lemma in hyp_lemmas:
				if hyp_lemma in slot['slot-values']:
					found_counter += 1
					found_lemmas.append(hyp_lemma)
		if (found_counter > 0):
			found.append({'device': device, 'score': found_counter, 'found': found_lemmas})
	found = sorted(found, key = lambda item: item['score'], reverse = True)
	return found

def lookup_lemmas_post_handle(lemmas, slot_set):
	lookup_result = lookup_lemmas(lemmas, slot_set)
	if len(lookup_result) > 0:
		lookup_result = lookup_result[0]
		return {'labels': [lookup_result['device']] + lookup_result['found'], 'score': lookup_result['not-null-score'], 'text': lookup_result['text']}
	return {'labels': [], 'score': 0, 'text': []}

def get_labels(cmd, filename = lemmas_path, morph = pymorphy2.MorphAnalyzer()): #filename = "requests.json"):
	lemmas = read_json(filename)
	#print(f"Input command: {cmd}")
	system = get_slot_value_many(cmd, system_slot_name, morph)
	feature = get_slot_value_many(cmd, feature_slot_name, morph)
	command = get_slot_value_many(cmd, command_slot_name, morph)
	param = get_slot_value_many(cmd, param_slot_name, morph)

	slot_sets = np.array(np.meshgrid(system, feature, command, param)).T.reshape(-1,4)

	#print(slot_sets)
	if VERBOSE:
		print("Looking up for:")
		print(f"system = {system}\nfeature = {feature}\ncommand = {command}\nparam = {param}")
	
	lookup_results = list(map(lambda slot_set: lookup_lemmas_post_handle(lemmas, slot_set), slot_sets))
	#print(lookup_results)
	lookup_results = sorted(lookup_results, key = lambda i: i['score'], reverse = True)
	#print(f"Lookup primary result: {lookup_results[0]['labels']}")
	#print("="*50)
	#print(lookup_results[0])

	if VERBOSE:
		print(f"Lookup results = {lookup_results}")
		print("="*50)

	return lookup_results[0]#['labels']
	# lookup_primary_result = None
	# for lookup_result in lookup_results:
	# 	if lookup_result and (not lookup_primary_result or lookup_primary_result == empty_result_mark):
	# 		lookup_primary_result = lookup_result



def get_raw_labels(cmd, filename = lemmas_path, morph = pymorphy2.MorphAnalyzer()):
	lemmas = read_json(filename)
	cmd_deannotated = list(map(lambda pair: morph.parse(pair[0])[0].normal_form if USE_LEMMAS else pair[0], cmd))
	#print(f"Input command: {cmd_deannotated}")
	lookup_result = lookup_raw_lemmas(lemmas, cmd_deannotated)
	#print(f"Raw lookup result: {lookup_result}")
	lookup_primary_result = list(map(lambda item: {'device': item['device'], 'text': item['found'], 'score': item['score']}, lookup_result))
	return lookup_primary_result if len(lookup_primary_result) else [{'device': MOST_FREQUENT_LABEL, 'text': '', 'score': 0}]
	# system = get_slot_value_many(cmd, system_slot_name, morph)
	# feature = get_slot_value_many(cmd, feature_slot_name, morph)
	# command = get_slot_value_many(cmd, command_slot_name, morph)
	# param = get_slot_value_many(cmd, param_slot_name, morph)

	# slot_sets = np.array(np.meshgrid(system, feature, command, param)).T.reshape(-1,4)

	# #print(slot_sets)

	# print("Looking up for:")
	# print(f"system = {system}\nfeature = {feature}\ncommand = {command}\nparam = {param}")
	
	# lookup_results = list(map(lambda slot_set: lookup_lemmas_post_handle(lemmas, slot_set), slot_sets))
	# print(lookup_results)
	# lookup_results = sorted(lookup_results, key = lambda i: i['score'], reverse = True)
	# print(f"Lookup primary result: {lookup_results[0]['labels']}")
	# print("="*50)
	# return lookup_results[0]['labels']


if __name__ == '__main__':
	#get_request_type((("включай", "COMMAND"), ("кофеварку", "DEVICE"), ("быстрее", "DeviceFeature"), ("медленнее", "CommandFeature"), ("эту", "CommandParameter"), ("так", "DeviceParameter")))
	get_labels((("включай", "COMMAND"), ("кофеварку", "DEVICE"), ("быстрее", "DeviceFeature"), ("медленнее", "CommandFeature"), ("эту", "CommandParameter"), ("так", "DeviceParameter")))

