import json, pymorphy2
import numpy as np
from .. import utils
import collections

if __name__ == "__main__":
	config = utils.load_config(utils.parse_args().config)

system_slot_name = "DEVICE"
command_slot_name = "COMMAND"
feature_slot_name = ["CommandFeature", "DeviceFeature", "ParameterFeature", "Devicefeature"]
param_slot_name = ["CommandParameter", "DeviceParameter", "DeviceParameterValue", "CommandParameterValue"]

requests_path = config['paths']['datasets']['request-mapping']['requests'] if __name__ == "__main__" else "requests.json"

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
	return morph.parse(selected_slot_value)[0].normal_form

def get_slot_value_many(command, slot_name, morph):
	slot_values = collections.defaultdict(list)
	for pair in command:
		if are_slot_names_equal(pair[1], slot_name): #and slot_value == '':
			slot_values[pair[1]].append(pair[0])
	return list(set([morph.parse(slot_value)[0].normal_form for slot_name in slot_values for slot_value in slot_values[slot_name]] + ['']))
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
	print(f"Input command: {cmd}")
	system = get_slot_value_many(cmd, system_slot_name, morph)
	feature = get_slot_value_many(cmd, feature_slot_name, morph)
	command = get_slot_value_many(cmd, command_slot_name, morph)
	param = get_slot_value_many(cmd, param_slot_name, morph)

	slot_sets = np.array(np.meshgrid(system, feature, command, param)).T.reshape(-1,4)

	#print(slot_sets)

	print("Looking up for:")
	print(f"system = {system}\nfeature = {feature}\ncommand = {command}\nparam = {param}")

	
	lookup_results = list(map(lambda slot_set: lookup(requests, slot_set), slot_sets))
	lookup_primary_result = None
	for lookup_result in lookup_results:
		if lookup_result and (not lookup_primary_result or lookup_primary_result == empty_result_mark):
			lookup_primary_result = lookup_result

	print(f"Lookup results = {lookup_results}")
	print(f"Lookup primary result = {lookup_primary_result}")
	print("="*50)


if __name__ == '__main__':
	get_request_type((("включай", "COMMAND"), ("кофеварку", "DEVICE"), ("быстрее", "DeviceFeature"), ("медленнее", "CommandFeature"), ("эту", "CommandParameter"), ("так", "DeviceParameter")))

