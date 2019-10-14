import json, pymorphy2
system_slot_name = "DEVICE"
command_slot_name = "COMMAND"
feature_slot_name = "CommandFeature"
param_slot_name = "CommandParameter"

requests_path = "requests.json"

def read_json(filename):
	with open(filename) as f:
		return json.load(f)

def get_slot_value(command, slot_name, morph):
	slot_value = ''
	for pair in command:
		if pair[1] == slot_name and slot_value == '':
			slot_value = pair[0]
	return morph.parse(slot_value)[0].normal_form

def lookup(requests, path):
	node = requests
	for key in path:
		if key in node:
			node = node[key]
		else:
			return None
	return node

def get_request_type(cmd, filename = requests_path, morph = pymorphy2.MorphAnalyzer()): #filename = "requests.json"):
	#
	requests = read_json(filename)
	print(f"Input command: {cmd}")
	system = get_slot_value(cmd, system_slot_name, morph)
	feature = get_slot_value(cmd, feature_slot_name, morph)
	command = get_slot_value(cmd, command_slot_name, morph)
	param = get_slot_value(cmd, param_slot_name, morph)
	print("Looking up for:")
	print(f"system = {system}\nfeature = {feature}\ncommand = {command}\nparam = {param}")
	print(f"Lookup result = {lookup(requests, (system, feature, command, param))}")
	print("="*50)


if __name__ == '__main__':
	get_request_type((("включай", "COMMAND"), ("кофеварку", "DEVICE")))

