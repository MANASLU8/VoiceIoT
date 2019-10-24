import json, pymorphy2
import numpy as np
import collections

from .. import utils
from . import collect_lemmas as cl, use_lemmas as ul

if __name__ == "__main__":
	config = utils.load_config(utils.parse_args().config)

system_slot_name = ["DEVICE", "CommandFeature", "DeviceFeature", "ParameterFeature", "Devicefeature"]
command_slot_name = "COMMAND"
feature_slot_name = ["CommandFeature", "DeviceFeature", "ParameterFeature", "Devicefeature"]
param_slot_name = ["CommandParameter", "DeviceParameter", "DeviceParameterValue", "CommandParameterValue"]

slot_name_mapping = {'system': system_slot_name, 'feature': feature_slot_name, 'commands-on': command_slot_name, 'commands-off': command_slot_name, 'param': param_slot_name}
slot_label_mapping = {'system': None, 'feature': None, 'commands-on': 'on', 'commands-off': 'off', 'param': None}

requests_path = config['paths']['datasets']['request-mapping']['requests'] if __name__ == "__main__" else "requests.json"

empty_result_mark = '<REQUEST_TYPE>'

lemmas = ul.read_json(config['paths']['datasets']['request-mapping']['lemmas'])



on_commands = ['', 'врубить', 'устанавливать', 'погреть', 'подсветить', 'проветрить', 'постирать', 'испечь', 'поставить', 'свари', 'разбудить',
				'распечатать', 'приготовить', 'разогреть', 'сварить',  'включать', 'установи', 'вруби', 'разбуди', 'поставь', 'проветри', 'распечатай']
off_commands = ['выключать', 'закрыть', 'оффнуть', 'остановить', 'прекратить', 'вырубить', 'прекрати', 'оффни', 'выруби', 'закрой', 'останови']

commands = []
for device in lemmas:
	lemmas[device]['commands-on'] = [command for command in lemmas[device]['command'] if command in on_commands]
	lemmas[device]['commands-off'] = [command for command in lemmas[device]['command'] if command in off_commands]
	del lemmas[device]['command']

new_lemmas = {}

for device in lemmas:
	new_lemmas[device] = []
	for slot in lemmas[device]:
		new_lemmas[device].append({'slot-names': slot_name_mapping[slot], 'slot-label': slot_label_mapping[slot], 'slot-values': lemmas[device][slot]})

cl.write_json(config['paths']['datasets']['request-mapping']['lemmas-fine-splitted'], new_lemmas)

#if __name__ == '__main__':
#	get_request_type((("включай", "COMMAND"), ("кофеварку", "DEVICE"), ("быстрее", "DeviceFeature"), ("медленнее", "CommandFeature"), ("эту", "CommandParameter"), ("так", "DeviceParameter")))

