import os
from .. import file_operators as fo

TMP_OUT_FILE = 'cv/out.json'

i = 0
collected = []
while (os.path.isfile(TMP_OUT_FILE.split('.')[0]+f'_{i}.'+TMP_OUT_FILE.split('.')[1])):
	filename = TMP_OUT_FILE.split('.')[0]+f'_{i}.'+TMP_OUT_FILE.split('.')[1]
	for item in fo.read_json(filename):
		collected.append(item)
	i += 1
fo.write_json(TMP_OUT_FILE, collected)