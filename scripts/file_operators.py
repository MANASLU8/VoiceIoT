import json

def write_lines(filename, lines):
	with open(filename, "w") as f:
		f.write('\n'.join(lines))

def write_json(filename, content):
	with open(filename, "w") as f:
		f.write(json.dumps(content, indent=2).encode().decode('unicode-escape'))

def read_lines(filename):
	with open(filename) as f:
		return [line.replace('\n', '') for line in f.readlines()]

def read_json(filename):
	with open(filename) as f:
		return json.load(f, strict=False)
			#''.join(f.readlines()).encode().decode('unicode-escape'))
