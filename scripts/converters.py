def decode_bio(bio):
	dicti = {}
	for pair in bio:
		entity_name = pair[1].split("-", maxsplit = 1)[1]
		if entity_name in dicti:
			dicti[entity_name].append(pair[0])
		else:
			dicti[entity_name] = [pair[0]]
	return dicti
