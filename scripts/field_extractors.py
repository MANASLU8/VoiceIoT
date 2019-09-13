def extract_prop(annotation, property):
	if type(annotation[property]) == list:
		return annotation[property][0]
	else:
		return annotation[property]

def extract_ontology_label(annotation):
	return extract_prop(annotation, 'ontology-label')

def extract_text(annotation):
	return extract_prop(annotation, 'text')
