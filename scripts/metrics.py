ACCURACY = 4

def get_precision(reference, hypothesis):
	print(reference)
	print(hypothesis)
	return round(len([1 for item in hypothesis if item in reference]) / len(reference), ACCURACY) if len(reference) > 0 else 0

def get_recall(reference, hypothesis):
	return round(len([1 for item in hypothesis if item in reference]) / len(hypothesis), ACCURACY) if len(hypothesis) > 0 else 0

def get_f1_score(reference, hypothesis):
	precision = get_precision(reference, hypothesis)
	recall = get_recall(reference, hypothesis)
	return round(2 * precision * recall / (precision + recall), ACCURACY) if precision + recall > 0 else 0