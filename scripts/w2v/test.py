import argparse, os
from gensim.models import KeyedVectors
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

from .. import utils
from . import normalize

DEFAULT_MODEL_FILE_NAME = "test.txt"

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', dest='model', help='path to the trained model')
utils.add_config_arg(parser)
args = parser.parse_args()
config = utils.load_config(args.config)

#
# read annotations dataset
#

df = normalize.handle_files(utils.list_dataset_files(config['paths']['datasets']['annotations']))
#
# load model
#

model = KeyedVectors.load_word2vec_format(os.path.join(config['paths']['models']['w2v'], DEFAULT_MODEL_FILE_NAME) if args.model is None else args.model, binary=False)

# map given results with input data
df.text = list(map(lambda text: np.average([[value for value in model[word]] for word in text.split(" ") if word in model], axis=0), df.text))
df = df[df.text.notnull()]

# save transformed csv
df.to_csv(os.path.join(config['paths']['datasets']['w2v']['root'], (DEFAULT_MODEL_FILE_NAME if args.model is None else args.model).split("/")[-1].split('.')[0] + ".csv"))

#
# classify using logistic regression classifier
#

def split(df, pieces=2):
	number_of_lines = df.shape[0]
	chunk_size = number_of_lines // pieces
	for i in range(pieces):
		yield {"test": df[i * chunk_size: (i + 1) * chunk_size], "train": pd.concat([df[: i * chunk_size], df[(i + 1) * chunk_size:]])}

def classify(df, classifier, pieces=5):
	results = []
	for item in split(df, pieces=pieces):
		x_train =  np.array([[value for value in item] for item in item['train'].text.to_numpy()])
		x_test = np.array([[value for value in item] for item in item['test'].text.to_numpy()])
		y_train = item['train'].type.to_numpy()
		y_test = item['test'].type.to_numpy()

		#print(f"X train = {x_train}")
		#print(f"Y train = {y_train}")

		classifier.fit(x_train, y_train)
		y_pred = classifier.predict(x_test)

		results.append({"accuracy": accuracy_score(y_test, y_pred), "f1-score": f1_score(y_test, y_pred, average='weighted')})

	print(f"Average accuracy: {np.average([item['accuracy'] for item in results])} (constant classifier gives {np.max(df.groupby('type').size())/(np.sum(df.groupby('type').size()))}: {df.groupby('type').size().to_numpy()})")
	print(f"Average f1-score: {np.average([item['f1-score'] for item in results])}")

	return {"accuracy": np.average([item['accuracy'] for item in results]), "f1-score": np.average([item['f1-score'] for item in results])}


classifiers = {
	"SVM": SVC(gamma='scale', decision_function_shape='ovo'),
	"Logistic regression (C = 1e9)": LogisticRegression(n_jobs=1, C=1e9),
	"Logistic regression (C = 1e12)": LogisticRegression(n_jobs=1, C=1e12),
	"KNN (3)": KNeighborsClassifier(3),
	"KNN (5)": KNeighborsClassifier(5),
	"KNN (7)": KNeighborsClassifier(7),
    "Gaussian process": GaussianProcessClassifier(1.0 * RBF(1.0)),
    "Decision tree (max depth = 3)": DecisionTreeClassifier(max_depth=3),
    "Decision tree (max depth = 5)": DecisionTreeClassifier(max_depth=5),
    "Decision tree (max depth = 7)": DecisionTreeClassifier(max_depth=7),
    "Random forest": RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    "MLP": MLPClassifier(alpha=1, max_iter=1000),
    "AdaBoost": AdaBoostClassifier(),
    "Gaussian": GaussianNB(),
    "QDA": QuadraticDiscriminantAnalysis()
}

results = {classifier: classify(df, classifiers[classifier]) for classifier in classifiers.keys()}

classifiers = []
accuracy = []
f1_score = []

# reformat
max_accuracy = -1
max_accuracy_classifier = ""
max_f1_score = -1
max_f1_score_classifier = ""
for classifier in results.keys():
	classifiers.append(classifier)
	accuracy.append(results[classifier]["accuracy"])
	if (results[classifier]["accuracy"] > max_accuracy):
		max_accuracy_classifier = classifier
		max_accuracy = results[classifier]["accuracy"]
	f1_score.append(results[classifier]["f1-score"])
	if (results[classifier]["f1-score"] > max_f1_score):
		max_f1_score_classifier = classifier
		max_f1_score = results[classifier]["f1-score"]

rdf = pd.DataFrame.from_dict({"classifier": classifiers, "accuracy": accuracy, "f1-score": f1_score})
print(rdf)
print(f"Best classifier according to accuracy: {max_accuracy_classifier}")
print(f"Best classifier according to f1-score: {max_f1_score_classifier}")
#print(results)