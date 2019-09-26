import argparse, os
from gensim.models import KeyedVectors
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

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

COMMON_COLOR = "#051e3e"
BRIGHT_COLOR = "#851e3e"

DEFAULT_MODEL_FILE_NAME = "test.txt"

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', dest='model', help='path to the trained model')
parser.add_argument('--no-cache', dest='nocache', action='store_true')
utils.add_config_arg(parser)
args = parser.parse_args()
config = utils.load_config(args.config)

pickle_path = os.path.join(config['paths']['datasets']['w2v']['root'], (DEFAULT_MODEL_FILE_NAME if args.model is None else args.model).split("/")[-1].split('.')[0] + ".pkl")
if os.path.isfile(pickle_path) and not args.nocache:
	df = pd.read_pickle(pickle_path)
else:
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
	df.to_pickle(pickle_path)

#
# classify using logistic regression classifier
#

def split(df, pieces=2):
	number_of_lines = df.shape[0]
	chunk_size = number_of_lines // pieces
	for i in range(pieces):
		yield {"test": df[i * chunk_size: (i + 1) * chunk_size], "train": pd.concat([df[: i * chunk_size], df[(i + 1) * chunk_size:]])}

def classify(df, classifier, pieces=config['cv-folds']):
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

	return {"accuracy": np.average([item['accuracy'] for item in results])*100, "f1-score": np.average([item['f1-score'] for item in results])*100}


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

#
# Sort
#

sorted_indices = np.argsort(accuracy)
classifiers = np.array(classifiers)[sorted_indices]
accuracy = np.array(accuracy)[sorted_indices]
f1_score = np.array(f1_score)[sorted_indices]

#
# Print
#

rdf = pd.DataFrame.from_dict({"classifier": classifiers, "accuracy": accuracy, "f1-score": f1_score})
print(rdf)
print(f"Best classifier according to accuracy ({max_accuracy}): {max_accuracy_classifier}")
print(f"Best classifier according to f1-score ({max_f1_score}): {max_f1_score_classifier}")
print(f"Constant classifier gives {np.max(df.groupby('type').size())/(np.sum(df.groupby('type').size()))}: {' + '.join([str(item) for item in df.groupby('type').size().to_numpy()])} = {np.sum(df.groupby('type').size())}")

#
# Draw graph 
#

def draw_bar_labels(x, y, axis, threshold=2, round_precision=2):
	smallest = sorted(y)[threshold]
	for c, v in zip(x, y):
		axis.text(c, v * 0.97 if v > smallest else v + 0.5, str(round(v, 2)), color='white' if v > smallest else 'black', horizontalalignment='center')

# create two sublots
fig, axs = plt.subplots(2)
plt.subplots_adjust(bottom=0.2, top=0.9, hspace=1.2)

fig.set_figwidth(19)
fig.set_figheight(10)
fig.suptitle("Results of commands' labelling", fontsize=16)

axs[0].bar(rdf.classifier, height=rdf.accuracy, color=[COMMON_COLOR if i < max_accuracy else BRIGHT_COLOR for i in accuracy], log = True)
axs[0].set_title('Accuracy')
axs[0].set_xlabel('Classifier')
axs[0].set_ylabel('Accuracy (%)')
const_classifier_accuracy = 100*np.max(df.groupby('type').size())/(np.sum(df.groupby('type').size()))
axs[0].axhline(y=const_classifier_accuracy, color="black")
axs[0].text(classifiers[0], const_classifier_accuracy + 0.3, str(round(const_classifier_accuracy, 2)) + ' '*10, color='black', horizontalalignment='right')
axs[0].yaxis.set_major_formatter(ScalarFormatter())
axs[0].yaxis.set_minor_formatter(ScalarFormatter())
for tick in axs[0].get_xticklabels():
     tick.set_rotation(45)
draw_bar_labels(classifiers, rdf['accuracy'].to_numpy(), axs[0])

axs[1].bar(rdf.classifier, height=rdf['f1-score'], color=[COMMON_COLOR if i < max_f1_score else BRIGHT_COLOR for i in f1_score], log = True)
axs[1].set_title('F1-score')
axs[1].set_xlabel('Classifier')
axs[1].set_ylabel('F1-score (%)')
axs[1].yaxis.set_major_formatter(ScalarFormatter())
axs[1].yaxis.set_minor_formatter(ScalarFormatter())
for tick in axs[1].get_xticklabels():
     tick.set_rotation(45)
draw_bar_labels(classifiers, rdf['f1-score'].to_numpy(), axs[1])

fig.savefig(os.path.join(config['paths']['images']['w2v']['root'], 'cv.png'))