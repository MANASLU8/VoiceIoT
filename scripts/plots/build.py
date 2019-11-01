import argparse, os
from gensim.models import KeyedVectors
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter

from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

from .. import utils
#from . import normalize

COMMON_COLOR = "#051e3e"
BRIGHT_COLOR = "#851e3e"

plt.rcParams.update({'font.size': 22})

def draw_bar_labels(x, y, axis, zero_offset = 4.1, threshold=2, round_precision=2):
		smallest = sorted(y)[threshold]
		for c, v in zip(x, y):
			axis.text(c, v - 4.5 if v > smallest and v > 0 else v + 0.5 if v > 0 else v + zero_offset, str(round(v, 2)), color='white' if v > smallest else 'black', horizontalalignment='center')


fig, axs = plt.subplots(1)
plt.subplots_adjust(bottom=0.2, top=0.9, hspace=0.65, left=0.05, right=0.95)

fig.set_figwidth(19)
fig.set_figheight(15)
#fig.suptitle("Accuracy of recognition", fontsize=16)

type_counts = {
	"AudioSystem": 35,
	"WindowRelay": 14,
	"WashingMachine": 8,
	"Printer": 10,
	"HeatingFloor": 3,
	"SmartFridge": 5,
	"CurtainsRelay": 5,
	"AlarmClock": 5,
	"CoffeeMachine": 10,
	"SmartBreadMaker": 5,
	"SmartMicrowave": 5,
	"Lamp": 5,
	"FloorLite": 2
}

device_accuracy = [
	["lemmas", 67.86],
	["labels", 42.86],
	["system", 83.04],
	["snips-nlu (lemmas)", 42.86],
	["snips-nlu (labels)", 20.54],
	["pytext (labels)", 8.04],
	["pytext (lemmas)", 15.18],
	["pytext", 4.545],
	["snips-nlu", 33.786]
]

command_accuracy = [
	["lemmas", 65.18, 67.86],
	["labels", 39.29, 42.86],
	["system", 81.25, 83.04],
	["snips-nlu (lemmas)", 35.71, 42.86],
	["snips-nlu (labels)", 19.64, 20.54],
	["pytext (labels)", 8.04, 8.04],
	["pytext (lemmas)", 14.29, 15.18]
]

#print(device_accuracy)

sorted_indices = np.argsort([item[1] for item in device_accuracy])
#classifiers = np.array(classifiers)[sorted_indices]
#accuracy = np.array(accuracy)[sorted_indices]
#f1_score = np.array(f1_score)[sorted_indices]
device_accuracy = [[item[0], float(item[1])] for item in np.array(device_accuracy)[sorted_indices]]
sorted_indices = np.argsort([item[1] for item in command_accuracy])
command_accuracy = [[item[0], float(item[1])] for item in np.array(command_accuracy)[sorted_indices]]
#command_accuracy = np.array(command_accuracy)[sorted_indices]
#print(device_accuracy)
max_accuracy = np.max(np.array([item[1] for item in device_accuracy]))
max_command_accuracy = np.max(np.array([item[1] for item in command_accuracy]))

type_counts['Total'] = int(np.sum([int(item) for item in type_counts.values()]))
#axs.transAxes, family='monospace')
axs.bar([item[0] for item in device_accuracy], height=[item[1] for item in device_accuracy], color=[COMMON_COLOR if i < max_accuracy else BRIGHT_COLOR for i in [item[1] for item in device_accuracy]])
#axs.set_title('Device recognition accuracy')
axs.set_xlabel('Метод')
axs.set_ylabel('Точность')
const_classifier_accuracy = 100*np.max([item[1] for item in device_accuracy])/(np.sum([item[1] for item in device_accuracy]))
# draw horizontal line
# axs.axhline(y=const_classifier_accuracy, color="black")
#axs.text(-0.5, const_classifier_accuracy + 0.3, str(round(const_classifier_accuracy, 2)), color='black', horizontalalignment='right')
axs.yaxis.set_major_formatter(FormatStrFormatter('%i'))
#axs.yaxis.set_minor_formatter(ScalarFormatter())

for tick in axs.get_xticklabels():
     tick.set_rotation(45)
draw_bar_labels([item[0] for item in device_accuracy], [item[1] for item in device_accuracy], axs)
fig.savefig('images/recognition/device.png')
axs.clear()
axs.bar([item[0] for item in command_accuracy], height=[item[1] for item in command_accuracy], color=[COMMON_COLOR if i < max_command_accuracy else BRIGHT_COLOR for i in [item[1] for item in command_accuracy]])
#axs.set_title('Command recognition accuracy')
axs.set_xlabel('Метод')
axs.set_ylabel('Точность')
#axs.yaxis.set_major_formatter(FormatStrFormatter('%i'))
#axs.yaxis.set_minor_formatter(ScalarFormatter())
for tick in axs.get_xticklabels():
     tick.set_rotation(45)
draw_bar_labels([item[0] for item in command_accuracy], [item[1] for item in command_accuracy], axs, 7.5)
fig.savefig('images/recognition/command.png')
