import sys, os
sys.path.append('..')

import tensorflow as tf
import pandas as pd
from keras.callbacks import ModelCheckpoint

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession, Session

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.mplot3d import Axes3D

class LogisticRegressionTF(tf.keras.Model):
	def __init__(self, num_classes, batch_size, epochs, callbacks):
		super(LogisticRegressionTF, self).__init__()
		self.dense = tf.keras.layers.Dense(num_classes)
		self.batch_size = batch_size
		self.epochs = epochs
		self.callbacks = callbacks
	
	def call(self, inputs, training=None, mask=None):
		output = self.dense(inputs)
		output = tf.nn.softmax(output)
		return output

	def fit_new(self, x, y):
		return self.fit(x, y, batch_size=self.batch_size, epochs=self.epochs, validation_data=(x, y), callbacks=self.callbacks, verbose=2)

	def predict_new(self, x):
		return self.predict(x)
