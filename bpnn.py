#!/usr/bin/env python3

from sklearn.neural_network import MLPClassifier
import numpy as np
from pdb import set_trace as d

def classify(testing_set, training_set, classifier_tuple):
	num_classes = classifier_tuple[0]
	testing_set = np.array(testing_set)
	training_set = np.array(training_set)
	training_labels = training_set[:, -1]
	training_set = training_set[:, :-1]
	clf = MLPClassifier(activation='logistic', solver='sgd', learning_rate='constant', learning_rate_init=.001)
	clf.fit(training_set, training_labels)
	num_correct = 0
	confusion_matrix = np.zeros((num_classes, num_classes))
	for sample in testing_set:
		predicted_label = clf.predict(sample[:-1].reshape(1, -1)).item()
		actual_label = sample[-1].item()
		if predicted_label == actual_label:
			num_correct += 1
		confusion_matrix[int(actual_label)][int(predicted_label)] += 1
	accuracy = num_correct / testing_set.shape[0]
	return confusion_matrix, accuracy