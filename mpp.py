#!/usr/bin/env python3

from pdb import set_trace as d
import numpy as np
import collections
import math

# discriminant function case 1
def disc1(x, mean, prior):
	return (-1 * np.power(np.linalg.norm(x - mean), 2)) + math.log(prior)

# discriminant function case 2
def disc2(x, mean, cov, prior):
	z = (-.5 * (x - mean) * np.linalg.inv(cov) * (x - mean).T) + math.log(prior)
	return z.item()

# discriminant function case 3
def disc3(x, mean, cov, prior):
	Wi = -.5 * np.linalg.inv(cov)
	wi = np.linalg.inv(cov) * mean.T
	wi0 = (-.5 * mean * np.linalg.inv(cov) * mean.T) - (.5 * math.log(np.linalg.det(cov))) + math.log(prior)
	z = (x * Wi * x.T) + (wi.T * x.T) + wi0
	return z.item()

def calc_priors(training_set):
	total_count = training_set.shape[0]
	label_count = collections.defaultdict(lambda: 0)
	for row in training_set:
		label = row[-1]
		label_count[label] += 1
	return {k: v / total_count for k, v in label_count.items()}

def calc_means(training_set):
	seperated_sets = collections.defaultdict(lambda: [])
	for row in training_set:
		label = row[-1]
		seperated_sets[label].append(row)
	return {k: np.mean(v, axis=0)[:-1] for k, v in seperated_sets.items()}

def classify_case_1(testing_set, training_set, classifier_tuple):
	num_classes = classifier_tuple[0]
	testing_set = np.array(testing_set)
	training_set = np.array(training_set)
	priors = calc_priors(training_set)
	means = calc_means(training_set)
	num_correct = 0
	confusion_matrix = np.zeros((num_classes, num_classes))
	for sample in testing_set:
		discriminants = []
		for c in range(num_classes):
			discriminants.append(disc1(sample[:-1], means[c], priors[c]))
		detected_class = np.argmax(discriminants)
		actual_class = sample[-1]
		if detected_class == actual_class:
			num_correct += 1
		confusion_matrix[actual_class][detected_class] += 1
	accuracy = num_correct / testing_set.shape[0]
	return confusion_matrix, accuracy


	