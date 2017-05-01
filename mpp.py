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
	return -.5 * (x - mean).T.dot(np.linalg.inv(cov)).dot(x - mean) + math.log(prior)

# discriminant function case 3
def disc3(x, mean, cov, prior):
	Wi = -.5 * np.linalg.inv(cov)
	wi = np.linalg.inv(cov).dot(mean)
	wi0 = -.5 * mean.T.dot(np.linalg.inv(cov)).dot(mean) - .5 * math.log(np.linalg.det(cov)) + math.log(prior)
	return x.T.dot(Wi).dot(x) + wi.T.dot(x) + wi0

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

def calc_covariances(training_set):
	seperated_sets = collections.defaultdict(lambda: [])
	for row in training_set:
		label = row[-1]
		seperated_sets[label].append(row[:-1])
	return {k: np.cov(v, rowvar=False) for k, v in seperated_sets.items()}

def classify(testing_set, training_set, classifier_tuple, discriminant_function):
	num_classes = classifier_tuple[0]
	testing_set = np.array(testing_set)
	training_set = np.array(training_set)
	priors = calc_priors(training_set)
	means = calc_means(training_set)
	if discriminant_function != disc1:
		covs = calc_covariances(training_set)
	num_correct = 0
	confusion_matrix = np.zeros((num_classes, num_classes))
	for sample in testing_set:
		discriminants = []
		for c in range(num_classes):
			if discriminant_function == disc1:
				discriminants.append(discriminant_function(sample[:-1], means[c], priors[c]))
			else:
				if c == 0:
					discriminants.append(-math.inf)
				else:
					discriminants.append(discriminant_function(sample[:-1], means[c], covs[c], priors[c]))
		detected_class = np.argmax(discriminants)
		actual_class = sample[-1]
		if detected_class == actual_class:
			num_correct += 1
		confusion_matrix[actual_class][detected_class] += 1
	accuracy = num_correct / testing_set.shape[0]
	return confusion_matrix, accuracy

def classify_case_1(testing_set, training_set, classifier_tuple):
	return classify(testing_set, training_set, classifier_tuple, disc1)

def classify_case_2(testing_set, training_set, classifier_tuple):
	return classify(testing_set, training_set, classifier_tuple, disc2)

def classify_case_3(testing_set, training_set, classifier_tuple):
	return classify(testing_set, training_set, classifier_tuple, disc3)