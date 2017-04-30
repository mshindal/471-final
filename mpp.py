#!/usr/bin/env python3

from pdb import set_trace as d

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

def classify(data):
	d()
	print(data)