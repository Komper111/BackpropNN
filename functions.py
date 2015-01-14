from __future__ import division
import math

def input_function(CON, NEUR):
	o = 0
	for C in CON:
		o += C.ally.out * C.weight
	return o

def sigmoidal_function(X):
	return 1/(1 + math.exp(-(X*3)))

def derivative_sigmoidal_function(X):
	return 3*math.exp(3*X)/pow(math.exp(3*X)+1,2)
	#return (1/(1 + math.exp(X)))-(1/pow(1 + math.exp(X), 2))
