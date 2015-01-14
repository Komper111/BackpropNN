from neuron import *
from functions import *
from connection import *

class Layer(object):
	"""docstring for Layer"""
	def __init__(self):
		self.neurons = []

	def activate(self, INPUTS, HIDDL = False):
		i = 0
		out = []
		if not HIDDL:
			for N in self.neurons:
				N.activate(INPUTS[i])
				i += 1
		else:
			for N in self.neurons:
				N.out = sigmoidal_function(INPUTS[i])
				# print "Neuron "+ str(N.name) + ": "+ str(N.out)
				i += 1

class InputLayer(Layer):
	"""docstring for InputLayer"""
	def __init__(self, NN):
		super(InputLayer, self).__init__()
		for i in range(NN):
			self.neurons.append(InputNeuron(sigmoidal_function))		
	
class HiddenLayer(Layer):
	"""docstring for HiddenLayer"""
	def __init__(self, BL, NN, BIAS):
		super(HiddenLayer, self).__init__()
		for i in range(NN):
			connections = []
			for N in BL.neurons:
				connections.append(Connection(N))
			connections.append(Connection(BIAS))
			self.neurons.append(HiddenNeuron(input_function, sigmoidal_function, connections))

	def activate(self, INPUTS = None):
		if INPUTS == None:
			for N in self.neurons:
				N.activate()
		else:
			super(HiddenLayer, self).activate(INPUTS, True)

	def activateAsInputLayer(self):
		pass