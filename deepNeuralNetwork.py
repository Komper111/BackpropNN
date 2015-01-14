from __future__ import division
from layer import *
from neuron import *
import math
import png

class DeepNeuralNetwork:
	def __init__(self, NEURONS, TYPE = "mainNet"):
		self.layers = []
		self.bias = Neuron()
		self.type = TYPE
		self.layers.append(InputLayer(NEURONS[0]))
		for i in range(1,len(NEURONS)):
			self.layers.append(HiddenLayer(self.layers[i-1], NEURONS[i], self.bias))
		
		
	def activate(self, INPUTS):
		# print "Neural network ", self.type, "activated with : ", INPUTS
		output = []
		self.layers[0].activate(INPUTS)
		for i in range(1, len(self.layers)):
			self.layers[i].activate()
		for N in self.layers[len(self.layers)-1].neurons:
			output.append(N.out)
		return output

	def toPNG(self, ITER):
		string = ""
		M = [[]]
		for y in xrange(0,100):
			for x in xrange(0,100):
				SA = self.activate([x/100, y/100])#, math.sin(x), math.sin(y), x*y])
				M[y].append(SA[0]*255)
			M.append([])
		f = open('./png/'+str(ITER)+'.png', 'wb')      # binary mode is important
		w = png.Writer(100, 101, greyscale=True)
		w.write(f, M)
		f.close()