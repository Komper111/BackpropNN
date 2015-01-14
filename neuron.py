from itertools import count
class Neuron(object):
	ids = count(0)
	def __init__(self):
		self.name = self.ids.next()
		self.out = -1

	def activate(self, INPUT):
		self.out = self.actFunc(INPUT)
		# print "Neuron " + str(self.name) + ": "+ str(self.out)	

class InputNeuron(Neuron):
	def __init__(self, ACTF):
		super(InputNeuron, self).__init__()
		self.actFunc = ACTF #activation function	

class HiddenNeuron(Neuron):
	def __init__(self, INF, ACTF, CON):
		super(HiddenNeuron, self).__init__()
		self.inFunc = INF #input function
		self.actFunc = ACTF #activation function
		self.connections = CON #connected neurons with weights
		self.errSig = 0 #error signal
	
	def activate(self):
		self.out = self.actFunc(self.getInput())
		# print "Neuron "+ str(self.name) + ": "+ str(self.out)

	def getInput(self):
		return self.inFunc(self.connections, self)
	
	def getWeight(self, N):
		for C in self.connections:
			if C.ally == N:
				return C.weight
		return None