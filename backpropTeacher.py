from deepNeuralNetwork import *
from functions import *

class BackpropTeacher(object):
	"""docstring for BackpropTeacher"""
	def __init__(self, NN, LPAR, TRAIND, TESTD,NOFL):
		self.net = NN #neural network
		self.LP = LPAR #learning parameter
		self.trainData = TRAIND#train data set
		self.testData = TESTD
		self.NOFL = NOFL
		self.nets = []
		for i in range(NOFL):
			self.nets.append(DeepNeuralNetwork([1,1,1], "miniNet"+str(i)))
			self.nets[i].layers[0] = self.net.layers[i]
			self.nets[i].layers[1] = self.net.layers[i+1]
			self.nets[i].layers[2] = HiddenLayer(self.nets[i].layers[1], len(self.nets[i].layers[0].neurons), self.net.bias)
	
	def onlyBackpropTrain(self, EPOCHS = 1):
		outputF = open('out.txt','a')
		netErrors = []
		for i in xrange(EPOCHS):
			print "Epoch : ", i
			for D in self.trainData:
				netErrors.append(self.train(self.net, D))
			if i % 30 == 0:
				if i != 0:
					avgErr = self.av(i-30*len(self.trainData), i*len(self.trainData), netErrors)
					if avgErr < 0.12: 
						self.LP = 0.3
					if avgErr < 0.1:
						self.LP = 0.2	
					if avgErr < 0.07:
						self.LP = 0.1
					outputF.write("Epoch " + str(i)+ "\n")
					self.net.toPNG(i)
					outputF.write("error: "+str(netErrors[len(netErrors)-1])+"\n")
					test = self.testNet()
					outputF.write("Test: "+str(test[0])+" of "+ str(test[1]) + "\n")
					outputF.write("LP: "+str(self.LP)+"\n")
					outputF.write("AVG E: "+str(avgErr)+"\n \n")
		return netErrors
	
	def train(self, NET, DATA):
		NET.activate(DATA[0])
		error = self.calculateOutputError(NET, DATA[1]) * .5
		for k in range(len(NET.layers)-2, 0, -1):
			self.errorBackpropagation(NET, k)				
		self.calculateNewWeights(NET)
		return error

	def errorBackpropagation(self, NET, LAYER):
		for N in NET.layers[LAYER].neurons:
			N.errSig = 0
			for NE in NET.layers[LAYER+1].neurons:
				#print "from ", NE.name, ":", NE.errSig, NE.getWeight(N),"to Neuron ", N.name, "errSig = ", N.errSig
				N.errSig += NE.errSig * NE.getWeight(N) 
			N.errSig = N.errSig * derivative_sigmoidal_function(N.getInput())
			# print "Neuron ",N.name, ": errSig = " ,N.errSig

	def calculateOutputError(self, NET, EOUT):
		netError = 0
		k = 0
		for N in NET.layers[len(NET.layers)-1].neurons:
			N.errSig = (EOUT[k] - N.out) * derivative_sigmoidal_function(N.getInput())
			netError += pow(EOUT[k] - N.out, 2)
			# print "Neuron "+ str(N.name) + ": error Signal - "+ str(N.errSig)
			k += 1

			netError = netError/len(NET.layers[len(NET.layers)-1].neurons)

		return netError

	def calculateNewWeights(self, NET):
		for i in range(1, len(NET.layers)):
			for N in NET.layers[i].neurons:
				for C in N.connections:
					C.weight += self.LP * N.errSig * C.ally.out
					# print "Connection - ",N.name, C.ally.name," weight- ",C.weight
	def testNet(self):
		good = 0
		_all = 0
		for D in self.testData:
			o = self.net.activate(D[0])
			for i in xrange(len(D[1])):
				if abs(D[1][i] - o[i]) < .5:
					good += 1
		return [good, len(self.testData)]

	def av(self, f, to, arr):
		su = 0
		for i in range(f,to):
			su += arr[i]
		su = su / (to-f)
		return su

	def pretrainAndTrain(self, EPOCHS = 1):
		netErrors = []
		miniNetErrors = self.preTrain(3)
		netErrors = self.onlyBackpropTrain(EPOCHS)
		return [netErrors, miniNetErrors]

	def preTrain(self, EPOCHS = 1):
		netErrors = []
		for i in range(self.NOFL):
			netErrors.append([])
		prelearned = False  
		for i in range(EPOCHS):
			for D in self.trainData:
				_input = D[0]
				j = 0
				for NET in self.nets:
					netErrors[j].append(self.train(NET, [_input,_input]))
					_input = []
					for N in NET.layers[1].neurons:
						_input.append(N.out)
					j += 1
		return netErrors