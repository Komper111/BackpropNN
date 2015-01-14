from deepNeuralNetwork import *
from backpropTeacher import *
import math

def readData(DOC):
	data = []
	dataDoc = open(DOC,'r')
	dataCount = int(dataDoc.readline())
	for i in xrange(dataCount):
		dataDoc.readline()
		x = float(dataDoc.readline())
		y = float(dataDoc.readline())
		# a = math.sin(x)
		# b = math.sin(y)
		# c = x*y
		color = int(dataDoc.readline()[2])
		data.append([[x, y], [color]])
	return data 

trainData = readData('circle_train.pat')#[[[1, 1], [0]]], [[3, 3], [3]]]
testData = readData('circle_test.pat')
# print trainData
# print testData
# raw_input("enter")

net = DeepNeuralNetwork([2, 10, 4, 1])
teacher = BackpropTeacher(net, .4, trainData, testData, 3)
err = teacher.onlyBackpropTrain(10000)
