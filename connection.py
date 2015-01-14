import neuron
import random

class Connection(object):
	"""docstring for Connection"""
	def __init__(self, NEUR):
		self.weight = random.random()
		self.ally = NEUR