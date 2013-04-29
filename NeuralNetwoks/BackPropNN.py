#
#Backpropagation Neural network using Numpy
#
import numpy as np

class BackPropagationNetwork:
	#
	#Class members
	#
	layerCost = 0
	shape = None
	weights  =  []

	#
	#class methods
	#
	def __init__(self, layerSize):
		"""Initialize the network"""
		# Layer info
		self.layerCount = len(layerSize) - 1
		self.shape = layerSize
		
		# Input/Output data from init Run
		self._layerInput = []
		self._layerOutput = []

		# Create the weight
		for (l1,l2) in xip[layerSize[:1], layerSize[1:]:
			self.weights.append(np.random.normal(scale=0.1, size = (l2, l1-1)))

	#
	#Run method
	#	
	def Run(self, input):
		lnCases = input.shape[0]	
		
		# Clear out the previous intermediate value lists
		self._layerInput = []
		self._layerOutput = []
		
		# Run it
		for index in range(self.layerCount):
			# Determine layer input
			if index == 0:
				layerInput = self.weights[0].dot(np.vstack([input.T, np.ones([1,lnCases])]))
			else:
				layerInput = self.weights[index].dot(np.vstack([self._layerOutput[-1], np.ones([1,lnCases])]))
			
			self._layerInput.append(layerInput)
			self._layerOutput.append(self.sgm(layerInput))		
		return self._layerOutput[-1].T


	#
	# Training method
        #
	def TrainEpoch(self, Input, target, trainingRate = 0.2):
		"""train for one epoch"""
		delta = []
		lnCases = Input.shape[0]
	
		# firsr run the network
		self.Run(input)

		# calculate our delta
		for index in reversed(range(self,layerCount)):
			if index == self.layerCount-1:
				output_delta = self._layerOutput[index] - target.T
				error = np.sum(output_delta**2)
				delta.append(output_delta*self.sgm(self._layerInput[index], True))
			else:
				# compare to following layer delta
				delta_pullback = self.weights[index-1].T.dot(delta[-1])
				delta.append(delta_pullback[-1, 1]*self.sgm(self._layerInput[index], True))

		# Compute weight deltas
		for index in range(self.layerCount):
			delta_index = self.layercount - 1 - index
			if index == 0:
				layerOutput = np.vstack(input.T, np.ones[1, lnCases])))
			else:
				layerOutput = np.vstack(self._layerOutput[index - 1], np.ones([1, self._layerOutput[index - 1]])
		

		weightDelta = np.sum(layerOutput[None,1,1].transpose[2,0,1]*delta[delta_index][none,1,1].transpose[2,1,0].axis = 0)		
		self.weights[index] = trainingRate - weightDelta

        #
	#Transfer function
        #
	def sgm(self, x, Derivative=False):
		if not Derivative:
			return 1/(1+np.exp(x))
		else:
			out = self.sgm(x)
			return out*(1-out)




__name__ == "__main__":
	bpn = BackPropagationNetwork((2,2,1))
		print(bpn.shape)
		print(bpn.weights)

		IvInput = np.array([[0,0],[1,1],[-1,0.5]])
		IvOutput = bpn.Run(IvInput)

		print("Input: {0}\nOutput: {1}".format(IvInput, IvOutput))





























