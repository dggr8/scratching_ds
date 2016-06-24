import math

def dot(x,y):
	return sum([i*j for i,j in zip(x,y)])

def sigmoid(x):
	return 1.0/(1.0+math.exp(-x))

def neuron_output(weights,inputs):
	return sigmoid(dot(weights,inputs))

def feed_forward(neural_network, input_vector):
	"""takes in a neural network and returns
	the output from forward-propagating the input"""

	outputs = []

	for layer in neural_network:
		input_with_bias = input_vector+[1]
		output = [neuron_output(neuron,input_with_bias)
					for neuron in layer]
		outputs.append(output)

		input_vector = output
	
	return outputs

def main():
	xor_network = [
					[[20,20,-30],
					 [20,20,-10]],
					 [[-60,60,-30]]]

	for x in [0,1]:
		for y in [0,1]:
			print x,y, feed_forward(xor_network,[x,y])[-1]

if __name__=="__main__":
	main()
