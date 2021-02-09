package bp7

import (
	"math/rand"
)

// The representation of the output layer of
// an MLP neural network. The output layer
// contains an array of neurons.
type OutputLayer struct {
	Neurons []Neuron
}

// The structure function implementation of the
// output layer initialization. During the
// initialization, we assign a random weight in
// the weight array of each neuron.
// -Input neuronCount: How many neurons are in the
// output layer.
// -Input hiddenNeuronsCount: How many hidden layer
// inputs.
func (ol *OutputLayer) Init(neuronCount int, hiddenNeuronsCount int) {
	neurons := make([]Neuron, 0)

	for i := 0; i < neuronCount; i++ {
		weights := make([]float32, 0)

		for j := 0; j < hiddenNeuronsCount + 1; j++ {
			weight := rand.Float32()
			weights = append(weights, weight)
		}

		neuron := Neuron{}
		neuron.Weights = weights

		neurons = append(neurons, neuron)
	}

	ol.Neurons = neurons
}

// The standalone function implementation of the
// output layer initialization. During the
// initialization, we assign a random weight in
// the weight array of each neuron.
// -Input neuronCount: How many neurons are in the
// output layer.
// -Input hiddenNeuronsCount: How many hidden layer
// inputs.
func CreateOutputLayer(neuronCount int, hiddenNeuronsCount int) OutputLayer {
	ol := OutputLayer{}

	neurons := make([]Neuron, 0)

	for i := 0; i < neuronCount; i++ {
		weights := make([]float32, 0)

		for j := 0; j < hiddenNeuronsCount + 1; j++ {
			weight := rand.Float32()
			weights = append(weights, weight)
		}

		neuron := Neuron{}
		neuron.Weights = weights

		neurons = append(neurons, neuron)
	}

	ol.Neurons = neurons

	return ol
}