package bp7

import (
	"math/rand"
)

// An output layer contains a list of neurons.
type OutputLayer struct {
	Neurons []Neuron
}

// An output layer must be initialized with some random initial weights.
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