package bp7

import (
	"math/rand"
)

// A hidden layer contains a list of neurons.
type HiddenLayer struct {
	Neurons []Neuron
}

// A hidden layer must be initialized with some random initial values.
func CreateHiddenLayer(neuronCount int, inputCount int) HiddenLayer {
	hl := HiddenLayer{}

	neurons := make([]Neuron, 0)

	for i := 0; i < neuronCount; i++ {
		weights := make([]float32, 0)

		for j := 0; j < inputCount + 1; j++ {
			weight := rand.Float32()
			weights = append(weights, weight)
		}

		neuron := Neuron{}
		neuron.Weights = weights

		neurons = append(neurons, neuron)
	}

	hl.Neurons = neurons

	return hl
}

func (hl *HiddenLayer) Init(neuronCount int, inputCount int) {
	neurons := make([]Neuron, 0)

	for i := 0; i < neuronCount; i++ {
		weights := make([]float32, 0)

		for j := 0; j < inputCount + 1; j++ {
			weight := rand.Float32()
			weights = append(weights, weight)
		}

		neuron := Neuron{}
		neuron.Weights = weights

		neurons = append(neurons, neuron)
	}

	hl.Neurons = neurons
}