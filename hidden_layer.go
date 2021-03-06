// Copyright 2021 Anastasios Daris
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package bp7

import (
	"math/rand"
)

// The representation of the hidden layer of
// an MLP neural network. The hidden layer
// contains an array of neurons.
type HiddenLayer struct {
	Neurons []Neuron
}

// The structure function implementation of the
// hidden layer initialization. During the
// initialization, we assign a random weight in
// the weight array of each neuron.
// -Input neuronCount: How many neurons are in the
// hidden layer.
// -Input inputCount: How many dataset inputs.
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

// The standalone function implementation of the
// hidden layer initialization. During the
// initialization, we assign a random weight in
// the weight array of each neuron.
// -Input neuronCount: How many neurons are in the
// hidden layer.
// -Input inputCount: How many dataset inputs.
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