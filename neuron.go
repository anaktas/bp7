// Copyright Copyright 2021 Anastasios Daris
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

import(
	"math"
)

// The neuron representation. Each neuron contains
// an array of weights (a weight for each input),
// the neuron output and a delta. The delta is the
// calculated error signal which is used in order
// to update the weights during training.
type Neuron struct {
	Weights []float32
	Output float32
	Delta float32
}

// The structure function implementation for the 
// activation of the neuron. The activation of 
// a neuron is the sum of the multiplication of
// each inout with each weight.
// -Input inputs: An array of the inouts.
func (n *Neuron) Activate(inputs []float32) float32 {
	activation := n.Weights[len(n.Weights) - 1]

	for i := 0; i < len(n.Weights); i++ {
		if i > (len(inputs) - 1) {
			break
		}

		activation += n.Weights[i] * inputs[i]
	}

	return activation
}

// The standalone function implementation for the 
// activation of the neuron. The activation of 
// a neuron is the sum of the multiplication of
// each inout with each weight.
// -Input inputs: An array of the inouts.
func Activate(weights []float32, inputs []float32) float32 {
	activation := weights[len(weights) - 1]

	for i := 0; i < len(weights); i++ {
		if i > (len(inputs) - 1) {
			break
		}

		activation += weights[i] * inputs[i]
	}

	return activation
}

// The neuron output function implementation. After 
// the calculation of the activation of the neuron, 
// we are using the sigmoid function in order to 
// calculate the actual output.
// -Input activation: The activation summary of the
// neuron.
//
//
// output = 1 / (1 + e^(-activation))
func Output(activation float32) float32 {
	return (float32) (1.0 / (1.0 + math.Exp(-1 * float64(activation))))
}