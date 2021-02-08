package bp7

import(
	"fmt"
	"math"
)

// A network is a list of layers
type Network struct {
	HiddenLayer HiddenLayer
	OutputLayer OutputLayer
}

// Initializing the network layers with some random weights.
func CreateNetwork(inputNeuronsCount int, hiddenLayerNeuronsCount int, outputLayerNeuronsCount int) Network {
	network := Network{}

	hiddenLayer := CreateHiddenLayer(hiddenLayerNeuronsCount, inputNeuronsCount)

	network.HiddenLayer = hiddenLayer

	outputLayer := CreateOutputLayer(outputLayerNeuronsCount, hiddenLayerNeuronsCount)

	network.OutputLayer = outputLayer

	return network
}

func (n *Network) Init(inputNeuronsCount int, hiddenLayerNeuronsCount int, outputLayerNeuronsCount int) {
	hiddenLayer := HiddenLayer{}
	hiddenLayer.Init(hiddenLayerNeuronsCount, inputNeuronsCount)

	n.HiddenLayer = hiddenLayer

	outputLayer := OutputLayer{}
	outputLayer.Init(outputLayerNeuronsCount, hiddenLayerNeuronsCount)

	n.OutputLayer = outputLayer
}

func ForwardPropagate(n *Network, row []float32) []float32 {
	inputs := row

	hiddenLayer := n.HiddenLayer
	outputLayer := n.OutputLayer

	newHiddenLayerInputs := make([]float32, 0)

	hiddenNeurons := hiddenLayer.Neurons

	for i := 0; i < len(hiddenNeurons); i++ {
		activation := Activate(hiddenNeurons[i].Weights, inputs)
		output := Transfer(activation)

		hiddenNeurons[i].Output = output

		newHiddenLayerInputs = append(newHiddenLayerInputs, output)
	}

	inputs = newHiddenLayerInputs

	newOutputLayerInputs := make([]float32, 0)

	outputNeurons := outputLayer.Neurons

	for j := 0; j < len(outputNeurons); j++ {
		activation := Activate(outputNeurons[j].Weights, inputs)
		output := Transfer(activation)

		outputNeurons[j].Output = output

		newOutputLayerInputs = append(newOutputLayerInputs, output)
	}

	inputs = newOutputLayerInputs

	return inputs
}

func (n *Network) ForwardPropagate(row []float32) []float32 {
	inputs := row

	hiddenLayer := n.HiddenLayer
	outputLayer := n.OutputLayer

	newHiddenLayerInputs := make([]float32, 0)

	hiddenNeurons := hiddenLayer.Neurons

	for i := 0; i < len(hiddenNeurons); i++ {
		activation := hiddenNeurons[i].Activate(inputs)
		output := Transfer(activation)

		hiddenNeurons[i].Output = output

		newHiddenLayerInputs = append(newHiddenLayerInputs, output)
	}

	inputs = newHiddenLayerInputs

	newOutputLayerInputs := make([]float32, 0)

	outputNeurons := outputLayer.Neurons

	for j := 0; j < len(outputNeurons); j++ {
		activation := outputNeurons[j].Activate(inputs)
		output := Transfer(activation)

		outputNeurons[j].Output = output

		newOutputLayerInputs = append(newOutputLayerInputs, output)
	}

	inputs = newOutputLayerInputs

	return inputs
}

func Transfer(activation float32) float32 {
	return (float32) (1.0 / (1.0 + math.Exp(-1 * float64(activation))))
}

func TransferDerivative(output float32) float32 {
	return output * (1.0 - output)
}

func (n *Network) BackPropagate(expected []float32) {
	outputLayerError := make([]float32, 0)

	for i := 0; i < len(n.OutputLayer.Neurons); i++ {
		error := expected[i] - n.OutputLayer.Neurons[i].Output
		outputLayerError = append(outputLayerError, error)
	}

	for i := 0; i < len(n.OutputLayer.Neurons); i++ {
		n.OutputLayer.Neurons[i].Delta = outputLayerError[i] * TransferDerivative(n.OutputLayer.Neurons[i].Output)
	}

	hiddenLayerErrors := make([]float32, 0)

	for i := 0; i < len(n.HiddenLayer.Neurons); i++ {
		var error float32 = 0.0

		for j := 0; j < len(n.HiddenLayer.Neurons[i].Weights); j++ {
			error += n.HiddenLayer.Neurons[i].Weights[j] * n.HiddenLayer.Neurons[i].Delta
			hiddenLayerErrors = append(hiddenLayerErrors, error)
		}
	}

	for i := 0; i < len(n.HiddenLayer.Neurons); i++ {
		n.HiddenLayer.Neurons[i].Delta = hiddenLayerErrors[i] * TransferDerivative(n.HiddenLayer.Neurons[i].Output)
	}
}

func BackPropagate(n *Network, expected []float32) {
	outputLayerError := make([]float32, 0)

	for i := 0; i < len(n.OutputLayer.Neurons); i++ {
		error := expected[i] - n.OutputLayer.Neurons[i].Output
		outputLayerError = append(outputLayerError, error)
	}

	for i := 0; i < len(n.OutputLayer.Neurons); i++ {
		n.OutputLayer.Neurons[i].Delta = outputLayerError[i] * TransferDerivative(n.OutputLayer.Neurons[i].Output)
	}

	hiddenLayerErrors := make([]float32, 0)

	for i := 0; i < len(n.HiddenLayer.Neurons); i++ {
		var error float32 = 0.0

		for j := 0; j < len(n.HiddenLayer.Neurons[i].Weights); j++ {
			error += n.HiddenLayer.Neurons[i].Weights[j] * n.HiddenLayer.Neurons[i].Delta
			hiddenLayerErrors = append(hiddenLayerErrors, error)
		}
	}

	for i := 0; i < len(n.HiddenLayer.Neurons); i++ {
		n.HiddenLayer.Neurons[i].Delta = hiddenLayerErrors[i] * TransferDerivative(n.HiddenLayer.Neurons[i].Output)
	}
}

func (n *Network) UpdateWeights(row []float32, learningRate float32) {
	inputs := row[0:(len(row) - 1)]

	for i := 0; i < len(n.HiddenLayer.Neurons); i++ {
		for j := 0; j < len(inputs); j++ {
			n.HiddenLayer.Neurons[i].Weights[j] += learningRate * n.HiddenLayer.Neurons[i].Delta * inputs[j]
		}

		weightsLength := len(n.HiddenLayer.Neurons[i].Weights)
		n.HiddenLayer.Neurons[i].Weights[weightsLength - 1] += learningRate * n.HiddenLayer.Neurons[i].Delta
	}

	outputLayerInputs := make([]float32, 0)

	for i := 0; i < len(n.HiddenLayer.Neurons); i++ {
		outputLayerInputs = append(outputLayerInputs, n.HiddenLayer.Neurons[i].Output)
	}

	for i := 0; i < len(n.OutputLayer.Neurons); i++ {
		for j := 0; j < len(outputLayerInputs); j++ {
			n.OutputLayer.Neurons[i].Weights[j] += learningRate * n.OutputLayer.Neurons[i].Delta * outputLayerInputs[j]
		}

		weightsLength := len(n.OutputLayer.Neurons[i].Weights)
		n.OutputLayer.Neurons[i].Weights[weightsLength - 1] += learningRate * n.OutputLayer.Neurons[i].Delta
	}
}

func UpdateWeights(n *Network, row []float32, learningRate float32) {
	inputs := row[0:(len(row) - 1)]

	for i := 0; i < len(n.HiddenLayer.Neurons); i++ {
		for j := 0; j < len(inputs); j++ {
			n.HiddenLayer.Neurons[i].Weights[j] += learningRate * n.HiddenLayer.Neurons[i].Delta * inputs[j]
		}

		weightsLength := len(n.HiddenLayer.Neurons[i].Weights)
		n.HiddenLayer.Neurons[i].Weights[weightsLength - 1] += learningRate * n.HiddenLayer.Neurons[i].Delta
	}

	outputLayerInputs := make([]float32, 0)

	for i := 0; i < len(n.HiddenLayer.Neurons); i++ {
		outputLayerInputs = append(outputLayerInputs, n.HiddenLayer.Neurons[i].Output)
	}

	for i := 0; i < len(n.OutputLayer.Neurons); i++ {
		for j := 0; j < len(outputLayerInputs); j++ {
			n.OutputLayer.Neurons[i].Weights[j] += learningRate * n.OutputLayer.Neurons[i].Delta * outputLayerInputs[j]
		}

		weightsLength := len(n.OutputLayer.Neurons[i].Weights)
		n.OutputLayer.Neurons[i].Weights[weightsLength - 1] += learningRate * n.OutputLayer.Neurons[i].Delta
	}
}

func (n *Network) Train(trainSet [][]float32, learningRate float32, epochs int, outputCount int) {
	for i := 0; i < epochs; i++ {
		var sumError float32 = 0.0

		for j := 0; j < len(trainSet); j++ {
			row := trainSet[j]
			outputs := n.ForwardPropagate(row)

			expected := make([]float32, 0)
			for k := 0; k < outputCount; k++ {
				expected = append(expected, 0)
			}

			//fmt.Print("Outputs: ")
			//fmt.Println(outputs)

			expected[int(row[len(row) - 1])] = 1

			//fmt.Print("Expected: ")
			//fmt.Println(expected)

			var error float32 = 0.0
			for k := 0; k < len(expected); k++ {
				error += float32(math.Pow(float64(expected[k] - outputs[k]), 2))
			}
			sumError += error

			n.BackPropagate(expected)
			n.UpdateWeights(row, learningRate)
		}

		fmt.Printf("+Epoch: %d, Learning rate: %.2f, Error: %.2f", i, learningRate, sumError)
		fmt.Println()
	}
}

func (n *Network) Predict(row []float32) int {
	outputs := n.ForwardPropagate(row)
	fmt.Print("Predicted outputs: ")
	fmt.Println(outputs)

	var max float32 = 0.0

	for i := 0; i < len(outputs); i++ {
		if outputs[i] > max {
			max = outputs[i]
		}
	}

	for i := 0; i < len(outputs); i++ {
		if max == outputs[i] {
			return i;
		}
	}

	return 0
}