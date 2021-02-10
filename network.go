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

import(
	"fmt"
	"math"
	"encoding/csv"
	"io"
	"log"
	"os"
	"strconv"
)

// The representation of an MLP neural
// network. A network contains a hidden
// and an output layers.
type Network struct {
	HiddenLayer HiddenLayer
	OutputLayer OutputLayer
}

// ======================= //
// The structure functions //
// ======================= //

// Initializes the MPL network by assigning an initial hidden layer and
// an initial output layer.
// -Input inputNeuronsCount: How many inputs does the network have.
// -Input hiddenLayerNeuronsCount: How many neurons are present in
// the hidden layer.
// -Input outputLayerNeuronsCount: How many neurons are present in
// the output layer.
func (n *Network) Init(inputNeuronsCount int, hiddenLayerNeuronsCount int, outputLayerNeuronsCount int) {
	hiddenLayer := HiddenLayer{}
	hiddenLayer.Init(hiddenLayerNeuronsCount, inputNeuronsCount)

	n.HiddenLayer = hiddenLayer

	outputLayer := OutputLayer{}
	outputLayer.Init(outputLayerNeuronsCount, hiddenLayerNeuronsCount)

	n.OutputLayer = outputLayer
}

// Propagates the output of each neuron of each layer to 
// the next layer. The output of this function is the final
// output vector of the network.
// -Input row: An entry row of the dataset array.
// -Output: The final output of the network.
func (n *Network) ForwardPropagate(row []float32) []float32 {
	inputs := row

	hiddenLayer := n.HiddenLayer
	outputLayer := n.OutputLayer

	newHiddenLayerInputs := make([]float32, 0)

	hiddenNeurons := hiddenLayer.Neurons

	for i := 0; i < len(hiddenNeurons); i++ {
		activation := hiddenNeurons[i].Activate(inputs)
		output := Output(activation)

		hiddenNeurons[i].Output = output

		newHiddenLayerInputs = append(newHiddenLayerInputs, output)
	}

	inputs = newHiddenLayerInputs

	newOutputLayerInputs := make([]float32, 0)

	outputNeurons := outputLayer.Neurons

	for j := 0; j < len(outputNeurons); j++ {
		activation := outputNeurons[j].Activate(inputs)
		output := Output(activation)

		outputNeurons[j].Output = output

		newOutputLayerInputs = append(newOutputLayerInputs, output)
	}

	finalOutput := newOutputLayerInputs

	return finalOutput
}

// Propagates backwards the calculated error of the final output
// in order let each network layer to update it's neuron weights.
// -Input expected: Is the array of the expected output values.
func (n *Network) BackPropagate(expected []float32) {
	// First we calculate the error of the output layer.
	outputLayerError := make([]float32, 0)

	for i := 0; i < len(n.OutputLayer.Neurons); i++ {
		error := expected[i] - n.OutputLayer.Neurons[i].Output
		outputLayerError = append(outputLayerError, error)
	}

	// We assign each error to the delta variable of each output neuron.
	for i := 0; i < len(n.OutputLayer.Neurons); i++ {
		n.OutputLayer.Neurons[i].Delta = outputLayerError[i] * OutputDerivative(n.OutputLayer.Neurons[i].Output)
	}

	// We propagate the error to the hidden layer.
	hiddenLayerErrors := make([]float32, 0)

	for i := 0; i < len(n.OutputLayer.Neurons); i++ {
		var error float32 = 0.0

		for j := 0; j < len(n.HiddenLayer.Neurons); j++ {
			for k := 0; k < len(n.HiddenLayer.Neurons[j].Weights); k++ {
				error += n.HiddenLayer.Neurons[j].Weights[k] * n.OutputLayer.Neurons[i].Delta
				hiddenLayerErrors = append(hiddenLayerErrors, error)
			}
		}
	}

	// We assign each error to the delta variable of each hidden layer neuron.
	for i := 0; i < len(n.HiddenLayer.Neurons); i++ {
		n.HiddenLayer.Neurons[i].Delta = hiddenLayerErrors[i] * OutputDerivative(n.HiddenLayer.Neurons[i].Output)
	}
}

// Updates each weight of each neuron of each layer during the training iteration.
// -Input row: The training data set entry row.
// -Input learingRate: The rate of the neuron weight adaptation.
func (n *Network) UpdateWeights(row []float32, learningRate float32) {
	// We are dropping out the last value which is the classification value.
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

// Trains a network with a given training data set.
// -Input trainSet: The array of the training data set.
// -Input learningRate: The weight learning adaptation.
// -Input epochs: How many iterations does the training have.
// -Input outputCount: How many classification categories exist
// for the given training set (this should match the output neurons).
func (n *Network) Train(trainSet [][]float32, learningRate float32, epochs int, outputCount int) {
	for i := 0; i < epochs; i++ {
		var sumError float32 = 0.0

		for j := 0; j < len(trainSet); j++ {
			row := trainSet[j]
			// Forward propagating the output.
			outputs := n.ForwardPropagate(row)

			// The expected array contains only zeros.
			expected := make([]float32, 0)
			for k := 0; k < outputCount; k++ {
				expected = append(expected, 0)
			}

			//fmt.Print("Outputs: ")
			//fmt.Println(outputs)

			// We assign '1' to the index of the classification value.
			// In example if the classification array is [0, 1, 2, 3]
			// and the class of the row is 2, we want to modify the
			// expected array in order to make it [0, 0, 1, 0].
			expected[int(row[len(row) - 1])] = 1

			//fmt.Print("Expected: ")
			//fmt.Println(expected)

			var error float32 = 0.0
			for k := 0; k < len(expected); k++ {
				error += float32(math.Pow(float64(expected[k] - outputs[k]), 2))
			}
			sumError += error

			// Backwards propagating the error.
			n.BackPropagate(expected)
			// Updating the weight of each neuron of each layer.
			n.UpdateWeights(row, learningRate)
		}

		fmt.Printf("+Epoch: %d, Learning rate: %.2f, Error: %.2f", i, learningRate, sumError)
		fmt.Println()
	}
}

// Given an input row, it predicts the output categorization.
// -Input row: An entry to predict the category.
func (n *Network) Predict(row []float32) int {
	outputs := n.ForwardPropagate(row)
	fmt.Print("Predicted outputs: ")
	fmt.Println(outputs)

	// We have to find the maximum value of the
	// output array. I.e. if the output array is
	// [0.7864535, 0.235678] we consider that the
	// output is [1, 0] which means that the 
	// prediction is that the entry belongs to the
	// first class/category of the available 
	// classes/categories.
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

// Extracts the hidden layer and the output layer neuron weights.
func (n *Network) Extract() {
	hiddenLayer := n.HiddenLayer
	outputLayer := n.OutputLayer

	hiddenLayerFile, err := os.Create("hidden_layer.csv")
	defer hiddenLayerFile.Close()

	if err != nil {
		panic("Fail to create file")
	}

	hiddenLayerWriter := csv.NewWriter(hiddenLayerFile)
	defer hiddenLayerWriter.Flush()

	for i := 0; i < len(hiddenLayer.Neurons); i++ {
		weights := hiddenLayer.Neurons[i].Weights

		strWeights := make([]string, 0)

		for j := 0; j < len(weights); j++ {
			strValue := fmt.Sprintf("%g", weights[j])

			strWeights = append(strWeights, strValue)
		}

		if err := hiddenLayerWriter.Write(strWeights); err != nil {
			panic("Fail to write hidden layer weight to file")
		}
	}

	outputLayerFile, err := os.Create("output_layer.csv")
	defer outputLayerFile.Close()

	if err != nil {
		panic("Fail to create file")
	}

	outputLayerWriter := csv.NewWriter(outputLayerFile)
	defer outputLayerWriter.Flush()

	for i := 0; i < len(outputLayer.Neurons); i++ {
		weights := outputLayer.Neurons[i].Weights

		strWeights := make([]string, 0)

		for j := 0; j < len(weights); j++ {
			strValue := fmt.Sprintf("%g", weights[j])

			strWeights = append(strWeights, strValue)
		}

		if err := outputLayerWriter.Write(strWeights); err != nil {
			panic("Fail to write hidden layer weight to file")
		}
	}
}

// Imports the hidden and the output layer neuron weights into the network.
// -Input hiddenFilePath: The hidden layer neuron weights file path.
// -Inout outputFilePath: The output layer neuron weights file path.
func (n *Network) Import(hiddenFilePath string, outputFilePath string) {
	hiddenFile, err := os.Open(hiddenFilePath)
	if err != nil {
		log.Fatalln("Couldn't open the hidden layer csv file", err)
	}

	hiddenReader := csv.NewReader(hiddenFile)

	hiddenNeurons := make([]Neuron, 0)

	for {
		entry, err := hiddenReader.Read()

		if err == io.EOF {
			break
		}

		if err != nil {
			log.Fatal(err)
		}

		weights := make([]float32, 0)

		for i := 0; i < len(entry); i++ {
			weight, err := strconv.ParseFloat(entry[i], 32)
			if err != nil {
				log.Fatal(err)
			}

			weights = append(weights, float32(weight))
		}

		neuron := Neuron{}
		neuron.Weights = weights

		hiddenNeurons = append(hiddenNeurons, neuron)
	}

	n.HiddenLayer.Neurons = hiddenNeurons

	outputFile, err := os.Open(outputFilePath)
	if err != nil {
		log.Fatalln("Couldn't open the output layer csv file", err)
	}

	outputReader := csv.NewReader(outputFile)

	outputNeurons := make([]Neuron, 0)

	for {
		entry, err := outputReader.Read()

		if err == io.EOF {
			break
		}

		if err != nil {
			log.Fatal(err)
		}

		weights := make([]float32, 0)

		for i := 0; i < len(entry); i++ {
			weight, err := strconv.ParseFloat(entry[i], 32)
			if err != nil {
				log.Fatal(err)
			}

			weights = append(weights, float32(weight))
		}

		neuron := Neuron{}
		neuron.Weights = weights

		outputNeurons = append(outputNeurons, neuron)
	}

	n.OutputLayer.Neurons = outputNeurons
}

// ======================== //
// The standalone functions //
// ======================== //

// Initializes the MPL network by assigning an initial hidden layer and
// an initial output layer.
// -Input inputNeuronsCount: How many inputs does the network have.
// -Input hiddenLayerNeuronsCount: How many neurons are present in
// the hidden layer.
// -Input outputLayerNeuronsCount: How many neurons are present in
// the output layer.
// -Output: Returns a network structure.
func CreateNetwork(inputNeuronsCount int, hiddenLayerNeuronsCount int, outputLayerNeuronsCount int) Network {
	network := Network{}

	hiddenLayer := CreateHiddenLayer(hiddenLayerNeuronsCount, inputNeuronsCount)

	network.HiddenLayer = hiddenLayer

	outputLayer := CreateOutputLayer(outputLayerNeuronsCount, hiddenLayerNeuronsCount)

	network.OutputLayer = outputLayer

	return network
}

// Propagates the output of each neuron of each layer to 
// the next layer. The output of this function is the final
// output vector of the network.
// -Input n: A network.
// -Input row: An entry row of the dataset array.
// -Output: The final output of the network.
func ForwardPropagate(n *Network, row []float32) []float32 {
	inputs := row

	hiddenLayer := n.HiddenLayer
	outputLayer := n.OutputLayer

	newHiddenLayerInputs := make([]float32, 0)

	hiddenNeurons := hiddenLayer.Neurons

	for i := 0; i < len(hiddenNeurons); i++ {
		activation := Activate(&hiddenNeurons[i], inputs)
		output := Output(activation)

		hiddenNeurons[i].Output = output

		newHiddenLayerInputs = append(newHiddenLayerInputs, output)
	}

	inputs = newHiddenLayerInputs

	newOutputLayerInputs := make([]float32, 0)

	outputNeurons := outputLayer.Neurons

	for j := 0; j < len(outputNeurons); j++ {
		activation := Activate(&outputNeurons[j], inputs)
		output := Output(activation)

		outputNeurons[j].Output = output

		newOutputLayerInputs = append(newOutputLayerInputs, output)
	}

	inputs = newOutputLayerInputs

	return inputs
}

// Propagates backwards the calculated error of the final output
// in order let each network layer to update it's neuron weights.
// -Input n: A network.
// -Input expected: Is the array of the expected output values.
func BackPropagate(n *Network, expected []float32) {
	outputLayerError := make([]float32, 0)

	for i := 0; i < len(n.OutputLayer.Neurons); i++ {
		error := expected[i] - n.OutputLayer.Neurons[i].Output
		outputLayerError = append(outputLayerError, error)
	}

	for i := 0; i < len(n.OutputLayer.Neurons); i++ {
		n.OutputLayer.Neurons[i].Delta = outputLayerError[i] * OutputDerivative(n.OutputLayer.Neurons[i].Output)
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
		n.HiddenLayer.Neurons[i].Delta = hiddenLayerErrors[i] * OutputDerivative(n.HiddenLayer.Neurons[i].Output)
	}
}

// Updates each weight of each neuron of each layer during the training iteration.
// -Input n: A network.
// -Input row: The training data set entry row.
// -Input learingRate: The rate of the neuron weight adaptation.
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

// Trains a network with a given training data set.
// -Input n: A network.
// -Input trainSet: The array of the training data set.
// -Input learningRate: The weight learning adaptation.
// -Input epochs: How many iterations does the training have.
// -Input outputCount: How many classification categories exist
// for the given training set (this should match the output neurons).
func Train(n *Network, trainSet [][]float32, learningRate float32, epochs int, outputCount int) {
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

// Given an input row, it predicts the output categorization.
// -Input n: A network.
// -Input row: An entry to predict the category.
func Predict(n *Network, row []float32) int {
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

// Extracts the hidden layer and the output layer neuron weights.
// -Input n: A network.
func Extract(n *Network) {
	hiddenLayer := n.HiddenLayer
	outputLayer := n.OutputLayer

	hiddenLayerFile, err := os.Create("hidden_layer.csv")
	defer hiddenLayerFile.Close()

	if err != nil {
		panic("Fail to create file")
	}

	hiddenLayerWriter := csv.NewWriter(hiddenLayerFile)
	defer hiddenLayerWriter.Flush()

	for i := 0; i < len(hiddenLayer.Neurons); i++ {
		weights := hiddenLayer.Neurons[i].Weights

		strWeights := make([]string, 0)

		for j := 0; j < len(weights); j++ {
			strValue := fmt.Sprintf("%g", weights[j])

			strWeights = append(strWeights, strValue)
		}

		if err := hiddenLayerWriter.Write(strWeights); err != nil {
			panic("Fail to write hidden layer weight to file")
		}
	}

	outputLayerFile, err := os.Create("output_layer.csv")
	defer outputLayerFile.Close()

	if err != nil {
		panic("Fail to create file")
	}

	outputLayerWriter := csv.NewWriter(outputLayerFile)
	defer outputLayerWriter.Flush()

	for i := 0; i < len(outputLayer.Neurons); i++ {
		weights := hiddenLayer.Neurons[i].Weights

		strWeights := make([]string, 0)

		for j := 0; j < len(weights); j++ {
			strValue := fmt.Sprintf("%g", weights[j])

			strWeights = append(strWeights, strValue)
		}

		if err := outputLayerWriter.Write(strWeights); err != nil {
			panic("Fail to write hidden layer weight to file")
		}
	}
}

// Imports the hidden and the output layer neuron weights into the network.
// -Input n: A network.
// -Input hiddenFilePath: The hidden layer neuron weights file path.
// -Inout outputFilePath: The output layer neuron weights file path.
func Import(n *Network, hiddenFilePath string, outputFilePath string) {
	hiddenFile, err := os.Open(hiddenFilePath)
	if err != nil {
		log.Fatalln("Couldn't open the hidden layer csv file", err)
	}

	hiddenReader := csv.NewReader(hiddenFile)

	hiddenNeurons := make([]Neuron, 0)

	for {
		entry, err := hiddenReader.Read()

		if err == io.EOF {
			break
		}

		if err != nil {
			log.Fatal(err)
		}

		weights := make([]float32, 0)

		for i := 0; i < len(entry); i++ {
			weight, err := strconv.ParseFloat(entry[i], 32)
			if err != nil {
				log.Fatal(err)
			}

			weights = append(weights, float32(weight))
		}

		neuron := Neuron{}
		neuron.Weights = weights

		hiddenNeurons = append(hiddenNeurons, neuron)
	}

	n.HiddenLayer.Neurons = hiddenNeurons

	outputFile, err := os.Open(outputFilePath)
	if err != nil {
		log.Fatalln("Couldn't open the output layer csv file", err)
	}

	outputReader := csv.NewReader(outputFile)

	outputNeurons := make([]Neuron, 0)

	for {
		entry, err := outputReader.Read()

		if err == io.EOF {
			break
		}

		if err != nil {
			log.Fatal(err)
		}

		weights := make([]float32, 0)

		for i := 0; i < len(entry); i++ {
			weight, err := strconv.ParseFloat(entry[i], 32)
			if err != nil {
				log.Fatal(err)
			}

			weights = append(weights, float32(weight))
		}

		neuron := Neuron{}
		neuron.Weights = weights

		outputNeurons = append(outputNeurons, neuron)
	}

	n.OutputLayer.Neurons = outputNeurons
}

// Calculates the output derivative/slope.
func OutputDerivative(output float32) float32 {
	return output * (1.0 - output)
}