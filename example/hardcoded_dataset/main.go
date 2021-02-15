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

package main

import(
	"fmt"
	nn "7linternational.com/bp7"
)

func main() {
	dataSet := [][]float32{
		{2.7810836,   2.550537003,  0}, // Expected output [1, 0]
		{1.465489372, 2.362125076,  0}, // Expected output [1, 0]
		{3.396561688, 4.400293529,  0}, // Expected output [1, 0]
		{1.38807019,  1.850220317,  0}, // Expected output [1, 0]
		{3.06407332,  3.005305973,  0}, // Expected output [1, 0]
		{7.627531214, 2.759262235,  1}, // Expected output [0, 1]
		{5.332441248, 2.088626775,  1}, // Expected output [0, 1]
		{6.922596716, 1.77106367,   1}, // Expected output [0, 1]
		{8.675418651, -0.242068655, 1}, // Expected output [0, 1]
		{7.673756466, 3.508563011,  1}, // Expected output [0, 1]
	}
	
	network := nn.Network{}
	network.Init(2, 4, 2)

	fmt.Print("Network: ")
	fmt.Println(network)


	network.Train(dataSet, 0.2, 1000, 2)

	fmt.Println("+++++++++++++++++++++++++++++++++++++")
	fmt.Print("Network after training propagation: ")
	fmt.Println(network)

	var score int = 0

	for i := 0; i < len(dataSet); i++ {
		row := dataSet[i]
		prediction := network.Predict(row)

		fmt.Print("Row: ")
		fmt.Println(row)

		fmt.Printf(">>Expected: %d, Predicted: %d", int(row[len(row) - 1]), prediction)
		fmt.Println()

		if prediction == int(row[len(row) - 1]) {
			score++
		}
	}

	percentage := (float32(score) / float32(len(dataSet))) * 100

	fmt.Printf("Correct prediction: %d out of %d", score, len(dataSet))
	fmt.Println()
	fmt.Printf("Accuracy: %.2f%%", percentage)
	fmt.Println()
}