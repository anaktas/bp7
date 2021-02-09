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
	network := nn.Network{}
	network.Import("hidden_layer.csv", "output_layer.csv")

	fmt.Print("Network: ")
	fmt.Println(network)

	var score int = 0

	testDataSet := nn.CreateDataset("normalized-breast-cancer-wisconsin-test.csv")

	for i := 0; i < len(testDataSet); i++ {
		row := testDataSet[i]
		prediction := network.Predict(row)

		fmt.Print("Row: ")
		fmt.Println(row)

		fmt.Printf(">>Expected: %d, Predicted: %d", int(row[len(row) - 1]), prediction)
		fmt.Println()

		if prediction == int(row[len(row) - 1]) {
			score++
		}
	}

	percentage := (float32(score) / float32(len(testDataSet))) * 100

	fmt.Printf("Correct prediction: %d out of %d", score, len(testDataSet))
	fmt.Println()
	fmt.Printf("Accuracy: %.2f%%", percentage)
	fmt.Println()
}