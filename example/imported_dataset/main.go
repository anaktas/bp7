package main

import(
	"fmt"
	nn "7linternational.com/bp7"
)

func main() {
	dataSet := nn.CreateDataset("normalized-breast-cancer-wisconsin.csv")
	//dataSet := nn.CreateDataset("breast-cancer-wisconsin.csv")
	
	network := nn.Network{}
	network.Init(9, 18, 2)

	fmt.Print("Network: ")
	fmt.Println(network)


	network.Train(dataSet, 0.2, 1000, 2)

	fmt.Println("+++++++++++++++++++++++++++++++++++++")
	fmt.Print("Network after training: ")
	fmt.Println(network)

	var score int = 0

	//testDataSet := nn.CreateDataset("breast-cancer-wisconsin-test.csv")
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

	network.Extract()
}