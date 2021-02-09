package bp7

import(
	"encoding/csv"
	"io"
	"log"
	"os"
	"strconv"
)

// Converts a .csv file into a two dimensional array.
func CreateDataset(filePath string) [][]float32 {
	csvFile, err := os.Open(filePath)
	if err != nil {
		log.Fatalln("Couldn't open the csv file", err)
	}

	reader := csv.NewReader(csvFile)

	dataSet := make([][]float32, 0)

	for {
		record, err := reader.Read()

		if err == io.EOF {
			break
		}

		if err != nil {
			log.Fatal(err)
		}

		entry := make([]float32, 0)

		for i := 0; i < len(record); i++ {
			value, err := strconv.ParseFloat(record[i], 32)
			if err != nil {
				log.Fatal(err)
			}

			entry = append(entry, float32(value))
		}

		dataSet = append(dataSet, entry)
	}

	return dataSet
}