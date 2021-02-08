package bp7

// The node struct represents the node of a neural network.
// Each node contains an array of neuron weigths.
type Neuron struct {
	Weights []float32
	Output float32
	Delta float32
}

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