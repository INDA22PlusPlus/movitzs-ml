package main

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"time"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"movitz-sunar-ctf-challs-enterprise.com.gov/skynet/mnist"
)

// yoinked from https://datadan.io/blog/neural-net-with-go
// i would fail me if i was you

func main() {
	dataSet, err := mnist.ReadDataSet("./train-images.idx3-ubyte", "./train-labels.idx1-ubyte")
	if err != nil {
		panic(err.Error())
	}
	n := network{}

	rand.Seed(time.Now().Unix())

	n.train(dataSet)

	correct := 0
	for i := 55001; i < 60000; i++ {
		if n.predictPrint(dataSet, i) {
			correct++
		}
	}
	fmt.Println("accuracy:", float64(correct)/5000.0)
}

func (n *network) predictPrint(dataSet *mnist.DataSet, x int) bool {
	res, err := n.predict(mat.NewDense(1, 28*28, normalize(dataSet.Data[x].Image)))
	if err != nil {
		panic(err.Error())
	}

	biggest := -1.0
	biggestI := -1
	for i, v := range res.RawMatrix().Data {
		if biggest < v {
			biggestI = i
			biggest = v
		}
	}

	return biggestI == dataSet.Data[x].Digit
}

type network struct {
	wHidden *mat.Dense
	bHidden *mat.Dense
	wOut    *mat.Dense
	bOut    *mat.Dense
}

var (
	hiddenNodes = 128
	outputNodes = 10
	inputNodes  = 28 * 28
)

func (n *network) train(x *mnist.DataSet) {
	n.wHidden = mat.NewDense(inputNodes, hiddenNodes, nil)
	n.bHidden = mat.NewDense(1, hiddenNodes, nil)
	n.wOut = mat.NewDense(hiddenNodes, outputNodes, nil)
	n.bOut = mat.NewDense(1, outputNodes, nil)

	for _, x := range [][]float64{
		n.bHidden.RawMatrix().Data,
		n.wHidden.RawMatrix().Data,
		n.wOut.RawMatrix().Data,
		n.bOut.RawMatrix().Data,
	} {
		for i := range x {
			x[i] = rand.Float64()
		}
	}

	n.backpropagate(x)
}

func (n *network) backpropagate(x *mnist.DataSet) {
	// 60000 samples
	for xz := 0; xz < 3; xz++ {
		for i := 0; i < 55000; i++ {

			output := mat.NewDense(1, 10, nil)
			learningRate := 0.2
			dat := x.Data[i]
			datNorm := normalize(dat.Image)
			datMatr := mat.NewDense(1, 28*28, datNorm)

			hiddenLayerInput := new(mat.Dense)
			hiddenLayerInput.Mul(datMatr, n.wHidden)
			hiddenLayerInput.Apply(func(_, j int, v float64) float64 { return v + n.bHidden.At(0, j) }, hiddenLayerInput)

			hiddenLayerActivations := new(mat.Dense)
			hiddenLayerActivations.Apply(func(i, j int, v float64) float64 { return sigmoid(v) }, hiddenLayerInput)

			outputLayerInput := new(mat.Dense)
			outputLayerInput.Mul(hiddenLayerActivations, n.wOut)
			outputLayerInput.Apply(func(i, j int, v float64) float64 { return v + n.bOut.At(0, j) }, outputLayerInput)

			output.Apply(func(i, j int, v float64) float64 { return sigmoid(v) }, outputLayerInput)

			networkError := new(mat.Dense)
			networkError.Sub(correctVector(dat.Digit), output)

			slopeOutputLayer := new(mat.Dense)
			applySigmoidPrime := func(_, _ int, v float64) float64 { return sigmoidPrime(v) }
			slopeOutputLayer.Apply(applySigmoidPrime, output)
			slopeHiddenLayer := new(mat.Dense)
			slopeHiddenLayer.Apply(applySigmoidPrime, hiddenLayerActivations)

			dOutput := new(mat.Dense)
			dOutput.MulElem(networkError, slopeOutputLayer)
			errorAtHiddenLayer := new(mat.Dense)
			errorAtHiddenLayer.Mul(dOutput, n.wOut.T())

			dHiddenLayer := new(mat.Dense)
			dHiddenLayer.MulElem(errorAtHiddenLayer, slopeHiddenLayer)

			// Adjust the parameters.
			wOutAdj := new(mat.Dense)
			wOutAdj.Mul(hiddenLayerActivations.T(), dOutput)
			wOutAdj.Scale(learningRate, wOutAdj)
			n.wOut.Add(n.wOut, wOutAdj)

			bOutAdj, err := sumAlongAxis(0, dOutput)
			if err != nil {
				panic(err.Error())
			}
			bOutAdj.Scale(learningRate, bOutAdj)
			n.bOut.Add(n.bOut, bOutAdj)

			wHiddenAdj := new(mat.Dense)
			wHiddenAdj.Mul(datMatr.T(), dHiddenLayer)
			wHiddenAdj.Scale(learningRate, wHiddenAdj)
			n.wHidden.Add(n.wHidden, wHiddenAdj)

			bHiddenAdj, err := sumAlongAxis(0, dHiddenLayer)
			if err != nil {
				panic("hej: " + err.Error())
			}
			bHiddenAdj.Scale(learningRate, bHiddenAdj)
			n.bHidden.Add(n.bHidden, bHiddenAdj)
		}
	}
}

func (n *network) predict(x *mat.Dense) (*mat.Dense, error) {
	output := new(mat.Dense)

	hiddenLayerInput := new(mat.Dense)
	hiddenLayerInput.Mul(x, n.wHidden)
	addBHidden := func(_, col int, v float64) float64 { return v + n.bHidden.At(0, col) }
	hiddenLayerInput.Apply(addBHidden, hiddenLayerInput)

	hiddenLayerActivations := new(mat.Dense)
	applySigmoid := func(_, _ int, v float64) float64 { return sigmoid(v) }
	hiddenLayerActivations.Apply(applySigmoid, hiddenLayerInput)

	outputLayerInput := new(mat.Dense)
	outputLayerInput.Mul(hiddenLayerActivations, n.wOut)
	addBOut := func(_, col int, v float64) float64 { return v + n.bOut.At(0, col) }
	outputLayerInput.Apply(addBOut, outputLayerInput)
	output.Apply(applySigmoid, outputLayerInput)

	return output, nil
}

func sumAlongAxis(axis int, m *mat.Dense) (*mat.Dense, error) {
	numRows, numCols := m.Dims()

	var output *mat.Dense

	switch axis {
	case 0:
		data := make([]float64, numCols)
		for i := 0; i < numCols; i++ {
			col := mat.Col(nil, i, m)
			data[i] = floats.Sum(col)
		}
		output = mat.NewDense(1, numCols, data)
	case 1:
		data := make([]float64, numRows)
		for i := 0; i < numRows; i++ {
			row := mat.Row(nil, i, m)
			data[i] = floats.Sum(row)
		}
		output = mat.NewDense(numRows, 1, data)
	default:
		return nil, errors.New("invalid axis, must be 0 or 1")
	}

	return output, nil
}

func correctVector(x int) *mat.Dense {
	res := mat.NewDense(1, 10, nil)
	res.Set(0, x, 1.0)
	return res
}

func normalize(x [][]uint8) []float64 {
	res := make([]float64, len(x)*len(x))

	for i := range x {
		for i2 := range x[i] {
			res[i*len(x)+i2] = float64(x[i][i2]) / 256.0
		}
	}

	return res
}

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func sigmoidPrime(x float64) float64 {
	return sigmoid(x) * (1.0 - sigmoid(x))
}
