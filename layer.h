#pragma once

#include "neuron.h"

enum LayerType{FullyConnected, Convolutional, MaxPooling, AveragePooling, Abs, Normalization, SubNetwork};

/// a fully connected layer
class Layer
{
protected:
	LayerType type;
	vector<Neuron> neurons;
	int prevLayerIndex;
	int inputSize;
	vector<double> outputs;
	vector<double> weightedSums; // z
	vector<double> derivatives; // derivative of the activation function at z
	vector<vector<double> > wts; // all weights in this layer
	void calcOutputs()
	{
		int nSize = neurons.size();
		outputs = vector<double>(nSize, 0);
#pragma omp parallel for
		for(int i = 0;i < nSize;i++)
		{
			outputs[i] = neurons[i].getOuput();
		}
	}
	void calcWeightedSums()
	{
		int nSize = neurons.size();
		weightedSums = vector<double>(nSize,0);
#pragma omp parallel for
		for (int i = 0; i < nSize; i++)
		{
			weightedSums[i] = neurons[i].getWeightedSum();
		}
	}
	void calcDerivatives()
	{
		int nSize = neurons.size();
		derivatives = vector<double>(nSize, 0);
#pragma omp parallel for
		for(int i = 0;i < nSize;i++)
		{
			derivatives[i] = neurons[i].getDerivative();
		}
	}
	void calcWeights()
	{
		int nSize = neurons.size();
		wts = vector<vector<double> >(nSize, vector<double>());
#pragma omp parallel for
		for(int i = 0;i < nSize;i++)
		{
			wts[i] = neurons[i].getWeights();
		}
	}
	
public:
	virtual void setType()
	{ 
		type = FullyConnected; 
	};
	LayerType getType()
	{
		return type;
	}
	Layer(int prevInd) : prevLayerIndex(prevInd)
	{
		// do nothing
		setType();
	}
	// load
	Layer(ActivationFunction * activFunc, ifstream &inp)
	{
		neurons.clear();
		int n;
		inp >> prevLayerIndex;
		inp >> n;
		for(int i = 0;i < n;i++)
		{
			Neuron neuron(activFunc, inp);
			neurons.push_back(neuron);
		}
		setType();
	}
	// layerWts size must be equal to the count
	Layer(int count, ActivationFunction * activFunc, vector<vector<double> > & layerWts, int prevInd)
		: prevLayerIndex(prevInd)
	{
		for(int i = 0;i < count;i++)
		{
			Neuron neuron(activFunc, layerWts[i]);
			neurons.push_back(neuron);
		}
		setType();
	}
	// initializing the weights randomly
	Layer(int numInputs,int numNeurons, ActivationFunction * activFunc, int prevInd)
		: prevLayerIndex(prevInd)
	{
		std::random_device rd;
		std::mt19937 e2(rd());
		int inputSize = 1 + numInputs;
		vector<double> tmpWts(inputSize,0);
		std::normal_distribution<> normal_distWts(0, 1 / sqrt(inputSize));
		for (int i = 0; i < numNeurons; i++)
		{
			for (int j = 0; j < inputSize; j++)
			{
				tmpWts[j] = normal_distWts(e2);
			}
			Neuron neuron(activFunc, tmpWts);
			neurons.push_back(neuron);
		}
		setType();
	}
	virtual void operate(const vector<double>& inputs)
	{
		int neuronsSize = neurons.size();
#pragma omp parallel for
		for (int i = 0; i < neuronsSize; i++)
		{
			neurons[i].calcSummation(inputs);
		}
		// apply activations
		calcWeightedSums();
#pragma omp parallel for
		for (int i = 0; i < neuronsSize; i++)
		{
			neurons[i].calcActivation(weightedSums);
		}
		calcOutputs();
		calcDerivatives();
		calcWeights();
	}
	
	virtual void updateWeights(vector<vector<double> > & deltaWts, double eeta, double lambda, int totalSize)
	{
		//cout << "deltaWts.size() == neurons.size() " << deltaWts.size() << " " << neurons.size() << endl;
		assert(deltaWts.size() == neurons.size());
		
		for(unsigned int i = 0;i < neurons.size();i++)
		{
			neurons[i].updateWts(deltaWts[i],eeta,lambda,totalSize);
		}
		calcWeights();
	}
	virtual vector<double> getOutputs()
	{
		return outputs;
	}
	virtual vector<double> getWeightedSums()
	{
		return weightedSums;
	}
	virtual vector<double> getDerivatives()
	{
		return derivatives;
	}
	virtual vector<vector<double> > getWts()
	{
		return wts;
	}
	virtual int size()
	{
		return neurons.size();
	}
	int getPrevLayerIndex() const
	{
		return prevLayerIndex;
	}
	void setPrevLayerIndex(int ind)
	{
		prevLayerIndex = ind;
	}
	friend ostream& operator<<(ostream& oup, const Layer& layer);
};

ostream& operator<<(ostream& oup, const Layer& layer)
{
	oup << layer.getPrevLayerIndex() << endl;
	oup << layer.neurons.size() << endl;
	for(unsigned int i = 0;i < layer.neurons.size();i++)
	{
		oup << layer.neurons[i];
	}
	return oup;
}
