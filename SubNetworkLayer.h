#pragma once
#include "neuralnetwork.h"
class SubNetworkLayer :
	public Layer
{
private:
private:
	int inputsCount; // number of inputs to this Neural Network
	vector< vector<Layer*> > parallelLayerPaths; // first layer is the input layer, and last one is output layer
public:
	virtual void setType(){ type = SubNetwork; };
	SubNetworkLayer(const vector<Layer*>& layers,int parallelPaths, int prevIndex)
		: Layer(prevIndex)
	{
		setType();
		for (int i = 0; i < parallelPaths; i++)
		{
			parallelLayerPaths.push_back(vector<Layer*>(layers.size(), 0));
			for (int l = 0; l < layers.size(); l++)
			{
				Layer * layer = 0;
				switch (layers[l]->getType())
				{
				case FullyConnected:
					parallelLayerPaths[i][l] = new Layer(*layers[l]);
					break;
				case Convolutional:
					parallelLayerPaths[i][l] = new ConvolutionLayer(*(ConvolutionLayer*)(layers[l]));
					break;
				case MaxPooling:
					parallelLayerPaths[i][l] = new MaxPoolingLayer(*(MaxPoolingLayer*)layers[l]);
					break;
				case AveragePooling:
					parallelLayerPaths[i][l] = new AveragePoolingLayer(*(AveragePoolingLayer*)layers[l]);
					break;
				case Abs:
					parallelLayerPaths[i][l] = new AbsLayer(*(AbsLayer*)layers[l]);
					break;
				case Normalization:
					parallelLayerPaths[i][l] = new NormalizationLayer(*(NormalizationLayer*)layers[l]);
					break;
				case SubNetwork:
					parallelLayerPaths[i][l] = new SubNetworkLayer(*(SubNetworkLayer*)layers[l]);
					break;
				default: // incuding subnetwork
					break;
				}
			}
		}
	}
	void operate(const vector<Layer*>& layers,const vector<double>& inputsGeneral)
	{
		// input layer
		layers[0]->operate(inputsGeneral);

		// iterate on the layers
		for (unsigned int l = 1; l < layers.size(); l++)
		{
			vector<double> intermediate = layers[l - 1]->getOutputs();
			layers[l]->operate(intermediate);
		}
	}
	virtual void operate(const vector<double>& inputs)
	{
		outputs.clear();
		weightedSums.clear();
		derivatives.clear();
		wts.clear();
		for (int i = 0; i < parallelLayerPaths.size(); i++)
		{
			operate(parallelLayerPaths[i], inputs);
			//
			vector<double> tmpOutput = parallelLayerPaths[i][parallelLayerPaths[i].size() - 1]->getOutputs();
			vector<double> tmpWtSum = parallelLayerPaths[i][parallelLayerPaths[i].size() - 1]->getWeightedSums();
			vector<double> tmpDeriv = parallelLayerPaths[i][parallelLayerPaths[i].size() - 1]->getDerivatives();
			//
			outputs.insert(outputs.end(), tmpOutput.begin(), tmpOutput.end());
			weightedSums.insert(weightedSums.end(), tmpWtSum.begin(), tmpWtSum.end());
			derivatives.insert(derivatives.end(), tmpDeriv.begin(), tmpDeriv.end());
		}
	}
	virtual void updateWeights(vector<vector<double> > & deltaWts, double eeta, double lambda, int totalSize)
	{
		// DO NOTHING FOR NOW
		/*
		assert(deltaWts.size() == neurons.size());

		for (unsigned int i = 0; i < neurons.size(); i++)
		{
			neurons[i].updateWts(deltaWts[i], eeta, lambda, totalSize);
		}
		calcWeights();
		*/
	}
	
	virtual int size()
	{
		int totalSize = 0;
		for (int i = 0; i < parallelLayerPaths.size(); i++)
		{
			totalSize += parallelLayerPaths[i][parallelLayerPaths[i].size() - 1]->size();
		}
		return totalSize;
	}

};