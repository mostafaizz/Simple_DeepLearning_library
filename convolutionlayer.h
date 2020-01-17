#pragma once

#include "layer.h"

/// trial to implement convolution layer
class ConvolutionLayer : public Layer
{
protected:
	vector<double> outputs;
	vector<double> weightedSums; // z
	vector<double> derivatives; // derivative of the activation function at z
	vector<double> wts; // all weights in this layer
	ActivationFunction * activFunc;
	//double bias;
	int imgCols, imgRows;
	int convCols, convRows;
	int stride;
	//virtual void calcOutputs(){}
	//virtual void calcWeightedSums(){}
	//virtual void calcDerivatives(){}
	//virtual void calcWeights(){}
	
public:
	virtual void setType(){ 
		type = Convolutional; 
	};
	// load
	ConvolutionLayer(ActivationFunction * activFunc, ifstream &inp) : Layer(0)
	{
		this->activFunc = activFunc;

		inp >> imgCols;
		inp >> imgRows;
		inp >> convCols;
		inp >> convRows;
		inp >> stride;
		inp >> prevLayerIndex;
		outputs = vector<double>((((imgRows - convRows) / stride) + 1) * (((imgCols - convCols) / stride) + 1), 0);
		weightedSums = outputs;
		derivatives = outputs;
		wts = vector<double>(convRows * convCols + 1, 0);
		for (int i = 0; i < convRows * convCols; i++)
		{
			inp >> wts[i + 1];
		}
		setType();
	}
	// layerWts size must be equal to the count
	ConvolutionLayer(ActivationFunction * activFunc, vector<double> & layerWts,
		int imgCols, int imgRows, int convCols, int convRows, int stride, int prevIndex) : Layer(prevIndex)
	{
		this->activFunc = activFunc;
		this->imgCols = imgCols;
		this->imgRows = imgRows;
		this->convCols = convCols;
		this->convRows = convRows;
		this->stride = stride;
		//int index = 0;
		wts = layerWts;
		setType();
	}
	// don't forget the weights 
	ConvolutionLayer(ActivationFunction * activFunc, 
		int imgCols, int imgRows, int convCols, int convRows, int stride, int prevIndex) 
		: Layer(prevIndex)
	{
		this->activFunc = activFunc;
		this->imgCols = imgCols;
		this->imgRows = imgRows;
		this->convCols = convCols;
		this->convRows = convRows;
		this->stride = stride;
		//int index = 0;
		wts = vector<double>(convRows * convCols + 1, 0);
		std::random_device rd;
		std::mt19937 e2(rd());
		std::normal_distribution<> normal_distWts(0, 1 / sqrt(convCols * convRows));
		for (int i = 0; i < wts.size(); i++)
		{
			wts[i] = normal_distWts(e2);
		}
		setType();
	}
	ConvolutionLayer(const ConvolutionLayer& inst) : Layer(inst.prevLayerIndex)
	{
		this->activFunc = inst.activFunc;
		this->imgCols = inst.imgCols;
		this->imgRows = inst.imgRows;
		this->convCols = inst.convCols;
		this->convRows = inst.convRows;
		this->stride = inst.stride;
		//int index = 0;
		wts = vector<double>(convRows * convCols + 1, 0);
		std::random_device rd;
		std::mt19937 e2(rd());
		std::normal_distribution<> normal_distWts(0, 1 / sqrt(convCols * convRows));
		for (int i = 0; i < wts.size(); i++)
		{
			wts[i] = normal_distWts(e2);
		}
		setType();
	}
	virtual void operate(const vector<double>& inputs)
	{
		weightedSums = vector<double>((((imgRows - convRows) / stride) + 1) * (((imgCols - convCols) / stride) + 1), 0);
		for (int r = 0; r < imgRows - convRows + 1; r += stride)
		{
#pragma omp parallel for
			for (int c = 0; c < imgCols - convCols + 1; c += stride)
			{
				int index =  r * (((imgCols - convCols) / stride) + 1) + c;
				for (int rconv = 0; rconv < convRows; rconv++)
				{
					for (int cconv = 0; cconv < convCols; cconv++)
					{
						//printf("weightedSums[%d] += wts[%d] * inputs[%d]\n", r * (imgCols - convCols + 1) + c,
						//	rconv * convCols + cconv + 1, (rconv + r) * imgCols + (cconv + c));
						weightedSums[index] += wts[rconv * convCols + cconv + 1] * inputs[(rconv + r) * imgCols + (cconv + c)];
					}
				}
			}
		}
		//puts("DONE");
		//system("PAUSE");
		outputs.clear();
		derivatives.clear();
		// apply activations
		for (unsigned int i = 0; i < weightedSums.size(); i++)
		{
			outputs.push_back(activFunc->applyFunction(weightedSums[i], inputs));
			derivatives.push_back(activFunc->applyDerivative(weightedSums[i]));
		}
	}

	virtual void updateWeights(vector<vector<double> > & deltaWts, double eeta, double lambda, int totalSize)
	{
		// nothing yet
		return;
		assert(deltaWts.size() == neurons.size());

		int index = 0;
		for (int i = 0; i < convRows * convCols + 1; i++)
		{
			wts[i] += deltaWts[0][i];
		}
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
		vector<vector<double> > ret;
		ret.push_back(wts);
		return ret;
	}
	virtual int size()
	{
		return (((imgRows - convRows) / stride) + 1) * (((imgCols - convCols) / stride) + 1);
	}
	friend ostream& operator<<(ostream& oup, const ConvolutionLayer& layer);
};

ostream& operator<<(ostream& oup, const ConvolutionLayer& layer)
{
	oup << " " << layer.imgCols;
	oup << " " << layer.imgRows;
	oup << " " << layer.convCols;
	oup << " " << layer.convRows;
	oup << " " << layer.stride;
	//oup << " " << layer.bias;
	//
	for (int i = 0; i < layer.wts.size(); i++)
	{
		oup << " " << layer.wts[i];
	}
	return oup;
}
