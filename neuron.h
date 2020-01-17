#pragma once

#include "activation.h"
#include "operation.h"

class Neuron
{
private:
	double derivative; // the derivative
	double output; // after applying sigmoid
	double z; // weighted sum
	ActivationFunction * activFunc;
	vector<double> wts;
public:
	// Constructor
	Neuron(ActivationFunction *activFunc, vector<double>& wts)
	{
		setActivationFunction(activFunc);
		setWeights(wts);
	}
	// setters and getters
	void setActivationFunction(ActivationFunction * activFuncPtr)
	{
		activFunc = activFuncPtr;
	}
	void setWeights(vector<double> & wts)
	{
		this->wts = wts;
	}
	vector<double> getWeights()
	{
		return vector<double>(wts.begin() + 1, wts.end());
	}
	void setBias(double val)
	{
		wts[0] = val;
	}
	double getBias()
	{
		return wts[0];
	}
	double getOuput()
	{
		return output;
	}
	double getWeightedSum()
	{
		return z;
	}
	double getDerivative()
	{
		return derivative;
	}
	// methods
	template<typename T>
	void calcSummation(const vector<T> & inputVector)
	{
		z = wts[0];
		int inputSize = wts.size();
		for(int i = 1;i < inputSize;i++)
		{
			z += wts[i] * inputVector[i - 1];
		}
	}
	void calcActivation(vector<double> & allSummations)
	{
		output = activFunc->applyFunction(z, allSummations);
		derivative = activFunc->applyDerivative(z);
	}
	void updateWts(vector<double>& dCWts, double eeta, double lambda, int totalSize)
	{
		//cout << wts.size() << " = " << dCWts.size() << endl;
		assert(wts.size() == dCWts.size());
		for (unsigned int i = 0; i < wts.size(); i++)
		{
			wts[i] = (1 - lambda*eeta / totalSize) * wts[i] - eeta * dCWts[i];
		}
	}
	// load
	Neuron(ActivationFunction *activFunc,ifstream &inp)
	{
		setActivationFunction(activFunc);
		int n;
		inp >> n;
		wts.resize(n,0);
		for(int i = 0;i < n;i++)
		{
			inp >> wts[i];
		}
	}
	friend ostream& operator<<(ostream& oup, const Neuron& neuron);
};
ostream& operator<<(ostream& oup, const Neuron& neuron)
{
	oup << neuron.wts.size();
	for(unsigned int i = 0;i < neuron.wts.size();i++)
	{
		oup << " " << neuron.wts[i];
	}
	oup << endl;
	return oup;
}
