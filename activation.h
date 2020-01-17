#pragma once

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <map>
#include <random>
#include <chrono>       // std::chrono::system_clock
#include <cassert>
#include <sstream>

using namespace std;

// Abstract class for the activation function
class ActivationFunction
{
public:
	virtual double applyFunction(double,const vector<double>&) = 0;
	virtual double applyDerivative(double){return 1;}
};

class StepFunction : public ActivationFunction
{
private:
	double treshold;
	double outputLow;
	double outputHigh;
public:
	StepFunction(double tr = 0,double low = 0,double hi = 1):treshold(tr),outputLow(low),outputHigh(hi)
	{}
	double applyFunction(double input,const vector<double>& dummy)
	{
		if(input < treshold)
		{
			return outputLow;
		}
		return outputHigh;
	}
};

class SigmoidFunction : public ActivationFunction
{
private:
	double func(double &input)
	{
		return 1.0 / (1.0 + exp(- input));
	}
public:
	double applyFunction(double input,const vector<double>& dummy)
	{
		return func(input);
	}
	double applyDerivative(double input)
	{
		return func(input) * (1 - func(input));
	}	
};

class SoftMaxFunction : public ActivationFunction
{
public:
	double applyFunction(double input,const vector<double>& allSummations)
	{
		double sum = 0;
		for (int i = 0; i < allSummations.size(); i++)
		{
			sum += exp(allSummations[i]);
		}
		return (exp(input) / sum);
	}
};

class ReLUFunction : public ActivationFunction
{
public:
	double applyFunction(double input, const vector<double>& dummy)
	{
		if (input < 0)
		{
			return 0;
		}
		return input;
	}
	double applyDerivative(double input)
	{
		if (input < 0)
		{
			return 0;
		}
		return 1;
	}
};

class LinearFunction : public ActivationFunction
{
public:
	double applyFunction(double input, const vector<double>& dummy)
	{
		return input;
	}
	double applyDerivative(double input)
	{
		return 1;
	}
};
