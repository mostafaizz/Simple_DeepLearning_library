#pragma once
#include "convolutionlayer.h"
class AbsLayer :
	public ConvolutionLayer
{
public:
	virtual void setType(){ type = Abs; };
	AbsLayer(int imgCols, int imgRows, int prevIndex)
		: ConvolutionLayer(0, imgCols, imgRows, 1, 1, 1, prevIndex)
	{
		setType();
	}
	virtual void operate(const vector<double>& inputs)
	{
		weightedSums = vector<double>(imgRows * imgCols, -10000000);
		outputs = weightedSums;
		derivatives = weightedSums;
#pragma omp parallel for
		for (int i = 0; i < inputs.size(); i++)
		{
			weightedSums[i] = std::abs(inputs[i]);
			outputs[i] = weightedSums[i];
			derivatives[i] = 1;
		}
	}
};