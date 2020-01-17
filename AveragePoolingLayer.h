#pragma once
#include "convolutionlayer.h"
class AveragePoolingLayer :
	public ConvolutionLayer
{
public:
	virtual void setType(){ type = AveragePooling; };
	AveragePoolingLayer(int imgCols, int imgRows, int cols, int rows, int stride, int prevIndex) 
		: ConvolutionLayer(0,imgCols,imgRows,cols,rows,stride, prevIndex)
	{
		setType();
	}
	virtual void operate(const vector<double>& inputs)
	{
		weightedSums = vector<double>((((imgRows - convRows) / stride) + 1) * (((imgCols - convCols) / stride) + 1), 0);
		outputs = weightedSums;
		derivatives = weightedSums;
		int index = 0;
		for (int r = 0; r < imgRows - convRows + 1; r += stride)
		{
//#pragma omp parallel for
			for (int c = 0; c < imgCols - convCols + 1; c += stride)
			{
				//int index = r * (((imgCols - convCols) / stride) + 1) + c;
				for (int rconv = 0; rconv < convRows; rconv++)
				{
					for (int cconv = 0; cconv < convCols; cconv++)
					{
						//printf("weightedSums[%d] += inputs[%d]\n", index, (rconv + r) * imgCols + (cconv + c));
						weightedSums[index] += inputs[(rconv + r) * imgCols + (cconv + c)];
					}
				}
				//system("PAUSE");
				outputs[index] = weightedSums[index] = weightedSums[index] / (convCols * convRows);
				derivatives[index++] = 1;
			}
		}
	}
};

