#pragma once
#include "convolutionlayer.h"
class NormalizationLayer :
	public ConvolutionLayer
{
public:
	virtual void setType(){ type = Normalization; };
	NormalizationLayer(int imgCols, int imgRows, int width, int prevIndex)
		: ConvolutionLayer(0, imgCols, imgRows, width, width, 1, prevIndex)
	{
		setType();
		// Create the Gaussian Filter
		double sigma = 1;
		double mean = width / 2;
		double sum = 0.0; // For accumulating the kernel values
		for (int x = 0; x < width; ++x)
		{
			for (int y = 0; y < width; ++y)
			{
				wts[x * width + y] = exp(-0.5 * (pow((x - mean) / sigma, 2.0) + pow((y - mean) / sigma, 2.0)))
					/ (2 * 3.14159265359 * sigma * sigma);

				// Accumulate the kernel values
				sum += wts[x * width + y];
			}
		}
		// Normalize the kernel
		for (int x = 0; x < width; ++x)
		{
			for (int y = 0; y < width; ++y)
			{
				wts[x * width + y] /= sum;
			}
		}
		
		outputs = vector<double>(imgRows * imgCols, 0);
		weightedSums = outputs;
		derivatives = outputs;
	}
	virtual void operate(const vector<double>& inputs)
	{
		vector<double> v = vector<double>(inputs.size(), 0);
		//cout << inputs.size() << endl;
		//pause();
		vector<double> sigma_jk = v;
		
		for (int r = 0; r < imgRows; r++)
		{
#pragma omp parallel for
			for (int c = 0; c < imgCols; c++)
			{
				int index = r * imgCols + c;
				for (int rconv = 0; rconv < convRows; rconv++)
				{
					int rr = (rconv + r - convRows / 2);
					if (rr >= 0 && rr < imgRows)
					{
						// if the row outside the image
						for (int cconv = 0; cconv < convCols; cconv++)
						{
							int cc = (cconv + c - convCols / 2);
							if (cc >= 0 && cc < imgCols)
							{
								// if the column outside the image
								//printf("v[%d] += wts[%d] * inputs[%d]\n",
								//	index, rconv * convCols + cconv + 1,
								//	(rconv + r - convRows/2) * imgCols + (cconv + c - convCols/2));
								v[index] += wts[rconv * convCols + cconv + 1] * inputs[rr * imgCols + cc];
							}
						}
					}
				}
				//cout << index << " almost done !!" << endl;
				v[index] = inputs[index] - v[index];
			}
		}
		// calculate sigma_jk
		double sigma_jk_mean = 0;
		for (int r = 0; r < imgRows; r++)
		{
#pragma omp parallel for
			for (int c = 0; c < imgCols; c++)
			{
				int index = r * imgCols + c;
				for (int rconv = 0; rconv < convRows; rconv++)
				{
					int rr = (rconv + r - convRows / 2);
					if (rr >= 0 && rr < imgRows)
					{
						// if the row outside the image
						for (int cconv = 0; cconv < convCols; cconv++)
						{
							int cc = (cconv + c - convCols / 2);
							if (cc >= 0 && cc < imgCols)
							{
								// if the column inside the image
								//printf("sigma_jk[%d] += wts[%d] * v[%d] * v[%d]\n",
								//	index, rconv * convCols + cconv + 1,
								//	(rconv + r - convRows/2) * imgCols + (cconv + c - convCols/2),
								//	(rconv + r - convRows/2) * imgCols + (cconv + c - convCols/2));
								int ind = rr * imgCols + cc;
								sigma_jk[index] += wts[rconv * convCols + cconv + 1] * v[ind] * v[ind];
							}
						}
					}
				}
				sigma_jk[index] = std::sqrt(sigma_jk[index]);
				sigma_jk_mean += sigma_jk[index];
				index++;
			}
		}
		// get the mean value
		sigma_jk_mean /= (imgCols * imgRows);
		//puts("DONE");
		//system("PAUSE");
		weightedSums.clear();
		outputs.clear();
		derivatives.clear();
		// apply activations
		for (unsigned int i = 0; i < v.size(); i++)
		{
			double y = v[i] / std::max(sigma_jk_mean,sigma_jk[i]);
			weightedSums.push_back(y);
			outputs.push_back(y);
			derivatives.push_back(1);
		}
	}

};

