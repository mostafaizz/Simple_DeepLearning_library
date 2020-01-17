#pragma once

#include "layer.h"
#include "convolutionlayer.h"
#include "AveragePoolingLayer.h"
#include "MaxPoolingLayer.h"
#include "AbsLayer.h"
#include "NormalizationLayer.h"


class NeuralNetwork
{
private:
	int inputsCount; // number of inputs to this Neural Network
	vector<Layer*> layers; // first layer is the input layer, and last one is output layer
public:

	vector<vector<vector<double> > > getWts()
	{
		vector<vector<vector<double> > > wts;
		for(unsigned int i = 0;i < layers.size();i++)
		{
			wts.push_back(layers[i]->getWts());
		}
		return wts;
	}
	// save neural network to a file
	void save(string fileName)
	{
		ostringstream ostr;
		ostr << fileName;
		for(unsigned int i = 0;i < layers.size();i++)
		{
			ostr << "_" << layers[i]->size();
		}
		fileName = ostr.str();
		ofstream oup(fileName);
		oup << *this;
		oup.close();
		cout << "Save Complete!" << endl;
	}
	
	// load neural-network from file
	NeuralNetwork(string fileName, ActivationFunction * activFunc)
	{
		ifstream inp(fileName);
		inp >> inputsCount;
		int n;
		inp >> n;
		for(int i = 0;i < n;i++)
		{
			Layer *layer = new Layer(activFunc,inp);
			layers.push_back(layer);
		}
		inp.close();
		cout << "Load Complete!" << endl;
	}
	// layers count is a vector with the same length as number of layers 
	// and each element is equal to number of nodes in this layer
	// randomWeights: 1 means that the weights are initialized uniformly between [0,1], 0 means all ONES
	NeuralNetwork(int numInputs, vector<int> layersCount,ActivationFunction * activFunc, 
		int randomWeights = 1,double bias = 0)
	{
		this->inputsCount = numInputs;
		// Seed with a real random value, if available
		std::random_device rd;
		// Generate a normal distribution around that mean
		std::mt19937 e2(rd());
		//std::normal_distribution<> normal_dist(0, 1);
		for(int i = 0;i < layersCount.size();i++)
		{
			// if the input layer add weights from the input count
			// otherwise use the inputs from the outputs of the previous layer
			// add one in all cases for the bias term
			int layerInputSize = 1 + (i == 0 ? inputsCount : layersCount[i - 1]);	
			vector<vector<double> > layerWts(layersCount[i],vector<double>(layerInputSize,bias));
			std::normal_distribution<> normal_distWts(0, 1/sqrt(layerInputSize));
			for(int j = 0;j < layersCount[i];j++)
			{
							
				for(int k = 0;k < layerInputSize;k++)
				{
					if(randomWeights)
					{
						/*if(k == 0)
						{
							layerWts[j][k] = normal_dist(e2);
						}
						else*/
						{
							// init with N~(0,1/sqrt(n_in))
							layerWts[j][k] = normal_distWts(e2);
						}
					}
					else
					{
						layerWts[j][k] = 1;
					}
				}
			}
			Layer *obj = new Layer(layersCount[i], activFunc, layerWts, i - 1);
			layers.push_back(obj);
		}
		//layers[getLayerIndex(-1)].setBiasZeros();
	}
	
	// numInputs:	num of inputs to the network
	// layers:		is an output layers vector (to work with different types Neural Networks)
	NeuralNetwork(int numInputs, vector<Layer*> layers)
	{
		this->inputsCount = numInputs;
		this->layers = layers;
	}
	// copy constructor
	/*
	NeuralNetwork(const NeuralNetwork& inst)
	{
		inputsCount = inst.inputsCount;
		int layersSize = inst.layers.size();
		layers = vector<Layer*>(layersSize, 0);
		for (int l = 0; l < layersSize; l++)
		{
			switch (inst.layers[l]->getType())
			{
			case FullyConnected:
				layers[l] = new Layer(*inst.layers[l]);
				break;
			case Convolutional:
				layers[l] = new ConvolutionLayer(*(ConvolutionLayer*)(inst.layers[l]));
				break;
			case MaxPooling:
				layers[l] = new MaxPoolingLayer(*(MaxPoolingLayer*)inst.layers[l]);
				break;
			case AveragePooling:
				layers[l] = new AveragePoolingLayer(*(AveragePoolingLayer*)inst.layers[l]);
				break;
			case Abs:
				layers[l] = new AbsLayer(*(AbsLayer*)inst.layers[l]);
				break;
			case Normalization:
				layers[l] = new NormalizationLayer(*(NormalizationLayer*)inst.layers[l]);
				break;
			case SubNetwork:
				layers[l] = new SubNetworkLayer(*(SubNetworkLayer*)inst.layers[l]);
				break;
			default: // incuding subnetwork
				break;
			}
		}
	}
	*/

	// destructor
	virtual ~NeuralNetwork()
	{
		for (int i = 0; i < layers.size(); i++)
		{
			delete layers[i];
		}
	}
	
	// run the network
	void operate(const vector<double>& inputsGeneral)
	{
		// input layer
		layers[0]->operate(inputsGeneral);
		
		// iterate on the layers
		for(unsigned int l = 1;l < layers.size();l++)
		{
			vector<double> intermediate = layers[l - 1]->getOutputs();
			layers[l]->operate(intermediate);
		}
	}
	// get output after the activation function in layer i
	// if i is positive then this is the index directly
	// if i is negative then we subtract i from the vector size
	int getLayerIndex(int i)
	{
		int index = i;
		if(i < 0)
		{
			index = layers.size() + i;
		}
		return index;
	}	
	
	// get last layer output
	vector<double> getOutput()
	{
		return layers[getLayerIndex(-1)]->getOutputs();
	}
	// get weighted sum from last layer
	vector<double> getWeightedSum()
	{
		return layers[getLayerIndex(-1)]->getWeightedSums();
	}
	// get derivatives from last layer
	vector<double> getDerivatives()
	{
		return layers[getLayerIndex(-1)]->getDerivatives();
	}
	// backprobagation Algorithm
	// i is the index of the sample
	// costFunc = 0 if quadratic and 1 if cross-entropy
	vector<vector<vector<double> > > backProp(const vector<double> & trainData, 
											  const vector<short> & label,int CostFunc = 1)
	{
		vector<vector<double> > delta(layers.size());
		// calcualte the output
		operate(trainData);
		// calculate errors in the last layer
		vector<double> a_L = layers[getLayerIndex(-1)]->getOutputs(); // last location
		vector<double> nabla_a_C = sub(a_L, label);
		
		// calculate deltas
		if(CostFunc == 0)
		{
			// calculate sigmoid derivatives
			vector<double> derivative_L = layers[getLayerIndex(-1)]->getDerivatives(); // last index
			// quadratic
			delta[getLayerIndex(-1)] = mul(nabla_a_C, derivative_L);
		}
		else if(CostFunc == 1)
		{
			// entropy
			delta[getLayerIndex(-1)] = nabla_a_C;
		}
		// then calculate the deltas for the previous layers
		int layersSize = layers.size();
		for(int l = layersSize - 1; l > 0;l--)
		{
			int prevIndex = layers[l]->getPrevLayerIndex();
			switch (layers[prevIndex]->getType())
			{
			case FullyConnected:
			{
				vector<double> tmp = mulMatVec(layers[l]->getWts(), delta[l]);
				delta[prevIndex] =
					mul(tmp, layers[prevIndex]->getDerivatives());
			}
			break;
			default:
			{
				// not ready yet
				//delta[prevIndex] = vector<double>(layers[l]->size() + 1, 0);
			}
			break;
			}
			
		}
		// then calculate the partial derivatives of C
		// delta_C/delta_w^{l}_{jk} = a^{l−1}_{k} delta^{l}_{j} and delta_C/delta_b^{l}_{j} = delta^{l}_{j}.
		// dC is the dC for this particular sample and it is 3D
		// the dimensions are LxJxK
		// L represents number of layers
		// J represents the neuron index inside the layer
		// K represents the input index inside the neuron
		// The biases are at K = 0 and the weights are shifted by 1 
		// (not actually shifted but for the purpose of using the same indeces as the book I say shifted)
		vector<vector<vector<double> > > dC;
		// the first layer
		// J x K
		for(int l = 0;l < layersSize;l++)
		{
			// J x K
			vector<vector<double> > dClayer;
			int layersLSize = layers[l]->size();
			for(int j = 0;j < layersLSize;j++)
			{
				int inputSize;
				// other layers
				int prevLayerInd = layers[l]->getPrevLayerIndex();
				if(prevLayerInd < 0)
				{
					// input layer
					inputSize = this->inputsCount;
				}					
				else
				{
					inputSize = layers[prevLayerInd]->size();
				}
				// check for fully connected layers
				// this is the only type that has valid training now
				vector<double> dCneuron(inputSize + 1, 0);
				if (layers[l]->getType() == FullyConnected)
				{
					//
					dCneuron[0] = delta[l][j];
					vector<double> refData;
					if (prevLayerInd >= 0)
					{
						refData = layers[prevLayerInd]->getOutputs();
					}
					else
					{
						refData = trainData;
					}

					for (int k = 0; k < inputSize; k++)
					{
						dCneuron[k + 1] = refData[k] * delta[l][j];
					}
				}
				dClayer.push_back(dCneuron);
			}
			dC.push_back(dClayer);
		}
		return dC;
	}

	void updateMiniBatch(vector<int>& miniBatch,const vector<vector<double> >& trainData, 
		const vector<vector<short> >& labels, double eeta,double lambda, int totalSize)
	{
		vector<vector<vector<double> > > dC;
		int miniBatchSize = miniBatch.size();
		for(int i = 0;i < miniBatchSize;i++)
		{
			if(i == 0)
			{
				dC = backProp(trainData[miniBatch[i]],labels[miniBatch[i]]);
			}
			else
			{
				vector<vector<vector<double> > > temp = backProp(trainData[miniBatch[i]],labels[miniBatch[i]]);
				int dCSize = dC.size();
				for(int l = 0;l < dCSize;l++)
				{
					for(unsigned int j = 0;j < dC[l].size();j++)
					{
						dC[l][j] = add(dC[l][j], temp[l][j]);
					}
				}
			}
		}
		//vector<vector<vector<double> > > eeta_dC = dC;
		// average updates
		double batchSize = miniBatch.size();
		for(unsigned int l = 0;l < dC.size();l++)
		{
			for(unsigned int j = 0;j < dC[l].size();j++)
			{
				for(unsigned int k = 0;k < dC[l][j].size();k++)
				{
					dC[l][j][k] = (dC[l][j][k] / batchSize);
				}
			}
			// then update the weights
			layers[l]->updateWeights(dC[l], eeta,lambda,totalSize);
		}
	}
	// Stochastic Gradient Descent
	void SGD(const vector<vector<double> >& trainData, const vector<vector<short> >& labels,
		int countTrain, int miniBatchSize, double eeta, int iterations, double lambda = 1, 
		int validate = 0,string saveName = "NN",int displayProgress = 1)
	{
		if (displayProgress)
		{
			cout << "Training with:" << endl << countTrain << " Training Samples" << endl;
			cout << miniBatchSize << " Mini Batch Size" << endl;
			cout << eeta << " Learning Rate" << endl;
			cout << iterations << " Iterations" << endl;
			cout << lambda << " Regularization Rate" << endl;
		}
		vector<int> trainingSet;
		vector<int> validationSet;
		int portionLimit = countTrain / labels[0].size();
		if (countTrain == trainData.size())
		{
			portionLimit = countTrain;
		}
		vector<int> counts(labels[0].size(), 0);
		for (unsigned int i = 0; i < trainData.size(); i++)
		{
			int ind = distance(labels[i].begin(), max_element(labels[i].begin(), labels[i].end()));
			if (counts[ind] < portionLimit)
			{
				counts[ind]++;
				countTrain--;
				trainingSet.push_back(i);
			}
			else
			{
				validationSet.push_back(i);
			}
		}

		//
		for (int epoch = 0; epoch < iterations; epoch++)
		{
			// obtain a time-based seed:
			unsigned seed = 0;//std::chrono::system_clock::now().time_since_epoch().count();

			shuffle(trainingSet.begin(), trainingSet.end(), std::default_random_engine(seed));
			for (unsigned int i = 0; i < trainingSet.size(); i += miniBatchSize)
			{
				vector<int> subIndeces(trainingSet.begin() + i, trainingSet.begin() + i + miniBatchSize);
				updateMiniBatch(subIndeces, trainData, labels, eeta, lambda, trainingSet.size());
				if (displayProgress)
				{
					if (!(i % 2000))
					{
						printf(".");
						cout.flush();
					}
					/*
					if (!(i % 10000) && validate)
					{
						test(trainData, labels, validationSet);
					}*/
				}
			}
			if (displayProgress)
			{
				printf("Epoch %d is done...!\n", epoch);
			}
			cout.flush();
			if (validate)
			{
				test(trainData, labels, validationSet);
			}
		}
		if (saveName.length() != 0)
		{
			ostringstream ostr;
			ostr.precision(6);
			ostr << saveName << "_" << countTrain << "_" << miniBatchSize << "_";
			ostr << fixed << eeta << "_" << iterations << "_" << fixed << lambda << "_";
			save(ostr.str());
		}
		
	}
	double test(const vector<vector<double> >& data, const vector<vector<short> >& labels,vector<int>& indeces)
	{
		countTests = vector<int>(10, 0);
		actualLabels = vector<int>(10, 0);
		double ratio = 0;
		int count = 0;
		for(unsigned int i = 0;i < indeces.size();i++)
		{
			if(testOneImage(data[indeces[i]],labels[indeces[i]]) > 0)
			{
				count++;
			}
		}
		//cout << endl;
		/*for (int i = 0; i < 10; i++)
		{
			cout << i << " -\t" << countTests[i] << "/" << actualLabels[i] << " = " << (countTests[i] * 100.0 / actualLabels[i]) << endl;
		}*/
		ratio = (count*1.0) / indeces.size();
		printf("%d/%d = %0.2f%%\n", count, indeces.size(), ratio * 100);
		return ratio;
	}
	vector<int> countTests, actualLabels;
	
	double test(const vector<vector<double> >& data, const vector<vector<short> >& labels)
	{
		countTests = vector<int>(labels[0].size(),0);
		actualLabels = vector<int>(labels[0].size(),0);
		double ratio = 0;
		int count = 0;
		for(unsigned int i = 0;i < data.size();i++)
		{
			if(testOneImage(data[i],labels[i]) > 0)
			{
				count++;
			}
		}
		//cout << endl;
		for(int i = 0;i < labels[0].size();i++)
		{
			cout << i << " -\t" << countTests[i] << "/" << actualLabels[i] << " = " << (countTests[i] * 100.0 / actualLabels[i]) << endl;
		}
		ratio = (count*1.0) / data.size();
		printf("%d/%d = %0.2f%%\n", count, data.size(), ratio*100);
		return ratio;
	}
	int testOneImage(const vector<double> & sample, const vector<short> & label)
	{
		int maxInd = getOutputOneImage(sample);
		//
		int correctInd = distance(label.begin(), max_element(label.begin(), label.end()));
		if ((int)(actualLabels.size()) > correctInd)
		{
			actualLabels[correctInd]++;
		}
		if(correctInd == maxInd)
		{
			if ((int)(countTests.size()) > correctInd)
			{
				countTests[correctInd]++;
			}
			return 1;
		}
		return -1;
	}
	int getOutputOneImage(const vector<double> & sample)
	{
		operate(sample);
		vector<double> a = getOutput();
		return distance(a.begin(), max_element(a.begin(), a.end()));
	}
	int getSize()
	{
		return layers[getLayerIndex(-1)]->size();
	}
	friend ostream& operator<<(ostream& oup, const NeuralNetwork& nn);
};
ostream& operator<<(ostream& oup, const NeuralNetwork& nn)
{
	oup << nn.inputsCount << endl;
	oup << nn.layers.size() << endl;
	for(unsigned int i = 0;i < nn.layers.size();i++)
	{
		oup << nn.layers[i];
	}
	return oup;
}
