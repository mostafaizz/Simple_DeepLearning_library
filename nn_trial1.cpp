#include "mnist.h"
#include "ensemblenn.h"
#include "SubNetworkLayer.h"

void testMINST()
{
	//freopen("VS.txt", "w", stdout);
	MINST::readTestLabels("MINST/t10k-labels.idx1-ubyte");
	MINST::readTestImages("MINST/t10k-images.idx3-ubyte");
	//return;
	//Neuron perceptron;
	StepFunction step;
	SigmoidFunction sig;
	
	//NeuralNetwork nn("NN_5iter__0_10_0.001000_100_5.000000__50_10",&sig);
	NeuralNetwork nn("Ensemble2_50000_1000_0.001000_48_5.000000__50_10",&sig);
	///NeuralNetwork nn("Ensemble1_50000_10_0.001000_50_5.000000__50_10",&sig);
	///NeuralNetwork nn("Ensemble2_50000_10_0.001000_50_5.000000__50_10",&sig);
	///NeuralNetwork nn("Ensemble3_50000_10_0.001000_50_5.000000__50_10",&sig);
	///NeuralNetwork nn("Ensemble4_50000_10_0.001000_50_5.000000__50_10",&sig);
	
	
	nn.test(MINST::testImages, MINST::testLabels);
	//cout << nn.getOutputOneImage(MINST::testImages[0]) << endl;
}
#ifdef _WIN32
extern "C" __declspec (dllexport) 
#endif
int getNumber(int* data)
{
	SigmoidFunction sig;

	NeuralNetwork nn("NN_5iter__0_10_0.001000_100_5.000000__50_10", &sig);
	vector<double> sample;
	for (int i = 0; i < 784; i++)
	{
		sample.push_back(data[i]);
	}
	return nn.getOutputOneImage(sample);
}


void gridSearhMINST()
{
	//freopen("Iter100_nodes100.log", "w", stdout);

	MINST::readTrainLabels("MINST/train-labels.idx1-ubyte");
	MINST::readTestLabels("MINST/t10k-labels.idx1-ubyte");
	MINST::readTrainImages("MINST/train-images.idx3-ubyte");
	MINST::readTestImages("MINST/t10k-images.idx3-ubyte");
	SigmoidFunction sig;
	SoftMaxFunction smax;
	//for (int hiddenLayer = 50; hiddenLayer <= 50; hiddenLayer += 10)
	{
		double eeta = 0.001;
		//for (double eeta = 0.0001; eeta < 10; eeta *= 10)
		{
			vector<Layer*> layers(6);
			//Layer l1(MINST::trainImages[0].size(), hiddenLayer, &sig, -1);
			layers[0] = new ConvolutionLayer(&sig, 28, 28, 5, 5, 1, -1);
			layers[1] = new AbsLayer(24, 24, 0);
			layers[2] = new NormalizationLayer(24, 24, 9, 1);
			layers[3] = new MaxPoolingLayer(24, 24, 2, 2, 2, 2);
			layers[4] = new Layer(144, 200, &sig, 3);
			layers[5] = new Layer(200, 10, &sig, 4);
			
			//NeuralNetwork nn(MINST::trainImages[0].size(), layers, &sig, 1);
			NeuralNetwork nn(MINST::trainImages[0].size(), layers);

			nn.SGD(MINST::trainImages, MINST::trainLabels, 50000, 10, eeta, 5, 5.0, 1, "NN_conv_");
			nn.test(MINST::testImages, MINST::testLabels);
		}
	}
}

void testConvolutioanl()
{
	//freopen("Iter100_nodes100.log", "w", stdout);

	MINST::readTrainLabels("MINST/train-labels.idx1-ubyte");
	MINST::readTestLabels("MINST/t10k-labels.idx1-ubyte");
	MINST::readTrainImages("MINST/train-images.idx3-ubyte");
	MINST::readTestImages("MINST/t10k-images.idx3-ubyte");
	SigmoidFunction sig;
	SoftMaxFunction smax;
	//for (int hiddenLayer = 50; hiddenLayer <= 50; hiddenLayer += 10)
	{
		double eeta = 0.001;
		//for (double eeta = 0.0001; eeta < 10; eeta *= 10)
		{
			vector<Layer*> subLayers1(4);
			
			subLayers1[0] = new ConvolutionLayer(&sig, 28, 28, 5, 5, 1, -1);
			subLayers1[1] = new AbsLayer(24, 24, 0);
			subLayers1[2] = new NormalizationLayer(24, 24, 9, 1);
			subLayers1[3] = new AveragePoolingLayer(24, 24, 2, 2, 2, 2);
			//NeuralNetwork sub_nn1(MINST::trainLabels[0].size(), subLayers1);
			
			vector<Layer*> subLayers2(4);
			
			subLayers2[0] = new ConvolutionLayer(&sig, 48, 48, 5, 5, 1, -1);
			subLayers2[1] = new AbsLayer(20, 20, 0);
			subLayers2[2] = new NormalizationLayer(20, 20, 9, 1);
			subLayers2[3] = new AveragePoolingLayer(20, 20, 2, 2, 2, 2);
			//cout << "sub_nn1.getSize() = " << sub_nn1.getSize() << endl;
			//NeuralNetwork sub_nn2(sub_nn1.getSize(), subLayers2);
			
			vector<Layer*> layers(3);
			layers[0] = new SubNetworkLayer(subLayers1, 16, -1);
			layers[0] = new SubNetworkLayer(subLayers1, 16, -1);
			layers[1] = new SubNetworkLayer(subLayers2, 64, 0);
			//cout << "layers[1]->size() = " << layers[1]->size() << endl;
			layers[2] = new Layer(layers[1]->size(), 200, &sig, 1);
			layers[3] = new Layer(layers[2]->size(), 10, &sig, 1);

			//NeuralNetwork nn(MINST::trainImages[0].size(), layers, &sig, 1);
			NeuralNetwork nn(MINST::trainImages[0].size(), layers);

			nn.SGD(MINST::trainImages, MINST::trainLabels, 50000, 10, eeta, 5, 5.0, 1, "NN_conv_");
			nn.test(MINST::testImages, MINST::testLabels);
		}
	}
}

void testConvolutioanl_1layer()
{
	//freopen("Iter100_nodes100.log", "w", stdout);

	MINST::readTrainLabels("MINST/train-labels.idx1-ubyte");
	MINST::readTestLabels("MINST/t10k-labels.idx1-ubyte");
	MINST::readTrainImages("MINST/train-images.idx3-ubyte");
	MINST::readTestImages("MINST/t10k-images.idx3-ubyte");
	SigmoidFunction sig;
	SoftMaxFunction smax;
	ReLUFunction relu;
	LinearFunction linear;
	//for (int hiddenLayer = 50; hiddenLayer <= 50; hiddenLayer += 10)
	{
		double eeta = 0.001;
		//for (double eeta = 0.0001; eeta < 10; eeta *= 10)
		{
			vector<Layer*> subLayers1;

			//subLayers1.push_back(new ConvolutionLayer(&sig, 28, 28, 5, 5, 1, ((int)subLayers1.size()) - 1));
			subLayers1.push_back(new ConvolutionLayer(&linear, 28, 28, 5, 1, 1, ((int)subLayers1.size()) - 1));
			subLayers1.push_back(new ConvolutionLayer(&linear, 24, 28, 1, 5, 1, ((int)subLayers1.size()) - 1));
			subLayers1.push_back(new AbsLayer(24, 24, ((int)subLayers1.size()) - 1));
			//subLayers1.push_back(new NormalizationLayer(24, 24, 9, ((int)subLayers1.size()) - 1));
			subLayers1.push_back(new AveragePoolingLayer(24, 24, 2, 2, 2, ((int)subLayers1.size()) - 1));
			//subLayers1.push_back(new Layer(subLayers1[((int)subLayers1.size()) - 1]->size(), 
			//												10, &sig, ((int)subLayers1.size()) - 1));


			vector<vector<double> > train_features,test_features;
			SubNetworkLayer featuresNet(subLayers1, 16, -1);
			for (int i = 0; i < MINST::trainImages.size(); i++)
			{
				featuresNet.operate(MINST::trainImages[i]);
					train_features.push_back(featuresNet.getOutputs());
			}
			for (int i = 0; i < MINST::testImages.size(); i++)
			{
				featuresNet.operate(MINST::testImages[i]);
				test_features.push_back(featuresNet.getOutputs());
			}

			vector<Layer*> layers;
			//layers.push_back(new SubNetworkLayer(subLayers1, 10, -1));
			//layers.push_back(new Layer(layers[0]->size(), 100, &sig, 0));
			//layers.push_back(new Layer(layers[1]->size(), 10, &sig, 1));
			layers.push_back(new Layer(train_features[0].size(), 50, &sig, -1));
			layers.push_back(new Layer(layers[0]->size(), 10, &sig, 0));

			//NeuralNetwork nn(MINST::trainImages[0].size(), layers, &sig, 1);
			//NeuralNetwork nn(MINST::trainImages[0].size(), layers);
			NeuralNetwork nn(train_features[0].size(), layers);

			//nn.SGD(&MINST::trainImages, MINST::trainLabels, 50000, 10, eeta, 30, 5.0, 1, "NN_conv_");
			//nn.SGD(&MINST::trainImages, MINST::trainLabels, 60000, 10, eeta, 3, 5.0, 0, "NN_conv_1_");
			//nn.test(&MINST::testImages, MINST::testLabels);
			nn.SGD(train_features, MINST::trainLabels, 50000, 10, eeta, 60, 5.0, 1, "NN_conv_");
			nn.SGD(train_features, MINST::trainLabels, 60000, 10, eeta, 1, 5.0, 0, "NN_conv_1_");
			nn.test(test_features, MINST::testLabels);
		}
	}
}

void ensembelSearhMINST()
{
	//freopen("Iter100_nodes100.log", "w", stdout);

	MINST::readTrainLabels("MINST/train-labels.idx1-ubyte");
	MINST::readTestLabels("MINST/t10k-labels.idx1-ubyte");
	MINST::readTrainImages("MINST/train-images.idx3-ubyte");
	MINST::readTestImages("MINST/t10k-images.idx3-ubyte");
	
	for (int hiddenLayer = 50; hiddenLayer <= 50; hiddenLayer += 10)
	{
		vector<int> layers;
		layers.push_back(hiddenLayer); // hidden layer
		layers.push_back(10); // output layer
		EnsembleNN enn;
		enn.test(5, layers, MINST::trainImages, MINST::trainLabels,
			60000, 1000, 0.001, 2, 5, 1, MINST::testImages, MINST::testLabels);
	}
}


int main()
{
	//vector<vector<double> > mat(2,vector<double>(3,1));
	//vector<double> vec(3,1);
	//vector<double> res = mulMatVec(mat, vec);
	//for(int i = 0;i < res.size();i++)
	//{
	//	cout << res[i] << endl;
	//}
	//
	//test2D();
	//test10();
	//gridSearhMINST();
	//testConvolutioanl();
	testConvolutioanl_1layer();
	//testMINST();
	//ensembelSearhMINST();
	//vector<int> tmp(784, 0);
	//cout << getNumber(tmp) << endl;
}
