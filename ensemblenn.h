#include "neuralnetwork.h"

class EnsembleNN
{
public:
	void test(int n,vector<int> layers, const vector<vector<double> >& trainData, const vector<vector<short> >& trainLabels,
		int trainCount, int batchSize,double eeta,int iterations,double lambda, int test,
		const vector<vector<double> >& testData, const vector<vector<short> >& testLabels)
	{
		SigmoidFunction sig;
		vector<NeuralNetwork*> nns(n,0);
		string names[] = {
			"Ensemble0_50000_1000_0.001000_48_5.000000__50_10",
			"Ensemble1_50000_1000_0.001000_48_5.000000__50_10",
			"Ensemble2_50000_1000_0.001000_48_5.000000__50_10",
			"Ensemble3_50000_1000_0.001000_48_5.000000__50_10",
			"Ensemble4_50000_1000_0.001000_48_5.000000__50_10"
		};
		for (int i = 0; i < n; i++)
		{
			// load the networks
			//nns[i] = new NeuralNetwork(trainData[0].size(), layers, &sig, 1);
			nns[i] = new NeuralNetwork(names[i],&sig);
		}
		// train
		for (int i = 0; i < iterations; i++)
		{
			for (int j = 0; j < n; j++)
			{
				nns[j]->SGD(trainData, trainLabels, trainCount, batchSize, eeta, 1, lambda, 0,"");
			}
			// test on the validataion
			if (test)
			{
				int countCorrect = 0;
				for (int k = 0; k < testData.size(); k++)
				{
					vector<int> countVec(testLabels[k].size(),0);
					for (int j = 0; j < n; j++)
					{
						countVec[nns[j]->getOutputOneImage(testData[k])]++;
					}
					int maxtInd = distance(countVec.begin(), max_element(countVec.begin(), countVec.end()));
					int correctInd = distance(testLabels[k].begin(), max_element(testLabels[k].begin(), testLabels[k].end()));
					if (maxtInd == correctInd)
					{
						countCorrect++;
					}
				}
				cout << countCorrect << "/" << testData.size()  << endl;
			}
		}
		
		for(int i = 0;i < nns.size();i++)
		{
			// save the networks
			ostringstream ostr;
			ostr.precision(6);
			ostr << "Ensemble" << i << "_" << trainCount << "_" << batchSize << "_";
			ostr << fixed << eeta << "_" << iterations << "_" << fixed << lambda << "_";
			nns[i]->save(ostr.str());
			// delete networks
			delete nns[i];
		}
	}
};