#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace std;

// reading MINST dataset
class MINST
{
	static unsigned char readByte(ifstream& inp)
	{
		char buff[2];
		inp.read(buff,1);
		unsigned char num = buff[0];
		return num; // make sure there is no negative extension
	}
	static int readInt(ifstream& inp)
	{
		char buff[5];
		inp.read(buff,4);
		int num = buff[0] & 0xff;
		for(int i = 1;i < 4;i++)
		{
			num = (num << 8) | (buff[i]&0xff); 
		}
		return num;
	}
	static vector<vector<short> > readLabelFile(string fileName)
	{
		ifstream inp(fileName, std::ifstream::binary);
		// read the magic number
		int magic = readInt(inp);
		cout << "Magic = " << magic << endl;
		// read num of labels
		int labelsCnt = readInt(inp);
		vector<vector<short> > labels(labelsCnt, vector<short>(10,0));
		for(int i = 0;i < labelsCnt;i++)
		{
			labels[i][readByte(inp)] = 1;
		}
		inp.close();
		/*
		cout << "Number of labels = " << labels.size() << endl;
		vector<int> count(10, 0);
		for (int i = 0; i < labels.size(); i++)
		{
			int d = distance(labels[i].begin(), max_element(labels[i].begin(), labels[i].end()));
			count[d]++;
		}
		int total = 0;
		for (int i = 0; i < 10; i++)
		{
			cout << i << "\t" << count[i] << endl;
			total += count[i];

		}
		cout << "Total = " << total << endl;
		*/
		return labels;
	}
	static vector<vector<double> > readImagesFile(string fileName)
	{
		ifstream inp(fileName, std::ifstream::binary);
		// read the magic number
		int magic = readInt(inp);
		cout << "Images File Magic = " << magic << endl;
		// read num of labels
		int imagesCnt = readInt(inp);
		vector<vector<double> > images(imagesCnt, vector<double>());
		// read rows and columns
		int rows = readInt(inp);
		int cols = readInt(inp);
		//cout << i << "\trows,cols = " << rows << "\t" << cols << endl;
		char * buff = new char[rows*cols*imagesCnt + 1];
		
		inp.read(buff, rows*cols*imagesCnt);
		int j = 0;
		for(int i = 0;i < imagesCnt;i++)
		{
			for(int r = 0;r < rows;r++)
			{
				for(int c = 0;c < cols;c++)
				{
					short tmp = (buff[j++] & 0xff);// readByte(inp);
					//cout << tmp << "\t";
					images[i].push_back(tmp);
				}
			}
			//puts("");
			//cout << images[i][400] << endl;
		}
		delete[]buff;
		inp.close();
		return images;
	}
public:
	static vector<vector<short> > trainLabels, testLabels;
	static vector<vector<double> > trainImages, testImages;
	// ----------------------------------------
	static void readTrainImages(string fileName)
	{
		trainImages = readImagesFile(fileName);
		cout << "Train Images = " << trainImages.size() << endl;
	}
	static void readTestImages(string fileName)
	{
		testImages = readImagesFile(fileName);
		cout << "Test Images = " << testImages.size() << endl;
	}
	static void readTrainLabels(string fileName)
	{
		trainLabels = readLabelFile(fileName);
		cout << "Train Labels = " << trainLabels.size() << endl;
	}
	static void readTestLabels(string fileName)
	{
		testLabels = readLabelFile(fileName);
		cout << "Test Labels = " << testLabels.size() << endl;		
	}
};

vector<vector<short> > MINST::trainLabels, MINST::testLabels;
vector<vector<double> > MINST::trainImages, MINST::testImages;
