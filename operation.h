#pragma once

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <map>
#include <cassert>

using namespace std;

void pause()
{
	std::cin.ignore( std::numeric_limits<std::streamsize>::max(), '\n' );
}

// elementwise subtraction
// assume the sizes are equal
template<typename T,typename S>
vector<double> sub(const vector<T>& v1,const vector<S>& v2)
{
	//cout << v1.size() << " = " << v2.size() << endl;
	//pause();
	assert(v1.size() == v2.size());
	
	vector<double> result;
	for(unsigned int i = 0;i < v1.size();i++)
	{
		result.push_back(v1[i] - v2[i]);
	}
	return result;
}

// elementwise addition
// assume the sizes are equal
template<typename T,typename S>
vector<double> add(const vector<T>& v1,const vector<S>& v2)
{
	assert(v1.size() == v2.size());
	
	vector<double> result = v1;
	int v1Size = v1.size();
#pragma omp parallel for
	for(int i = 0;i < v1Size;i++)
	{
		result[i] += v2[i];
	}
	return result;
}

// elementwise multiplication
// assume the sizes are equal
template<typename T,typename S>
vector<double> mul(const vector<T>& v1,const vector<S>& v2)
{
	//cerr << "v1 size = " << v1.size() << "v2 size = " << v2.size() << endl;
	assert(v1.size() == v2.size());
	
	vector<double> result;
	for(unsigned int i = 0;i < v1.size();i++)
	{
		result.push_back(v1[i] * v2[i]);
	}
	return result;
}


// multiply matrix by vector
vector<double> mulMatVec(const vector< vector<double> >& matrix,const vector<double> & vec)
{
	assert(matrix.size() == vec.size());
	
	vector<double> result(matrix[0].size(), 0);
	double tmpRes;
	for(unsigned int i = 0;i < matrix[0].size();i++)
	{
		for(unsigned int j = 0;j < vec.size();j++)
		{
			tmpRes = matrix[j][i] * vec[j];
			result[i] += tmpRes;
		}
	}
	return result;
}

// multiply two vectors to get matrix
vector<vector<double> > mulVecVec(const vector<double>& v1,const vector<double> & v2)
{
	vector<vector<double> > result(v1.size(), vector<double>(v2.size(),0));
	for(unsigned int i = 0;i < v1.size();i++)
	{
		for(unsigned int j = 0;j < v2.size();j++)
		{
			result[i][j] = v1[i] * v2[j];
		}
	}
	return result;
}
