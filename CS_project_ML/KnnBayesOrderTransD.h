#ifndef KNNBAYESORDERTRANSD_H
#define KNNBAYESORDERTRANSD_H
#define THREAD_NUM 8

#include "TransD.h"
#include <thread>

class KnnBayesOrderTransD : public TransD
{
private:
	int k;
	int round_limit;
	vector<vector<double>> dis_matrix;
	vector<MyData> X;
	vector<MyData> T;
	vector<MyData> total_data;
	void predict_thread(int n, KNNClassifier knn);
	vector<int> sort_by_density(vector<MyData>& tmp);
public:
	KnnBayesOrderTransD(vector<MyData> &X, vector<MyData> &T, int k);
	void performTrans(vector<vector<vector<double>>> &dis_matrixs, vector<int> &knn_results);
};

#endif
