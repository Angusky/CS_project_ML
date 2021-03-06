#include"KnnBayesOrderTransD.h"
#include<time.h>
#include<stdlib.h>
#include<random>


KnnBayesOrderTransD::KnnBayesOrderTransD(vector<MyData> &X, vector<MyData> &T, int k) {
	this->k = k;
	this->X = X;
	this->T = T;
	round_limit = 20;
	//generate distance matrix
	total_data = X;
	total_data.insert(total_data.end(), T.begin(), T.end());
	genDismatrix(total_data, dis_matrix);
	//set initial knn_label and class_weight
	for (int i = 0; i < total_data.size(); i++) {
		if (i >= X.size()) {
			total_data[i].knn_label = total_data[i].label;
			total_data[i].class_w = 1;
			total_data[i].class_w_table.push_back(pair<int, double>(0, 0));
		}
	}
}

bool compfunc_density(pair<int, MyData> a, pair<int, MyData> b)
{
	return a.first > b.first;
}

bool compfunc_dis(double a, double b)
{
	return a < b;
}

vector<int> KnnBayesOrderTransD::sort_by_density(vector<MyData>& tmp)
{
	double radius = 0;
	srand(time(NULL));
	int rnd = rand() % tmp.size();
	vector<double> dis_vector_rnd(dis_matrix[rnd+X.size()].begin(), dis_matrix[rnd+X.size()].end());
	sort(dis_vector_rnd.begin(), dis_vector_rnd.end(), compfunc_dis);
	radius = dis_vector_rnd[total_data.size() / 10];
	vector<pair<int,MyData>> density;
	for (int i = 0; i < tmp.size(); i++)
	{
		vector<double> dis_vector(dis_matrix[i+X.size()].begin(), dis_matrix[i + X.size()].end());
		int d_count = 0;
		for (int j = 0; j < dis_vector.size(); j++)
		{
			if (dis_vector[j] <= radius)
			{
				d_count++;
			}
		}
		density.push_back(pair<int,MyData>(d_count,tmp[i]));
	}
	sort(density.begin(), density.end(), compfunc_density);
	vector<int> order_index;
	for (int i = 0; i < density.size(); i++)
	{
		int idx = -1;
		for (int j = 0; j < total_data.size(); j++)
		{
			if (density[i].second.num == total_data[j].num)
			{
				idx = j;
				break;
			}
		}
		if (idx != -1)
		{
			order_index.push_back(idx);
		}
	}
	//debugging
	/*for (int i = 0; i < order_index.size(); i++)
	{
		cout << order_index[i] << "a ";
	}
	system("pause");*/
	return order_index;
}

void KnnBayesOrderTransD::predict_thread(int n, KNNClassifier knn) {
	//get knn class weight and label
	vector<MyData> tmp;
	for (int i = X.size(); i < total_data.size(); i++)
	{
		tmp.push_back(total_data[i]);
	}
	vector<int> order_index = sort_by_density(tmp);
	for (int i = X.size(); i < total_data.size(); i++) {
		if (i % THREAD_NUM == n) {
			vector<double> dis_vector(dis_matrix[order_index[i - X.size()]].begin(), dis_matrix[order_index[i - X.size()]].end());
			total_data[order_index[i-X.size()]].knn_label = knn.bayesprediction_order(total_data[order_index[i- X.size()]], dis_vector);
		}
	}
}
void KnnBayesOrderTransD::performTrans(vector<vector<vector<double>>> &dis_matrixs, vector<int> &knn_results) {

	double v = 0.1;

	KNNClassifier one_nn(X, 1);
	NMIClassifier one_mi(X, dis_matrix, 1);

	vector<vector<double>> tmpdis;

	dis_matrixs.push_back(dis_matrix);
	knn_results.resize(T.size());
	tmpdis = dis_matrix;

	for (int rc = 0; rc < round_limit; rc++) {
		double lambda = 1, epsilon;
		double r = 0.5;
		double w = 1.05;

		//cout << "Round " << rc + 1 << " ." << endl;

		//create thread
		vector<thread> threads;
		vector<vector<pair<int, double>>> tmpknn_result;
		KNNClassifier knn(total_data, k);
		for (int i = 0; i < THREAD_NUM; i++) {
			threads.push_back(thread(&KnnBayesOrderTransD::predict_thread, this, i, knn));
		}
		//join thread
		for (int i = 0; i < THREAD_NUM; i++) {
			threads[i].join();
		}
		//set bayes knn results
		for (int i = X.size(); i < total_data.size(); i++) {
			knn_results[i - X.size()] = total_data[i].knn_label;
		}
		//for each pair, calculate new dis
		for (int i = 0; i < total_data.size(); i++) {
			for (int j = i + 1; j < total_data.size(); j++) {
				double f = 1;
				//
				/*if (total_data[i].is_train || total_data[j].is_train) {
				lambda = 1;
				}
				else {
				lambda = 0.5;
				}*/
				//
				epsilon = lambda * total_data[i].class_w * total_data[j].class_w;
				if (r <= epsilon) {
					if (lambda == 0.5) {
						cout << total_data[i].class_w << " " << total_data[j].class_w << endl;
					}
					//change dis
					f = 1.05;
					if (total_data[i].knn_label == total_data[j].knn_label) {
						f = 1 / f;
					}
				}
				tmpdis[i][j] = dis_matrix[i][j] * f;
				tmpdis[j][i] = tmpdis[i][j];
			}
		}
		dis_matrix = tmpdis;
		//record distance matrixs
		dis_matrixs.push_back(dis_matrix);

		//verify 1-nn and 1mi
		int knn_result, nmi_result;
		bool check_flag = true;

		one_mi.setDisMatrix(dis_matrix);
		//cout << "1-nn  nmi" << endl;
		for (int i = 0; i < T.size(); i++) {
			vector<double> dis_vector(dis_matrix[X.size() + i].begin(), dis_matrix[X.size() + i].begin() + X.size());
			knn_result = one_nn.prediction(T[i], dis_vector);
			nmi_result = one_mi.prediction(T[i], dis_vector);
			//cout << knn_result << "    " << nmi_result << endl;
			if (knn_result != nmi_result) {
				check_flag = false;
				//cout << "T[" << i << "] fail, 1nn = " << knn_result << ", 1mi = " << nmi_result << endl;
				break;
			}
		}

		//output class weight
		/*string title = "training_label" + to_string(rc + 1) + ".txt";
		ofstream out(title);
		double *beauty_weight = new double[total_data.back().class_w_table.size()];
		for (int j = 0; j < total_data.back().class_w_table.size(); j++)
		{
		beauty_weight[j] = 0;
		}
		vector<MyData>sorted_data = total_data;
		sort(sorted_data.begin(), sorted_data.end(), mycompindex);
		for (int i = 0; i < sorted_data.size(); i++) {
		sort(sorted_data[i].class_w_table.begin(), sorted_data[i].class_w_table.end(), mycomp2);
		for (int j = 0; j < sorted_data[i].class_w_table.size(); j++)
		{
		beauty_weight[sorted_data[i].class_w_table[j].first] = sorted_data[i].class_w_table[j].second;
		}
		for (int j = 0; j <sorted_data.back().class_w_table.size(); j++) {
		out << fixed << setprecision(6) << j << "," << beauty_weight[j] << "\t";
		beauty_weight[j] = 0;
		}
		out << endl;
		}
		out.close();
		*/
		if (check_flag) {
			//cout << "KnnBayesTransD done by 1-NN and 1mi match." << endl;
			break;
		}
	}
}
