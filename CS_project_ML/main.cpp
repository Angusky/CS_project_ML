#include<iostream>
#include"KnnBayesSemi.h"
#include"NonlinearSemi.h"
#include"Utility.h"
#include"MyData.h"
#include"ClusterSemi.h"


using namespace std;
int main() {
	string datalist[] = { "a200",
		"breast_s",
		"cleveland_s",
		"d1_s","d2_s","d3_s","d4_s","d5_s","d6_s","d7_s",
		"diabetes_s",
		"ecoli_s",
		"heart_s",
		"hungarian_s",
		"imox_s",
		"seeds_s",
		"sonar_s",
		"SPECTF_s",
		"Faults_s" };
	string prefix = "D:\\Desk\\Junior\\CS_project\\testData2\\";
	string folder = "D:\\Desk\\Junior\\CS_project\\Clean\\CS_project_ML-steven\\CS_project_ML\\plot\\matrix\\";
	int k = 1;
	int fold_num = 50;

	//------------------------
	for (int j = 0; j < 20; j++) {

		string dataname = prefix + datalist[j] + ".data";
		string labeldir = prefix + datalist[j] + "\\label_encoded";
		ofstream expout(datalist[j] + "_nonlinear.txt");
		ofstream inverseout(datalist[j] + "_inverse.txt");
		for (int i = 1; i <= fold_num; i++) {

			vector<MyData> X;
			vector<MyData> XT; //for semi-supervised
			vector<MyData> T;
			vector<int> result;
			vector<vector<double>> new_dis;

			string labelname = labeldir + to_string(i / 10) + to_string(i % 10) + ".txt";
			extractData(X, XT, T, labelname);
			//extractData(X, T, dirname, i);

			//SemiTransD
			KnnBayesSemi stransd(X, XT, k);
			stransd.setT(T);
			stransd.performTrans();
			inverseout << stransd.getScore() << endl;


			//ClusterSemi
			/*if (i == 1)
			{
			CreateFolder(folder + datalist[j] + "\\");
			}
			ClusterSemi Cstransd(X, XT, k, folder+datalist[j]+"\\",i==1);
			Cstransd.setT(T);
			Cstransd.performTrans();
			expout << Cstransd.getScore() << endl;
			if (i == 1)
			{
			Cstransd.printSortedMatrixs();

			}*/

			//AffineSemi
			/*NonlinearSemi ntransd(X, XT, k);
			ntransd.setT(T);
			ntransd.performTrans();
			expout << ntransd.getScore() << endl;*/

			//TransD		
			/*TransD transd(X, T, k);
			transd.performTrans(new_dis);
			transd.getSortedMatrix(new_dis);
			printDismatrix(new_dis);*/

			//NMI
			/*NMIClassifier nmi(X, 1);
			result = nmi.prediction(T);*/

			//printing result
			/*wrong_count = checkResult(result, T);
			validation_err += wrong_count;
			accuracy = (double)(T.size() - wrong_count) / (double)T.size() * 100;
			cout << "Fold " << i << " done with accuracy " << accuracy << "%" << endl << endl;*/

			//addtional testing
			/*if (i == 1) {
			int vsize = result.size();
			for (int j = 0; j < vsize; j++) {
			if (result[j] != T[j].label) {
			cout << "Data No." << T[j].num << " should be " << T[j].label << endl;
			cout << "But labeled as " << result[j] << endl;
			}
			}
			}*/
	
			X.clear();
			T.clear();
			XT.clear();

		}
	}
	cout << "Data analyzing done." << endl;
	system("pause");

	return 0;

	

	return 0;
}
