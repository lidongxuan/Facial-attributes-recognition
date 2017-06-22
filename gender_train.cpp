//
// Created by lidongxuan on 2017/5/7.
//

#include "gender_train.h"

using namespace std;
using namespace cv;

void read_txt_gender(String &csvPath, vector<String> &trainPath, vector<int> &label, char separator = ';')
{
	string line, path, classLabel;
	ifstream file(csvPath.c_str(), ifstream::in);
	while (getline(file, line))
	{
		stringstream lines(line);
		getline(lines, path, separator);
		getline(lines, classLabel);
		if (!path.empty() && !classLabel.empty())
		{
			trainPath.push_back(path);
			label.push_back(atoi(classLabel.c_str()));
		}
	}
}

void gender_train()
{
	//批量读入训练样本路径
	string trainCsvPath = "./data/Gender_CAS_PEAL/train/at.txt";
	vector<String> vecTrainPath;
	vector<int> vecTrainLabel;
	int iNumTrain = (int)vecTrainLabel.size();

	read_txt_gender(trainCsvPath, vecTrainPath, vecTrainLabel);

	//初始化训练数据矩阵

	Mat trainDataHog;
	Mat trainLabel;
	trainLabel = Mat::zeros(iNumTrain, 1, CV_32FC1);

	//提取HOG特征，放入训练数据矩阵中
	Mat imageSrc;
	for (int i = 0; i < iNumTrain; i++)
	{
		imageSrc = imread(vecTrainPath[i].c_str(), 0);
		resize(imageSrc, imageSrc, Size(64, 64));
		//输入参数分别是：窗口大小，block size，block stride，cell size，360度内分隔的梯度方向数
		HOGDescriptor *hog = new HOGDescriptor(cvSize(64, 64), cvSize(16, 16),
			cvSize(8, 8), cvSize(8, 8), 9);
		vector<float> descriptor;
		//imageSrc代表输入的图片，descriptors表示保存特征结果的Vector，Size(1,1)表示windows的步进，第四个为padding，用于填充图片以适应大小
		hog->compute(imageSrc, descriptor, Size(1, 1), Size(0, 0));

		if (i == 0)
		{
			trainDataHog = Mat::zeros(iNumTrain, descriptor.size(), CV_32FC1);
		}

		// 将特征存入trainDataHog
		int n = 0;
		for (vector<float>::iterator iter = descriptor.begin(); iter != descriptor.end(); iter++)
		{
			trainDataHog.at<float>(i, n) = *iter;
			n++;
		}
		//存入label
		trainLabel.at<float>(i, 0) = vecTrainLabel[i];
	}

	//初始化SVM分类器
	CvSVM svm;
	CvSVMParams param;
	CvTermCriteria criteria;
	criteria = cvTermCriteria(CV_TERMCRIT_EPS, 1000, FLT_EPSILON);
	param = CvSVMParams(CvSVM::C_SVC, CvSVM::RBF,
		10.0, 0.09, 1.0, 10.0, 0.5, 1.0, NULL, criteria);

	//训练并保存SVM
	svm.train(trainDataHog, trainLabel, Mat(), Mat(), param);
	svm.save("./sources/gender_model.txt");
	svm.save("../Release/sources/gender_model.txt");
}